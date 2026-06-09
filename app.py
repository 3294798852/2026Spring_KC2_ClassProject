import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
import importlib
import hashlib
import base64

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFilter

from src.foreground import (
    apply_manual_alpha_mask,
    fit_foreground_to_background,
    refine_rgba_cutout,
    remove_background,
    resize_foreground,
)
from src.image_preprocess import resize_by_long_edge
from src.infer import rank_candidates_heatmap_guided, score_single_position
from src.reference_opa import ReferenceOPAScorer, ensure_simopa_weight
from src.user_feedback import analyze_candidate, spread_summary


def _patch_streamlit_canvas_compat() -> None:
    """
    streamlit-drawable-canvas may depend on `streamlit.elements.image.image_to_url`,
    which is removed in newer streamlit versions. Patch it from image_utils.
    """
    try:
        st_image_mod = importlib.import_module("streamlit.elements.image")
        image_utils_mod = importlib.import_module("streamlit.elements.lib.image_utils")
        if not hasattr(image_utils_mod, "image_to_url"):
            return
        target = getattr(st_image_mod, "image_to_url", None)
        # If legacy API already exists with old signature, keep it.
        if callable(target):
            try:
                import inspect

                param_count = len(inspect.signature(target).parameters)
                if param_count == 6:
                    return
            except Exception:
                pass

        # Build old-signature shim expected by streamlit-drawable-canvas.
        from types import SimpleNamespace

        new_impl = getattr(image_utils_mod, "image_to_url")

        def _legacy_image_to_url(image, width, clamp, channels, output_format, image_id):
            layout = SimpleNamespace(width=width)
            return new_impl(image, layout, clamp, channels, output_format, image_id)

        setattr(st_image_mod, "image_to_url", _legacy_image_to_url)
    except Exception:
        # Keep silent; fallback logic will handle canvas degradation.
        return


_patch_streamlit_canvas_compat()

try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore[reportMissingImports]

    HAS_DRAWABLE_CANVAS = True
except Exception:
    HAS_DRAWABLE_CANVAS = False


st.set_page_config(page_title="方向A-物体放置助手", layout="wide")
st.title("方向 A：智能物体放置与质量评分（本地推理）")
st.caption("主线模型：BCMI/libcom 的 SimOPA（已移除 legacy 路径）。")
expected_device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"预估推理设备：`{expected_device}`（启动推理前显示）")


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_heatmap_overlay(bg: Image.Image, heatmap: np.ndarray) -> Image.Image:
    hm = np.asarray(heatmap, dtype=np.float32)
    # Robust normalization to improve visibility when scores are tightly clustered.
    p5 = float(np.percentile(hm, 5))
    p95 = float(np.percentile(hm, 95))
    if p95 - p5 < 1e-6:
        # Fallback to min/max; if still flat, paint neutral medium intensity.
        hmin = float(hm.min())
        hmax = float(hm.max())
        if hmax - hmin < 1e-6:
            hm_norm = np.full_like(hm, 0.5, dtype=np.float32)
        else:
            hm_norm = (hm - hmin) / (hmax - hmin + 1e-8)
    else:
        hm_norm = np.clip((hm - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)

    hm_img = Image.fromarray((hm_norm * 255).astype(np.uint8), mode="L").resize(
        bg.size, Image.Resampling.BILINEAR
    )
    hm_arr = np.asarray(hm_img, dtype=np.float32) / 255.0

    # Build pseudo-color map (blue->cyan->green->yellow->red).
    # r rises with intensity; b drops with intensity; g peaks mid-range.
    r = np.clip(1.5 * hm_arr - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * hm_arr - 1.0) * 1.8, 0.0, 1.0)
    b = np.clip(1.2 - 1.6 * hm_arr, 0.0, 1.0)
    heat_rgb = np.stack([r, g, b], axis=-1) * 255.0

    bg_arr = np.asarray(bg.convert("RGB"), dtype=np.float32)
    # Ensure visible overlay even for low-contrast maps.
    alpha = np.clip(0.20 + 0.55 * hm_arr, 0.20, 0.75)[..., None]
    overlay = bg_arr * (1.0 - alpha) + heat_rgb * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _draw_topk_markers(image: Image.Image, ranked: list[dict], fg_w: int, fg_h: int) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(ranked):
        x = int(row["x"] + fg_w / 2)
        y = int(row["y"] + fg_h / 2)
        r = max(6, min(fg_w, fg_h) // 10)
        color = (255, 255, 0) if i == 0 else (0, 255, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        draw.text((x + r + 2, y - r - 2), f"#{i+1}", fill=color)
    return out


def _build_export_zip(
    ranked: list[dict],
    images: list[Image.Image],
    heatmap_overlay_with_marks: Image.Image | None,
    raw_heatmap: np.ndarray | None,
) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ranked": ranked,
        }
        if raw_heatmap is not None:
            meta["heatmap_stats"] = {
                "min": float(raw_heatmap.min()),
                "max": float(raw_heatmap.max()),
                "gap": float(raw_heatmap.max() - raw_heatmap.min()),
            }
        zf.writestr("ranking.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for i, img in enumerate(images):
            zf.writestr(f"topk/top_{i+1}.png", _image_to_png_bytes(img))
        if heatmap_overlay_with_marks is not None:
            zf.writestr("heatmap/overlay_with_topk.png", _image_to_png_bytes(heatmap_overlay_with_marks))
    return payload.getvalue()


def _odd(v: int) -> int:
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1


def _get_canvas_mask(
    canvas_image_data: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract keep/erase marks from canvas image.
    Robust to anti-aliasing and alpha blending.
    """
    empty = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
    if canvas_image_data is None:
        return empty, empty
    arr = np.asarray(canvas_image_data, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return empty, empty
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    a = arr[..., 3].astype(np.int16) if arr.shape[2] > 3 else np.full_like(r, 255)

    rgb = arr[..., :3].astype(np.float32) / 255.0
    r_f = rgb[..., 0]
    g_f = rgb[..., 1]
    b_f = rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0)

    # HSV hue in degrees [0, 360)
    hue = np.zeros_like(cmax, dtype=np.float32)
    nonzero = delta > 1e-6
    idx = nonzero & (cmax == r_f)
    hue[idx] = ((g_f[idx] - b_f[idx]) / delta[idx]) % 6.0
    idx = nonzero & (cmax == g_f)
    hue[idx] = ((b_f[idx] - r_f[idx]) / delta[idx]) + 2.0
    idx = nonzero & (cmax == b_f)
    hue[idx] = ((r_f[idx] - g_f[idx]) / delta[idx]) + 4.0
    hue = (hue * 60.0) % 360.0

    alpha_ok = a > 10
    sat_ok = sat > 0.20
    keep_mark = alpha_ok & sat_ok & (hue >= 70.0) & (hue <= 170.0)   # green-ish
    erase_mark = alpha_ok & sat_ok & ((hue <= 20.0) | (hue >= 340.0))  # red-ish

    keep_img = Image.fromarray((keep_mark.astype(np.uint8) * 255), mode="L").resize(
        target_size, Image.Resampling.BILINEAR
    )
    erase_img = Image.fromarray((erase_mark.astype(np.uint8) * 255), mode="L").resize(
        target_size, Image.Resampling.BILINEAR
    )
    keep_delta = np.asarray(keep_img, dtype=np.float32) / 255.0
    erase_delta = np.asarray(erase_img, dtype=np.float32) / 255.0
    return keep_delta, erase_delta


def _compose_manual_keep_mask(
    base_keep_mask: Optional[np.ndarray],
    keep_delta: np.ndarray,
    erase_delta: np.ndarray,
    combine_mode: str,
    expand_px: int = 0,
    shrink_px: int = 0,
    feather_px: int = 1,
) -> np.ndarray:
    if base_keep_mask is None:
        base = np.zeros_like(keep_delta, dtype=np.float32)
    else:
        base = np.clip(base_keep_mask.astype(np.float32), 0.0, 1.0)
        if base.shape != keep_delta.shape:
            base = np.asarray(
                Image.fromarray((base * 255).astype(np.uint8), mode="L").resize(
                    (keep_delta.shape[1], keep_delta.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0

    keep_bin = (keep_delta > 0.10).astype(np.float32)
    base_bin = (base > 0.10).astype(np.float32)
    if combine_mode == "并集":
        keep_mask = np.maximum(base_bin, keep_bin)
    elif combine_mode == "交集":
        # If no manual keep marks, keep base unchanged (avoid accidental empty output).
        if keep_bin.sum() < 1:
            keep_mask = base_bin.copy()
        else:
            keep_mask = base_bin * keep_bin
    elif combine_mode == "仅手工":
        keep_mask = keep_bin
    else:  # 仅智能
        keep_mask = base_bin

    erase_bin = (erase_delta > 0.10).astype(np.float32)
    keep_mask = keep_mask * (1.0 - erase_bin)
    keep_mask = np.clip(keep_mask, 0.0, 1.0)

    mask_img = Image.fromarray((keep_mask * 255).astype(np.uint8), mode="L")
    if expand_px > 0:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=_odd(expand_px)))
    if shrink_px > 0:
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=_odd(shrink_px)))
    if feather_px > 1:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def _uploaded_file_id(uploaded_file) -> str:
    if uploaded_file is None:
        return "none"
    payload = uploaded_file.getvalue()
    digest = hashlib.md5(payload).hexdigest()[:10]
    return f"{uploaded_file.name}:{len(payload)}:{digest}"


def _safe_canvas(
    fg_show: Image.Image,
    show_size: tuple[int, int],
    brush: int,
    stroke_color: str,
    fill_color: str,
    draw_mode: str,
    canvas_key: str,
):
    """
    Try background-image canvas first. If incompatible with Streamlit internals,
    fallback to plain transparent canvas with side-by-side reference image.
    """
    try:
        return st_canvas(
            fill_color=fill_color,
            stroke_width=brush,
            stroke_color=stroke_color,
            background_image=fg_show,
            update_streamlit=True,
            width=show_size[0],
            height=show_size[1],
            drawing_mode=draw_mode,
            key=canvas_key,
        ), False
    except Exception as exc:
        # streamlit_drawable_canvas older versions may call removed internals
        # (e.g. streamlit.elements.image.image_to_url). Fallback still supports
        # manual mask drawing and keeps the feature usable.
        st.warning(f"画布背景兼容失败，已切换兼容模式：{exc}")
        c_left, c_right = st.columns(2)
        with c_left:
            st.image(fg_show, caption="参考前景图（在右侧画布按同位置涂抹）", width="stretch")
        with c_right:
            canvas = st_canvas(
                fill_color=fill_color,
                stroke_width=brush,
                stroke_color=stroke_color,
                background_color="rgba(0, 0, 0, 0)",
                update_streamlit=True,
                width=show_size[0],
                height=show_size[1],
                drawing_mode=draw_mode,
                key=f"{canvas_key}_compat",
            )
        return canvas, True


def _get_cached_auto_seed_rgba(
    fg_preview: Image.Image,
    fg_source_id: str,
    cutout_target: str,
    show_spinner: bool = False,
) -> Image.Image | None:
    seed_cache_key = f"{fg_source_id}|{cutout_target}"
    if st.session_state.get("manual_auto_seed_cache_key") == seed_cache_key:
        return st.session_state.get("manual_auto_seed_rgba")
    try:
        if show_spinner:
            with st.spinner("计算智能抠图预览..."):
                auto_seed_rgba, _ = remove_background(fg_preview, target=cutout_target)
        else:
            auto_seed_rgba, _ = remove_background(fg_preview, target=cutout_target)
        st.session_state["manual_auto_seed_rgba"] = auto_seed_rgba
        st.session_state["manual_auto_seed_cache_key"] = seed_cache_key
        return auto_seed_rgba
    except Exception:
        st.session_state["manual_auto_seed_rgba"] = None
        st.session_state["manual_auto_seed_cache_key"] = seed_cache_key
        return None


def _show_small_rgba_preview(image: Image.Image, caption: str, max_w: int = 420) -> None:
    show_w = min(max_w, image.size[0])
    ratio = show_w / max(1, image.size[0])
    show_h = max(1, int(image.size[1] * ratio))
    preview = image.resize((show_w, show_h), Image.Resampling.BILINEAR)
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.image(preview, caption=caption, width="stretch")
    with col_r:
        st.caption(f"预览尺寸：{show_w}x{show_h}")


def _pil_rgba_to_data_url(image: Image.Image, max_side: int = 1200) -> str:
    img = image.convert("RGBA")
    w, h = img.size
    side = max(w, h)
    if side > max_side:
        scale = max_side / float(side)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_image_xy_from_canvas(
    json_data: dict | None,
    default_x: int,
    default_y: int,
    max_x: int,
    max_y: int,
) -> tuple[int, int]:
    x = int(default_x)
    y = int(default_y)
    try:
        if not json_data or "objects" not in json_data or len(json_data["objects"]) == 0:
            return x, y
        obj = json_data["objects"][0]
        x = int(round(float(obj.get("left", x))))
        y = int(round(float(obj.get("top", y))))
    except Exception:
        pass
    x = max(0, min(max_x, x))
    y = max(0, min(max_y, y))
    return x, y


def _extract_transform_from_canvas(
    json_data: dict | None,
    default_x: int,
    default_y: int,
    default_scale: float,
    fg_w: int,
    fg_h: int,
    max_x: int,
    max_y: int,
) -> tuple[int, int, float]:
    x = int(default_x)
    y = int(default_y)
    scale = float(default_scale)
    try:
        if not json_data or "objects" not in json_data or len(json_data["objects"]) == 0:
            return x, y, scale
        obj = json_data["objects"][0]
        left = float(obj.get("left", x))
        top = float(obj.get("top", y))
        width = float(obj.get("width", fg_w))
        height = float(obj.get("height", fg_h))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        box_w = max(8.0, width * scale_x)
        box_h = max(8.0, height * scale_y)
        # map transformed box back to uniform fg scale
        s_w = box_w / max(1.0, float(fg_w))
        s_h = box_h / max(1.0, float(fg_h))
        scale = max(0.3, min(2.5, (s_w + s_h) * 0.5))
        x = int(round(left))
        y = int(round(top))
        x = max(0, min(max_x, x))
        y = max(0, min(max_y, y))
    except Exception:
        pass
    return x, y, scale

with st.sidebar:
    st.header("模型与权重")
    if st.button("下载/检查 SimOPA 参考权重"):
        with st.spinner("准备 SimOPA 权重中..."):
            try:
                path = ensure_simopa_weight()
                st.success(f"已就绪: {path}")
            except Exception as exc:
                st.error(
                    "权重下载失败。请检查网络后重试，或手动将 `SimOPA.pth` 放到 `models/` 目录。"
                )
                st.caption(str(exc))
    st.info("本版本仅保留 SimOPA 评分主线。首次建议先点击一次“下载/检查权重”。")

bg_file = st.file_uploader(
    "上传背景图 (支持 jpg/jpeg/png/webp/bmp)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    key="bg",
)
fg_file = st.file_uploader("上传前景图 (支持 jpg/jpeg/png/webp/bmp)", type=["jpg", "jpeg", "png", "webp", "bmp"], key="fg")

st.subheader("前景处理选项")
col_a, col_b, col_c = st.columns(3)
with col_a:
    cutout_mode = st.selectbox(
        "前景抠图方式",
        options=["不抠图", "一键智能抠图(U2Net)", "手工抠图(画笔)"],
        index=0,
    )
with col_b:
    cutout_target = st.selectbox("抠图目标", options=["person", "foreground"], index=0)
with col_c:
    fg_scale = st.slider("前景缩放比例", min_value=0.3, max_value=2.5, value=1.0, step=0.05)

post_cfg = {}
if cutout_mode != "不抠图":
    st.caption("抠图后处理（适用于一键抠图与手工抠图）")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        post_cfg["alpha_threshold"] = st.slider("Alpha阈值", min_value=1, max_value=80, value=12, step=1)
    with p2:
        post_cfg["keep_largest"] = st.checkbox("仅保留最大连通域", value=True)
    with p3:
        post_cfg["feather_radius"] = st.slider("后处理羽化", min_value=1, max_value=8, value=2, step=1)
    with p4:
        post_cfg["invert_mask"] = st.checkbox("反选掩码", value=False)
    q1, q2 = st.columns(2)
    with q1:
        post_cfg["auto_crop"] = st.checkbox("自动裁掉透明边框", value=True)
    with q2:
        post_cfg["crop_padding"] = st.slider("裁边留白", min_value=0, max_value=30, value=6, step=1)

st.subheader("推理预处理")
resolution_profile = st.selectbox(
    "大图自动压缩策略（按长边）",
    options=["1080P (1920)", "2K (2560)", "关闭"],
    index=0,
    help="输入分辨率过大时，先缩小再推理以显著加速。",
)

st.subheader("搜索策略")
enable_scale_search = st.checkbox("启用多尺度搜索（推荐）", value=True)
parallel_scale_search = st.checkbox(
    "并行执行多尺度搜索",
    value=False,
    disabled=not enable_scale_search,
    help="并行的是不同缩放尺度的候选搜索；单个尺度内部仍使用批量模型推理。",
)
max_cpu_compose_workers = max(1, min(8, os.cpu_count() or 4))
cpu_compose_workers = st.slider(
    "CPU候选合成线程数",
    min_value=1,
    max_value=max_cpu_compose_workers,
    value=1,
    step=1,
    help="并行加速候选图和mask的CPU合成；如果已开启多尺度并行，内部会自动减少嵌套线程。",
)
scale_offsets = st.multiselect(
    "额外尝试缩放",
    options=["-20%", "-10%", "+10%", "+20%"],
    default=["-10%", "+10%"],
    help="在当前缩放比例附近额外尝试若干尺度，提升找到合理位置的概率。",
)

top_k = st.slider("展示推荐 Top-K", min_value=1, max_value=8, value=5)
search_budget = st.slider(
    "搜索预算（候选中心数）",
    min_value=16,
    max_value=120,
    value=48,
    step=4,
    help="用于两阶段搜索的候选中心数量。数值越大越可能找到更优位置，但推理更慢。"
         "这不是热力图采样点数量。",
)
precompute_heatmap = st.checkbox("推理时同步生成热力图（无需二次推理）", value=True)
heat_grid = st.slider(
    "热力图网格密度（每边采样点）",
    min_value=8,
    max_value=40,
    value=18,
    step=2,
    help="总采样点约为 grid^2。例如 20 表示 400 个点，不是 20 个点。",
)

if cutout_mode == "手工抠图(画笔)":
    st.subheader("手工抠图面板")
    if not HAS_DRAWABLE_CANVAS:
        st.warning("未安装 `streamlit-drawable-canvas`，请执行 `pip install streamlit-drawable-canvas` 后重启。")
    elif fg_file is not None:
        fg_source_id = _uploaded_file_id(fg_file)
        st.session_state["manual_fg_source_id"] = fg_source_id
        fg_preview = Image.open(fg_file).convert("RGB")
        max_show_w = 700
        show_scale = min(1.0, max_show_w / max(1, fg_preview.size[0]))
        show_size = (int(fg_preview.size[0] * show_scale), int(fg_preview.size[1] * show_scale))
        fg_show = fg_preview.resize(show_size, Image.Resampling.BILINEAR)
        c1, c2, c3 = st.columns(3)
        with c1:
            manual_seed_mode = st.selectbox("初始化掩码", options=["空白", "智能抠图结果"], index=1)
        with c2:
            manual_brush_mode = st.selectbox("画笔模式", options=["保留(绿色)", "擦除(红色)"], index=0)
        with c3:
            draw_mode = st.selectbox("绘制方式", options=["freedraw", "polygon", "line"], index=0)

        combine_mode = st.selectbox(
            "手工与智能掩码融合",
            options=["并集", "交集", "仅手工", "仅智能"],
            index=0,
            help="并集=保留两者任一区域；交集=只保留两者共同区域；其余为单独使用。",
        )

        auto_seed_rgba = None
        if manual_seed_mode == "智能抠图结果":
            auto_seed_rgba = _get_cached_auto_seed_rgba(
                fg_preview=fg_preview,
                fg_source_id=fg_source_id,
                cutout_target=cutout_target,
                show_spinner=True,
            )
            if auto_seed_rgba is not None:
                _show_small_rgba_preview(
                    auto_seed_rgba,
                    caption="智能抠图初始化结果（供手工精修参考）",
                )
            else:
                st.warning("智能抠图初始化结果暂不可用，将以空白掩码初始化。")

        b1, b2, b3 = st.columns(3)
        with b1:
            brush = st.slider("画笔粗细", min_value=4, max_value=60, value=18, step=2)
        with b2:
            expand_px = st.slider("掩码扩张", min_value=0, max_value=15, value=2, step=1)
        with b3:
            shrink_px = st.slider("掩码收缩", min_value=0, max_value=15, value=0, step=1)
        feather_px = st.slider("边缘羽化", min_value=1, max_value=12, value=2, step=1)

        if "manual_canvas_uid" not in st.session_state:
            st.session_state["manual_canvas_uid"] = f"canvas_{time.time_ns()}"
        if st.button("清空手工画布"):
            st.session_state["manual_canvas_uid"] = f"canvas_{time.time_ns()}"
            st.session_state["manual_canvas_data"] = None
            st.session_state["manual_canvas_compat_mode"] = False
            st.session_state["manual_applied_rgba"] = None
            st.session_state["manual_applied_info"] = None
            st.rerun()

        stroke_color = "#00FF00" if manual_brush_mode.startswith("保留") else "#FF0000"
        fill_color = "rgba(0,255,0,0.22)" if manual_brush_mode.startswith("保留") else "rgba(255,0,0,0.22)"
        st.caption("绿色表示保留区域，红色表示擦除区域；可切到 polygon 画封闭曲线。")
        canvas_result, is_compat_canvas = _safe_canvas(
            fg_show=fg_show,
            show_size=show_size,
            brush=brush,
            stroke_color=stroke_color,
            fill_color=fill_color,
            draw_mode=draw_mode,
            canvas_key=st.session_state["manual_canvas_uid"],
        )
        st.session_state["manual_canvas_data"] = canvas_result.image_data
        st.session_state["manual_canvas_compat_mode"] = is_compat_canvas
        st.session_state["manual_seed_mode"] = manual_seed_mode
        st.session_state["manual_expand_px"] = expand_px
        st.session_state["manual_shrink_px"] = shrink_px
        st.session_state["manual_feather_px"] = feather_px
        st.session_state["manual_combine_mode"] = combine_mode

        apply_cols = st.columns(2)
        with apply_cols[0]:
            apply_clicked = st.button("应用手工抠图结果", type="primary")
        with apply_cols[1]:
            if st.button("丢弃已应用结果"):
                st.session_state["manual_applied_rgba"] = None
                st.session_state["manual_applied_info"] = None
                st.rerun()

        if apply_clicked:
            base_rgba = fg_preview.convert("RGBA")
            base_keep = None
            if manual_seed_mode == "智能抠图结果":
                if auto_seed_rgba is not None:
                    base_keep = (np.asarray(auto_seed_rgba, dtype=np.uint8)[..., 3] > 20).astype(np.float32)

            keep_delta, erase_delta = _get_canvas_mask(
                canvas_result.image_data,
                target_size=base_rgba.size,
            )
            keep_mask = _compose_manual_keep_mask(
                base_keep_mask=base_keep,
                keep_delta=keep_delta,
                erase_delta=erase_delta,
                combine_mode=combine_mode,
                expand_px=expand_px,
                shrink_px=shrink_px,
                feather_px=feather_px,
            )
            fg_manual = apply_manual_alpha_mask(base_rgba, keep_mask, feather=1)
            if post_cfg:
                fg_manual = refine_rgba_cutout(
                    fg_manual,
                    alpha_threshold=int(post_cfg.get("alpha_threshold", 12)),
                    keep_largest=bool(post_cfg.get("keep_largest", True)),
                    feather_radius=int(post_cfg.get("feather_radius", 2)),
                    auto_crop=bool(post_cfg.get("auto_crop", True)),
                    crop_padding=int(post_cfg.get("crop_padding", 6)),
                    invert_mask=bool(post_cfg.get("invert_mask", False)),
                )
            st.session_state["manual_applied_rgba"] = fg_manual
            st.session_state["manual_applied_info"] = (
                f"已应用手工抠图：融合={combine_mode}，保留笔迹={int((keep_delta>0.1).sum())}，"
                f"擦除笔迹={int((erase_delta>0.1).sum())}"
            )
            st.session_state["manual_applied_source_id"] = fg_source_id

        if st.session_state.get("manual_applied_rgba") is not None and st.session_state.get("manual_applied_source_id") == fg_source_id:
            _show_small_rgba_preview(
                st.session_state["manual_applied_rgba"],
                caption=st.session_state.get("manual_applied_info", "已应用手工抠图结果"),
                max_w=520,
            )
        else:
            st.caption("尚未应用手工抠图结果。请完成绘制后点击“应用手工抠图结果”。")
    else:
        st.info("请先上传前景图后再进行手工抠图。")

if cutout_mode == "一键智能抠图(U2Net)" and fg_file is not None:
    fg_source_id = _uploaded_file_id(fg_file)
    fg_preview = Image.open(fg_file).convert("RGB")
    auto_seed_rgba = _get_cached_auto_seed_rgba(
        fg_preview=fg_preview,
        fg_source_id=fg_source_id,
        cutout_target=cutout_target,
        show_spinner=False,
    )
    if auto_seed_rgba is not None:
        _show_small_rgba_preview(auto_seed_rgba, caption="一键智能抠图预览")

action_cols = st.columns([1, 1, 4])
with action_cols[0]:
    start_recommend = st.button("开始推荐", type="primary")
with action_cols[1]:
    clear_results = st.button("清除当前推荐结果")

if clear_results:
    for key in [
        "last_result",
        "last_heatmap",
        "last_heatmap_overlay",
        "manual_score_result",
        "manual_drag_x",
        "manual_drag_y",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

if start_recommend:
    if bg_file is None or fg_file is None:
        st.warning("请先上传背景图和前景图。")
    else:
        bg = Image.open(bg_file).convert("RGB")
        orig_bg_size = bg.size
        bg_resize_factor = 1.0
        if resolution_profile.startswith("1080P"):
            bg = resize_by_long_edge(bg, 1920)
        elif resolution_profile.startswith("2K"):
            bg = resize_by_long_edge(bg, 2560)
        resized_bg_size = bg.size
        bg_resize_factor = resized_bg_size[0] / max(1, orig_bg_size[0])

        raw_fg = Image.open(fg_file)
        fg_info = "使用原图 alpha 通道（或不透明前景）。"
        if cutout_mode == "一键智能抠图(U2Net)":
            base_fg_rgba, fg_info = remove_background(raw_fg, target=cutout_target)
        elif cutout_mode == "手工抠图(画笔)":
            fg_source_id = _uploaded_file_id(fg_file)
            applied = st.session_state.get("manual_applied_rgba")
            applied_id = st.session_state.get("manual_applied_source_id")
            if applied is None or applied_id != fg_source_id:
                st.warning("请先在手工抠图面板点击“应用手工抠图结果”。")
                st.stop()
            base_fg_rgba = applied.convert("RGBA")
            fg_info = st.session_state.get("manual_applied_info", "已使用手工抠图结果。")
        else:
            base_fg_rgba = raw_fg.convert("RGBA")

        if cutout_mode != "不抠图" and post_cfg:
            base_fg_rgba = refine_rgba_cutout(
                base_fg_rgba,
                alpha_threshold=int(post_cfg.get("alpha_threshold", 12)),
                keep_largest=bool(post_cfg.get("keep_largest", True)),
                feather_radius=int(post_cfg.get("feather_radius", 2)),
                auto_crop=bool(post_cfg.get("auto_crop", True)),
                crop_padding=int(post_cfg.get("crop_padding", 6)),
                invert_mask=bool(post_cfg.get("invert_mask", False)),
            )
            fg_info += " 已应用后处理（连通域/羽化/裁边）。"
        # Keep expected fg/bg ratio when background is auto-resized.
        effective_fg_scale = fg_scale * bg_resize_factor
        fg = resize_foreground(base_fg_rgba, effective_fg_scale)
        fg = fit_foreground_to_background(fg, bg)

        st.caption(fg_info)
        if resized_bg_size != orig_bg_size:
            st.caption(f"背景图已预缩放: {orig_bg_size[0]}x{orig_bg_size[1]} -> {resized_bg_size[0]}x{resized_bg_size[1]}")
        else:
            st.caption(f"背景图尺寸保持不变: {orig_bg_size[0]}x{orig_bg_size[1]}")
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.image(bg, caption="背景图", width="stretch")
        with preview_cols[1]:
            st.image(fg, caption="处理后前景图", width="stretch")

        with st.spinner("本地推理中..."):
            t0 = time.time()
            try:
                scales = [fg_scale]
                if enable_scale_search:
                    for item in scale_offsets:
                        if item == "-20%":
                            scales.append(fg_scale * 0.8)
                        elif item == "-10%":
                            scales.append(fg_scale * 0.9)
                        elif item == "+10%":
                            scales.append(fg_scale * 1.1)
                        elif item == "+20%":
                            scales.append(fg_scale * 1.2)

                merged_rows = []
                merged_images = []
                scale_heatmaps = []
                seen = set()
                scorer = ReferenceOPAScorer(device="auto")
                
                scale_values = sorted(set(max(0.3, min(2.5, s)) for s in scales))
                effective_compose_workers = (
                    1 if parallel_scale_search and len(scale_values) > 1 else cpu_compose_workers
                )

                def search_one_scale(sc: float):
                    fg_sc = resize_foreground(base_fg_rgba, sc * bg_resize_factor)
                    fg_sc = fit_foreground_to_background(fg_sc, bg)
                    return rank_candidates_heatmap_guided(
                        bg,
                        fg_sc,
                        top_k=max(top_k, 6),
                        candidate_count=search_budget,
                        heatmap_grid=heat_grid,
                        scale_tag=sc,
                        scorer=scorer,
                        compose_workers=effective_compose_workers,
                    )

                scale_results = []
                if parallel_scale_search and len(scale_values) > 1:
                    max_workers = min(4, len(scale_values))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(search_one_scale, sc): sc for sc in scale_values}
                        for future in as_completed(futures):
                            sc = futures[future]
                            rows_sc, images_sc, hm_sc = future.result()
                            scale_results.append((sc, rows_sc, images_sc, hm_sc))
                    scale_results.sort(key=lambda item: item[0])
                else:
                    for sc in scale_values:
                        rows_sc, images_sc, hm_sc = search_one_scale(sc)
                        scale_results.append((sc, rows_sc, images_sc, hm_sc))

                for sc, rows_sc, images_sc, hm_sc in scale_results:
                    for r, img in zip(rows_sc, images_sc):
                        key = (int(r["x"]), int(r["y"]), round(float(r["scale"]), 2))
                        if key in seen:
                            continue
                        seen.add(key)
                        merged_rows.append(r)
                        merged_images.append(img)
                    fg_width = int(base_fg_rgba.size[0] * sc * bg_resize_factor)
                    fg_height = int(base_fg_rgba.size[1] * sc * bg_resize_factor)
                    scale_heatmaps.append((sc, hm_sc, fg_width, fg_height))

                merged = sorted(
                    list(zip(merged_rows, merged_images)), key=lambda t: float(t[0]["score"]), reverse=True
                )[:top_k]
                ranked = [x[0] for x in merged]
                images = [x[1] for x in merged]
                hm = None
                hm_overlay = None
                if precompute_heatmap and scale_heatmaps:
                    best_scale = float(ranked[0]["scale"]) if ranked else scale_heatmaps[0][0]
                    chosen = min(scale_heatmaps, key=lambda t: abs(t[0] - best_scale))
                    hm = chosen[1]
                    hm_overlay = _draw_topk_markers(
                        _render_heatmap_overlay(bg, hm), ranked, int(chosen[2]), int(chosen[3])
                    )
            except Exception as exc:
                st.error(f"推理失败: {exc}")
                st.stop()
            latency_ms = (time.time() - t0) * 1000.0

        st.session_state["last_result"] = {
            "bg": bg,
            "fg": fg,
            "base_fg_rgba": base_fg_rgba,
            "ranked": ranked,
            "images": images,
            "latency_ms": latency_ms,
            "orig_bg_size": orig_bg_size,
            "resized_bg_size": resized_bg_size,
            "device": str(scorer.device),
            "parallel_scale_search": bool(parallel_scale_search and len(scale_values) > 1),
            "cpu_compose_workers": int(cpu_compose_workers),
            "effective_compose_workers": int(effective_compose_workers),
        }
        st.session_state["manual_score_result"] = None
        st.session_state["last_heatmap"] = hm
        st.session_state["last_heatmap_overlay"] = hm_overlay

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    bg = res["bg"]
    fg = res["fg"]
    base_fg_rgba = res["base_fg_rgba"]
    ranked = res["ranked"]
    images = res["images"]
    latency_ms = res["latency_ms"]
    device_used = res.get("device", "cpu")
    parallel_used = res.get("parallel_scale_search", False)
    cpu_workers = res.get("cpu_compose_workers", 1)
    effective_workers = res.get("effective_compose_workers", cpu_workers)

    st.success(f"完成。总耗时 {latency_ms:.1f} ms")
    st.caption(
        f"推理设备：`{device_used}`（自动优先 GPU，不可用时回退 CPU）；"
        f"多尺度并行：{'开启' if parallel_used else '关闭'}；"
        f"CPU合成线程：{cpu_workers}（搜索实际使用 {effective_workers}）"
    )
    spread = spread_summary(ranked)
    st.caption(
        f"分数分布: min={spread['min']:.3f}, max={spread['max']:.3f}, "
        f"gap={spread['gap']:.3f}, std={spread['std']:.3f}"
    )
    cols = st.columns(len(images))
    for i, (row, img) in enumerate(zip(ranked, images)):
        with cols[i]:
            st.image(
                img,
                caption=f"#{i+1} 分数={row['score']:.3f} ({row['level']}) | scale={row['scale']:.2f}",
                width="stretch",
            )
            st.write(f"位置: x={row['x']}, y={row['y']}")
            cur_fg_w = max(16, int(base_fg_rgba.size[0] * float(row["scale"])))
            cur_fg_h = max(16, int(base_fg_rgba.size[1] * float(row["scale"])))
            tips = analyze_candidate(
                x=int(row["x"]),
                y=int(row["y"]),
                fg_w=cur_fg_w,
                fg_h=cur_fg_h,
                bg_w=bg.size[0],
                bg_h=bg.size[1],
                score=float(row["score"]),
            )
            st.caption("；".join(tips[:2]))

    st.subheader("排序结果")
    st.dataframe(ranked, width="stretch")

    st.subheader("手动微调打分")
    default_x = int(ranked[0]["x"]) if ranked else 0
    default_y = int(ranked[0]["y"]) if ranked else 0
    default_scale = float(ranked[0]["scale"]) if ranked else 1.0
    max_x = max(0, bg.size[0] - fg.size[0])
    max_y = max(0, bg.size[1] - fg.size[1])
    manual_x = int(st.session_state.get("manual_drag_x", min(default_x, max_x)))
    manual_y = int(st.session_state.get("manual_drag_y", min(default_y, max_y)))
    manual_scale = float(st.session_state.get("manual_drag_scale", default_scale))
    manual_scale = st.slider("手动缩放（拖拽模式）", min_value=0.3, max_value=2.5, value=manual_scale, step=0.01)
    st.session_state["manual_drag_scale"] = manual_scale
    manual_bg_resize_factor = float(res["resized_bg_size"][0]) / max(1.0, float(res["orig_bg_size"][0]))

    use_drag_manual = HAS_DRAWABLE_CANVAS
    if use_drag_manual:
        st.caption("直接拖动图中前景贴图到目标位置，再点击计算分数。")
        fg_manual_drag = resize_foreground(base_fg_rgba, manual_scale * manual_bg_resize_factor)
        fg_manual_drag = fit_foreground_to_background(fg_manual_drag, bg)
        max_x_drag = max(0, bg.size[0] - fg_manual_drag.size[0])
        max_y_drag = max(0, bg.size[1] - fg_manual_drag.size[1])
        manual_x = max(0, min(max_x_drag, manual_x))
        manual_y = max(0, min(max_y_drag, manual_y))

        drag_cache_key = (
            f"{fg_manual_drag.size[0]}x{fg_manual_drag.size[1]}|"
            f"{_uploaded_file_id(fg_file)}|bg:{bg.size[0]}x{bg.size[1]}"
        )
        if st.session_state.get("manual_drag_fg_data_url_key") != drag_cache_key:
            st.session_state["manual_drag_fg_data_url"] = _pil_rgba_to_data_url(fg_manual_drag)
            st.session_state["manual_drag_fg_data_url_key"] = drag_cache_key
        fg_data_url = st.session_state.get("manual_drag_fg_data_url")

        init_image = {
            "version": "4.4.0",
            "objects": [
                {
                    "type": "image",
                    "left": float(manual_x),
                    "top": float(manual_y),
                    "width": float(fg_manual_drag.size[0]),
                    "height": float(fg_manual_drag.size[1]),
                    "scaleX": 1.0,
                    "scaleY": 1.0,
                    "src": fg_data_url,
                    "lockRotation": True,
                    "lockScalingX": True,
                    "lockScalingY": True,
                }
            ],
        }
        # Reset drag state only when source image/size context changes.
        if st.session_state.get("manual_drag_state_key") != drag_cache_key:
            st.session_state["manual_drag_state_key"] = drag_cache_key
            st.session_state["manual_drag_x"] = manual_x
            st.session_state["manual_drag_y"] = manual_y
            st.session_state["manual_drag_canvas_key"] = f"manual_drag_canvas_image_{time.time_ns()}"
        if "manual_drag_canvas_key" not in st.session_state:
            st.session_state["manual_drag_canvas_key"] = f"manual_drag_canvas_image_{time.time_ns()}"

        drag_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=1,
            stroke_color="#FFD400",
            background_image=bg,
            update_streamlit=False,
            width=bg.size[0],
            height=bg.size[1],
            drawing_mode="transform",
            # Always inject a valid foreground object with stable src to prevent
            # the object from disappearing after reruns.
            initial_drawing=init_image,
            display_toolbar=False,
            key=st.session_state["manual_drag_canvas_key"],
        )
        current_drag_json = drag_canvas.json_data
        if current_drag_json is not None:
            manual_x_preview, manual_y_preview = _extract_image_xy_from_canvas(
                json_data=current_drag_json,
                default_x=manual_x,
                default_y=manual_y,
                max_x=max_x_drag,
                max_y=max_y_drag,
            )
            st.session_state["manual_drag_x"] = manual_x_preview
            st.session_state["manual_drag_y"] = manual_y_preview
        else:
            manual_x_preview = int(st.session_state.get("manual_drag_x", manual_x))
            manual_y_preview = int(st.session_state.get("manual_drag_y", manual_y))
        st.caption(f"当前位置：x={manual_x_preview}, y={manual_y_preview}；缩放={manual_scale:.3f}")
        manual_submit = st.button("计算推荐分数（当前位置）")
    else:
        with st.form("manual_score_form", clear_on_submit=False):
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                manual_x = st.slider("手动X位置", min_value=0, max_value=max_x, value=manual_x, step=1)
            with c_b:
                manual_y = st.slider("手动Y位置", min_value=0, max_value=max_y, value=manual_y, step=1)
            with c_c:
                manual_scale = st.slider("手动缩放", min_value=0.3, max_value=2.5, value=float(default_scale), step=0.01)
            manual_submit = st.form_submit_button("计算推荐分数")

    if manual_submit:
        scorer = ReferenceOPAScorer(device="auto")
        fg_manual = resize_foreground(base_fg_rgba, manual_scale * manual_bg_resize_factor)
        fg_manual = fit_foreground_to_background(fg_manual, bg)
        if use_drag_manual:
            current_drag_json = drag_canvas.json_data
            manual_x, manual_y = _extract_image_xy_from_canvas(
                json_data=current_drag_json,
                default_x=int(st.session_state.get("manual_drag_x", manual_x)),
                default_y=int(st.session_state.get("manual_drag_y", manual_y)),
                max_x=max(0, bg.size[0] - fg_manual.size[0]),
                max_y=max(0, bg.size[1] - fg_manual.size[1]),
            )
            st.session_state["manual_drag_x"] = manual_x
            st.session_state["manual_drag_y"] = manual_y
        manual = score_single_position(bg, fg_manual, manual_x, manual_y, scorer=scorer)
        tips = analyze_candidate(
            x=int(manual["x"]),
            y=int(manual["y"]),
            fg_w=fg_manual.size[0],
            fg_h=fg_manual.size[1],
            bg_w=bg.size[0],
            bg_h=bg.size[1],
            score=float(manual["score"]),
        )
        st.session_state["manual_score_result"] = {
            "manual": manual,
            "tips": tips,
            "scale": float(manual_scale),
        }

    if st.session_state.get("manual_score_result") is not None:
        cached = st.session_state["manual_score_result"]
        manual = cached["manual"]
        tips = cached["tips"]
        man_scale = float(cached.get("scale", 1.0))
        st.success(
            f"当前位置分数={manual['score']:.3f}（{manual['level']}），"
            f"x={manual['x']}, y={manual['y']}, scale={man_scale:.3f}"
        )
        st.caption("；".join(tips))

    st.subheader("位置热力图")
    if st.session_state.get("last_heatmap_overlay") is not None:
        hm = st.session_state.get("last_heatmap")
        overlay_with_marks = st.session_state["last_heatmap_overlay"]
        st.image(
            overlay_with_marks,
            caption=f"红色越强表示估计得分越高，已叠加 Top-K 标记点",
            width="stretch",
        )
        if hm is not None:
            st.caption(f"热力图统计: min={float(hm.min()):.3f}, max={float(hm.max()):.3f}, gap={float(hm.max()-hm.min()):.3f}")

    st.subheader("导出结果")
    json_bytes = json.dumps(
        {"ranked": ranked, "latency_ms": float(latency_ms), "spread": spread}, ensure_ascii=False, indent=2
    ).encode("utf-8")
    st.download_button(
        label="下载 ranking.json",
        data=json_bytes,
        file_name="ranking.json",
        mime="application/json",
    )
    zip_bytes = _build_export_zip(
        ranked=ranked,
        images=images,
        heatmap_overlay_with_marks=st.session_state.get("last_heatmap_overlay"),
        raw_heatmap=st.session_state.get("last_heatmap"),
    )
    st.download_button(
        label="下载结果包 (zip)",
        data=zip_bytes,
        file_name="simopa_results.zip",
        mime="application/zip",
    )

st.markdown("---")
st.write("运行入口: `streamlit run app.py`")
