import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

from src.foreground import (
    apply_manual_alpha_mask,
    fit_foreground_to_background,
    refine_rgba_cutout,
    remove_background,
    resize_foreground,
)
from src.image_preprocess import resize_by_long_edge
from src.infer import rank_candidates, score_heatmap, score_single_position
from src.config import STUDENT_CNN_PATH
from src.opa import BACKENDS, REFERENCE_BACKEND, STUDENT_BACKEND, create_opa_scorer
from src.reference_opa import ensure_simopa_weight
from src.user_feedback import analyze_candidate, spread_summary

try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore[reportMissingImports]

    HAS_DRAWABLE_CANVAS = True
except Exception:
    HAS_DRAWABLE_CANVAS = False


st.set_page_config(page_title="方向A-物体放置助手", layout="wide")
st.title("方向 A：智能物体放置与质量评分（本地推理）")
st.caption("评分模型可在侧边栏快速切换：Student CNN 或原始 SimOPA。")


@st.cache_resource(show_spinner=False)
def _load_scorer(model_backend: str):
    return create_opa_scorer(model_backend, device="auto")


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
    base_keep_mask: Optional[np.ndarray] = None,
    expand_px: int = 0,
    shrink_px: int = 0,
    feather_px: int = 1,
) -> np.ndarray:
    """
    Extract user painted keep-region mask from drawable canvas image_data.
    We use bright-green strokes as keep-mark, bright-red strokes as erase-mark.
    """
    if base_keep_mask is None:
        keep_mask = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
    else:
        keep_mask = np.clip(base_keep_mask.astype(np.float32), 0.0, 1.0)
        if keep_mask.shape != (target_size[1], target_size[0]):
            keep_mask = np.asarray(
                Image.fromarray((keep_mask * 255).astype(np.uint8), mode="L").resize(
                    target_size, Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0

    if canvas_image_data is None:
        return keep_mask
    arr = np.asarray(canvas_image_data, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return keep_mask
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    keep_mark = (g > 180) & (r < 120) & (b < 120)
    erase_mark = (r > 180) & (g < 120) & (b < 120)

    keep_img = Image.fromarray((keep_mark.astype(np.uint8) * 255), mode="L").resize(
        target_size, Image.Resampling.BILINEAR
    )
    erase_img = Image.fromarray((erase_mark.astype(np.uint8) * 255), mode="L").resize(
        target_size, Image.Resampling.BILINEAR
    )
    keep_delta = np.asarray(keep_img, dtype=np.float32) / 255.0
    erase_delta = np.asarray(erase_img, dtype=np.float32) / 255.0

    keep_mask = np.maximum(keep_mask, keep_delta)
    keep_mask = keep_mask * (1.0 - erase_delta)

    mask_img = Image.fromarray((np.clip(keep_mask, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    if expand_px > 0:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=_odd(expand_px)))
    if shrink_px > 0:
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=_odd(shrink_px)))
    if feather_px > 1:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    return np.asarray(mask_img, dtype=np.float32) / 255.0


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
    except AttributeError as exc:
        # streamlit_drawable_canvas older versions may call removed internals
        # (e.g. streamlit.elements.image.image_to_url). Fallback still supports
        # manual mask drawing and keeps the feature usable.
        st.warning(f"画布背景兼容失败，已切换兼容模式：{exc}")
        c_left, c_right = st.columns(2)
        with c_left:
            st.image(fg_show, caption="参考前景图（在右侧画布按同位置涂抹）", use_container_width=True)
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

with st.sidebar:
    st.header("模型与权重")
    model_backend = st.selectbox(
        "评分模型",
        options=BACKENDS,
        index=0,
        help="Student CNN 使用训练后的 models/student_cnn.pth；原始 SimOPA 使用 models/SimOPA.pth。",
    )
    if STUDENT_CNN_PATH.exists():
        st.success(f"Student CNN 权重已就绪: {STUDENT_CNN_PATH}")
    else:
        st.warning("未找到 Student CNN 权重。请先运行 `scripts/train_student_cnn.py` 训练。")
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
    if model_backend == STUDENT_BACKEND:
        st.info("当前网页推理会加载 `models/student_cnn.pth`。")
    elif model_backend == REFERENCE_BACKEND:
        st.info("当前网页推理会加载 `models/SimOPA.pth`。")

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

        # Real-time preview for manual cutout result.
        base_rgba = fg_preview.convert("RGBA")
        base_keep = None
        preview_info = "手工掩码预览（仅预览，最终以开始推荐时重新计算为准）"
        if manual_seed_mode == "智能抠图结果":
            try:
                auto_rgba, _ = remove_background(fg_preview, target=cutout_target)
                base_keep = (np.asarray(auto_rgba, dtype=np.uint8)[..., 3] > 20).astype(np.float32)
            except Exception:
                base_keep = None
        keep_mask_preview = _get_canvas_mask(
            canvas_result.image_data,
            target_size=base_rgba.size,
            base_keep_mask=base_keep,
            expand_px=expand_px,
            shrink_px=shrink_px,
            feather_px=feather_px,
        )
        fg_manual_preview = apply_manual_alpha_mask(base_rgba, keep_mask_preview, feather=1)
        if post_cfg:
            fg_manual_preview = refine_rgba_cutout(
                fg_manual_preview,
                alpha_threshold=int(post_cfg.get("alpha_threshold", 12)),
                keep_largest=bool(post_cfg.get("keep_largest", True)),
                feather_radius=int(post_cfg.get("feather_radius", 2)),
                auto_crop=bool(post_cfg.get("auto_crop", True)),
                crop_padding=int(post_cfg.get("crop_padding", 6)),
                invert_mask=bool(post_cfg.get("invert_mask", False)),
            )
        st.image(fg_manual_preview, caption=preview_info, use_container_width=True)
    else:
        st.info("请先上传前景图后再进行手工抠图。")

if st.button("开始推荐"):
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
            base_fg_rgba = raw_fg.convert("RGBA")
            canvas_data = st.session_state.get("manual_canvas_data")
            manual_seed_mode = st.session_state.get("manual_seed_mode", "空白")
            manual_expand_px = int(st.session_state.get("manual_expand_px", 2))
            manual_shrink_px = int(st.session_state.get("manual_shrink_px", 0))
            manual_feather_px = int(st.session_state.get("manual_feather_px", 2))
            base_keep = None
            if manual_seed_mode == "智能抠图结果":
                try:
                    auto_rgba, _ = remove_background(raw_fg, target=cutout_target)
                    base_keep = (np.asarray(auto_rgba, dtype=np.uint8)[..., 3] > 20).astype(np.float32)
                except Exception:
                    base_keep = None
            if canvas_data is not None:
                keep_mask = _get_canvas_mask(
                    canvas_data,
                    target_size=base_fg_rgba.size,
                    base_keep_mask=base_keep,
                    expand_px=manual_expand_px,
                    shrink_px=manual_shrink_px,
                    feather_px=manual_feather_px,
                )
                if float(keep_mask.mean()) > 1e-4:
                    base_fg_rgba = apply_manual_alpha_mask(base_fg_rgba, keep_mask, feather=2)
                    fg_info = "已使用手工抠图（支持保留/擦除、曲线绘制、边缘细化）。"
                else:
                    if base_keep is not None and float(base_keep.mean()) > 1e-4:
                        base_fg_rgba = apply_manual_alpha_mask(base_fg_rgba, base_keep, feather=2)
                        fg_info = "未检测到手工涂抹，已使用智能抠图初始化结果。"
                    else:
                        fg_info = "未检测到画笔涂抹，已使用原图。"
            else:
                fg_info = "未检测到手工抠图数据，已使用原图。"
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
            st.image(bg, caption="背景图", use_container_width=True)
        with preview_cols[1]:
            st.image(fg, caption="处理后前景图", use_container_width=True)

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
                seen = set()
                scorer = _load_scorer(model_backend)

                scale_values = sorted(set(max(0.3, min(2.5, s)) for s in scales))
                effective_compose_workers = (
                    1 if parallel_scale_search and len(scale_values) > 1 else cpu_compose_workers
                )

                def search_one_scale(sc: float):
                    fg_sc = resize_foreground(base_fg_rgba, sc * bg_resize_factor)
                    fg_sc = fit_foreground_to_background(fg_sc, bg)
                    return rank_candidates(
                        bg,
                        fg_sc,
                        top_k=max(top_k, 6),
                        candidate_count=search_budget,
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
                            scale_results.append((futures[future], *future.result()))
                    scale_results.sort(key=lambda item: item[0])
                else:
                    for sc in scale_values:
                        rows_sc, images_sc = search_one_scale(sc)
                        scale_results.append((sc, rows_sc, images_sc))

                for _, rows_sc, images_sc in scale_results:
                    for r, img in zip(rows_sc, images_sc):
                        key = (int(r["x"]), int(r["y"]), round(float(r["scale"]), 2))
                        if key in seen:
                            continue
                        seen.add(key)
                        merged_rows.append(r)
                        merged_images.append(img)

                merged = sorted(
                    list(zip(merged_rows, merged_images)), key=lambda t: float(t[0]["score"]), reverse=True
                )[:top_k]
                ranked = [x[0] for x in merged]
                images = [x[1] for x in merged]
                hm = None
                hm_overlay = None
                if precompute_heatmap:
                    hm = score_heatmap(
                        bg,
                        fg,
                        grid_size=heat_grid,
                        scorer=scorer,
                        compose_workers=cpu_compose_workers,
                    )
                    hm_overlay = _draw_topk_markers(
                        _render_heatmap_overlay(bg, hm), ranked, fg.size[0], fg.size[1]
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
            "model_backend": model_backend,
            "parallel_scale_search": bool(parallel_scale_search and len(scale_values) > 1),
            "cpu_compose_workers": int(cpu_compose_workers),
            "effective_compose_workers": int(effective_compose_workers),
        }
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
    model_used = res.get("model_backend", "Student CNN")
    parallel_used = res.get("parallel_scale_search", False)
    cpu_workers = res.get("cpu_compose_workers", 1)
    effective_workers = res.get("effective_compose_workers", cpu_workers)

    st.success(f"完成。总耗时 {latency_ms:.1f} ms")
    st.caption(
        f"评分模型：`{model_used}`；推理设备：`{device_used}`（自动优先 GPU，不可用时回退 CPU）；"
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
                use_container_width=True,
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
    st.dataframe(ranked, use_container_width=True)

    st.subheader("手动微调打分")
    default_x = int(ranked[0]["x"]) if ranked else 0
    default_y = int(ranked[0]["y"]) if ranked else 0
    max_x = max(0, bg.size[0] - fg.size[0])
    max_y = max(0, bg.size[1] - fg.size[1])
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        manual_x = st.slider("手动X位置", min_value=0, max_value=max_x, value=min(default_x, max_x), step=1)
    with col_m2:
        manual_y = st.slider("手动Y位置", min_value=0, max_value=max_y, value=min(default_y, max_y), step=1)
    if st.button("计算手动位置分数"):
        scorer = _load_scorer(model_backend)
        manual = score_single_position(bg, fg, manual_x, manual_y, scorer=scorer)
        st.image(
            manual["image"],
            caption=f"手动位置分数={manual['score']:.3f} ({manual['level']}) @ ({manual['x']},{manual['y']})",
            use_container_width=True,
        )
        tips = analyze_candidate(
            x=int(manual["x"]),
            y=int(manual["y"]),
            fg_w=fg.size[0],
            fg_h=fg.size[1],
            bg_w=bg.size[0],
            bg_h=bg.size[1],
            score=float(manual["score"]),
        )
        st.caption("；".join(tips))

    st.subheader("位置热力图")
    if st.session_state.get("last_heatmap_overlay") is not None:
        hm = st.session_state.get("last_heatmap")
        overlay_with_marks = st.session_state["last_heatmap_overlay"]
        st.image(
            overlay_with_marks,
            caption=f"红色越强表示估计得分越高，已叠加 Top-K 标记点",
            use_container_width=True,
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
