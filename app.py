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
from src.app_state import clear_result_state
from src.inference_service import merge_ranked_scale_outputs
from src.image_preprocess import resize_by_long_edge
from src.infer import rank_candidates_dense_map, rank_candidates_heatmap_guided, score_single_position
from src.opa import BACKENDS, REFERENCE_BACKEND, STUDENT_BACKEND, STUDENT_MID_BACKEND, create_opa_scorer
from src.config import STUDENT_CNN_PATH, STUDENT_DUAL_PATH, STUDENT_MID_PATH
from src.reference_opa import ensure_simopa_weight
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
expected_device = "cuda" if torch.cuda.is_available() else "cpu"
st.markdown(
    """
<style>
.stApp {
    background: #f8fafc;
    color: #1f2937;
}
header[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
.block-container {
    padding-top: 0.35rem;
    max-width: 96%;
}
.stApp, .stMarkdown, .stText, p, label, div, span, small,
h1, h2, h3, h4, h5, h6, [data-testid="stCaptionContainer"] {
    color: #1f2937 !important;
}
.workspace-toolbar {
    background: linear-gradient(90deg, rgba(226, 236, 255, 0.9), rgba(243, 246, 255, 0.9));
    border: 1px solid rgba(173, 189, 224, 0.75);
    border-radius: 16px;
    padding: 10px 16px;
    margin: 8px 0 12px 0;
    box-shadow: 0 6px 18px rgba(72, 96, 148, 0.12);
}
.workspace-chip {
    display: inline-block;
    margin-right: 10px;
    margin-bottom: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(169, 188, 232, 0.8);
    background: rgba(232, 240, 255, 0.92);
    font-size: 0.85rem;
    color: #1e3a8a !important;
}
.workspace-panel {
    border: 1px solid rgba(203, 213, 229, 0.9);
    border-radius: 12px;
    padding: 10px 12px;
    background: #ffffff;
    color: #1f2937 !important;
}
.workspace-column-shell {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 6px 8px;
    background: #ffffff;
    min-height: 0;
    box-shadow: none;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 10px;
    background: #eef4ff;
    border: 1px solid #dbe6ff;
    border-radius: 14px;
    padding: 6px 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 6px 14px;
    background: #ffffff;
    border: 1px solid #d5e2ff;
    color: #1f2937 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(90deg, #dbeafe, #e0e7ff);
    border-color: #93c5fd;
    color: #1d4ed8 !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #d9e2f2;
}
[data-testid="stMetric"] {
    background: #f8fbff;
    border: 1px solid #dbe5f3;
    border-radius: 12px;
    padding: 8px 10px;
}
[data-testid="stCheckbox"] {
    border: 1px solid #d8e2f0;
    border-radius: 10px;
    padding: 4px 8px;
    background: #ffffff;
}
[data-testid="stCheckbox"] label p {
    color: #1f2937 !important;
}
[data-baseweb="checkbox"] > div {
    background: #ffffff !important;
    border-color: #9ca3af !important;
}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #2563eb, #4f46e5) !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 10px !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


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


def _draw_topk_markers(
    image: Image.Image,
    ranked: list[dict],
    scale_to_fg_size: dict[float, tuple[int, int]],
    default_fg_size: tuple[int, int],
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(ranked):
        scale_key = round(float(row.get("scale", 1.0)), 3)
        fg_w, fg_h = scale_to_fg_size.get(scale_key, default_fg_size)
        x = int(row["x"] + fg_w / 2)
        y = int(row["y"] + fg_h / 2)
        r = max(6, min(fg_w, fg_h) // 10)
        color = (255, 255, 0) if i == 0 else (0, 255, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        draw.text((x + r + 2, y - r - 2), f"#{i+1}", fill=color)
    return out


def _compose_preview(bg: Image.Image, fg: Image.Image, x: int, y: int) -> Image.Image:
    base = bg.convert("RGBA").copy()
    base.alpha_composite(fg.convert("RGBA"), dest=(int(x), int(y)))
    return base.convert("RGB")


def _mock_level(score: float) -> str:
    if score >= 0.75:
        return "高"
    if score >= 0.5:
        return "中"
    return "低"


def _mock_inference_outputs(
    bg: Image.Image,
    base_fg_rgba: Image.Image,
    fg_scale: float,
    bg_resize_factor: float,
    top_k: int,
    heat_grid: int,
) -> tuple[list[dict], list[Image.Image], np.ndarray, Image.Image]:
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))
    candidate_count = max(top_k, 8)
    rows_all: list[dict] = []
    images_all: list[Image.Image] = []
    scale_to_fg_size: dict[float, tuple[int, int]] = {}

    for _ in range(candidate_count):
        sc = float(np.clip(fg_scale * float(rng.uniform(0.82, 1.18)), 0.3, 2.5))
        fg_sc = resize_foreground(base_fg_rgba, sc * bg_resize_factor)
        fg_sc = fit_foreground_to_background(fg_sc, bg)
        max_x = max(0, bg.size[0] - fg_sc.size[0])
        max_y = max(0, bg.size[1] - fg_sc.size[1])
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
        score = float(rng.uniform(0.05, 0.95))
        level = _mock_level(score)
        sc_key = round(sc, 3)
        scale_to_fg_size[sc_key] = (int(fg_sc.size[0]), int(fg_sc.size[1]))
        rows_all.append(
            {
                "x": int(x),
                "y": int(y),
                "score": float(score),
                "level": level,
                "scale": float(sc_key),
            }
        )
        images_all.append(_compose_preview(bg, fg_sc, x, y))

    order = sorted(range(len(rows_all)), key=lambda i: float(rows_all[i]["score"]), reverse=True)
    picked = order[:top_k]
    ranked = [rows_all[i] for i in picked]
    images = [images_all[i] for i in picked]

    hm_size = max(8, int(heat_grid))
    hm = rng.random((hm_size, hm_size), dtype=np.float32)
    if ranked:
        fallback_fg_size = scale_to_fg_size.get(
            round(float(ranked[0]["scale"]), 3),
            (int(bg.size[0] * 0.2), int(bg.size[1] * 0.2)),
        )
    else:
        fallback_fg_size = (int(bg.size[0] * 0.2), int(bg.size[1] * 0.2))
    hm_overlay = _draw_topk_markers(
        _render_heatmap_overlay(bg, hm),
        ranked,
        scale_to_fg_size=scale_to_fg_size,
        default_fg_size=fallback_fg_size,
    )
    return ranked, images, hm, hm_overlay


def _backend_weight_status(model_backend: str) -> tuple[bool, str]:
    if model_backend == STUDENT_BACKEND:
        if STUDENT_CNN_PATH.exists():
            return True, f"Student CNN 权重已找到：`{STUDENT_CNN_PATH}`"
        return False, "未找到 Student CNN 权重。请先训练或放置 `models/student_cnn.pth`。"
    if model_backend.startswith("Student Dual"):
        if STUDENT_DUAL_PATH.exists():
            return True, f"Student Dual 权重已找到：`{STUDENT_DUAL_PATH}`"
        return False, "未找到 Student Dual 权重。请先运行 `scripts/train_student_dual.py`。"
    if model_backend == STUDENT_MID_BACKEND:
        if STUDENT_MID_PATH.exists():
            return True, f"Student Mid 权重已找到：`{STUDENT_MID_PATH}`"
        return False, "未找到 Student Mid 权重。请先运行 `scripts/train_student_mid.py`。"
    return True, "当前后端无需额外学生权重。"


@st.cache_resource
def _cached_backend_scorer(model_backend: str, device: str):
    return create_opa_scorer(model_backend=model_backend, device=device)


def _apply_post_cfg_to_cutout(rgba: Image.Image, post_cfg: Optional[dict]) -> Image.Image:
    if not post_cfg:
        return rgba.convert("RGBA")
    return refine_rgba_cutout(
        rgba.convert("RGBA"),
        alpha_threshold=int(post_cfg.get("alpha_threshold", 12)),
        keep_largest=bool(post_cfg.get("keep_largest", True)),
        feather_radius=int(post_cfg.get("feather_radius", 2)),
        auto_crop=bool(post_cfg.get("auto_crop", True)),
        crop_padding=int(post_cfg.get("crop_padding", 6)),
        invert_mask=bool(post_cfg.get("invert_mask", False)),
    )


def _build_export_zip(
    ranked: list[dict],
    images: list[Image.Image],
    heatmap_overlay_with_marks: Image.Image | None,
    raw_heatmap: np.ndarray | None,
    model_backend: str,
    device_used: str,
    compare_report: list[dict] | None = None,
) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_backend": model_backend,
            "device": device_used,
            "ranked": ranked,
        }
        if raw_heatmap is not None:
            meta["heatmap_stats"] = {
                "min": float(raw_heatmap.min()),
                "max": float(raw_heatmap.max()),
                "gap": float(raw_heatmap.max() - raw_heatmap.min()),
            }
        if compare_report:
            meta["compare_report"] = compare_report
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


def _fit_image_to_box(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    src = image.convert("RGBA") if image.mode in ("RGBA", "LA") else image.convert("RGB")
    w, h = src.size
    scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
    if scale >= 0.999:
        return src
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return src.resize(new_size, Image.Resampling.BILINEAR)


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

left_panel, center_panel, right_panel = st.columns([0.75, 2.6, 0.95], gap="large")
start_recommend = False
clear_results = False
mock_infer_mode = bool(st.session_state.get("ui_mock_infer_mode", False))
cutout_mode = st.session_state.get("ui_cutout_mode", "不抠图")
cutout_target = st.session_state.get("ui_cutout_target", "person")
fg_scale = float(st.session_state.get("ui_fg_scale", 1.0))
post_cfg = {}
run_status_placeholder = None
run_preview_placeholder = None

with left_panel:
    st.markdown('<div class="workspace-column-shell">', unsafe_allow_html=True)
    bg_file = st.file_uploader(
        "上传背景图 (jpg/jpeg/png/webp/bmp)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="bg",
    )
    fg_file = st.file_uploader(
        "上传前景图 (jpg/jpeg/png/webp/bmp)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="fg",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_panel:
    st.markdown('<div class="workspace-column-shell">', unsafe_allow_html=True)
    st.caption(f"设备: `{expected_device}`")
    run_tab, cutout_tab, search_tab = st.tabs(["运行", "抠图", "搜索"])

    with run_tab:
        model_backend = st.selectbox(
            "评分模型后端",
            options=BACKENDS,
            index=BACKENDS.index(REFERENCE_BACKEND),
            help="原始 SimOPA：更稳健；Student 系列：按速度/精度分级对比。",
        )
        backend_ready, backend_msg = _backend_weight_status(model_backend)
        if backend_ready:
            if model_backend != REFERENCE_BACKEND:
                st.caption(backend_msg)
        else:
            st.warning(backend_msg)
        resolution_profile = st.selectbox(
            "大图自动压缩策略（按长边）",
            options=["1080P (1920)", "2K (2560)", "关闭"],
            index=0,
        )
        inference_mode = st.selectbox(
            "推理策略",
            options=["热力图引导搜索（默认）", "DenseMap加速（实验）"],
            index=0,
        )
        mock_infer_mode = st.checkbox(
            "测试后门：随机结果（跳过模型推理）",
            value=bool(st.session_state.get("ui_mock_infer_mode", False)),
            key="ui_mock_infer_mode",
            help="用于快速调试 GUI，结果为随机生成，不代表模型能力。",
        )
        with st.expander("模型权重管理", expanded=False):
            if st.button("下载/检查 SimOPA 参考权重", width="stretch"):
                with st.spinner("准备 SimOPA 权重中..."):
                    try:
                        path = ensure_simopa_weight()
                        st.success(f"已就绪: {path}")
                    except Exception as exc:
                        st.error("权重下载失败。请检查网络或手动放置 `SimOPA.pth`。")
                        st.caption(str(exc))
        st.markdown("---")
        start_recommend = st.button("开始推荐", type="primary", width="stretch")
        clear_results = st.button("清除当前推荐结果", width="stretch")

    with cutout_tab:
        cutout_mode = st.selectbox(
            "前景抠图方式",
            options=["不抠图", "一键智能抠图(U2Net)", "手工抠图(画笔)"],
            index=["不抠图", "一键智能抠图(U2Net)", "手工抠图(画笔)"].index(cutout_mode)
            if cutout_mode in ["不抠图", "一键智能抠图(U2Net)", "手工抠图(画笔)"]
            else 0,
            key="ui_cutout_mode",
        )
        cutout_target = st.selectbox(
            "抠图目标",
            options=["person", "foreground"],
            index=0 if cutout_target == "person" else 1,
            key="ui_cutout_target",
        )
        fg_scale = st.slider(
            "前景缩放比例",
            min_value=0.3,
            max_value=2.5,
            value=float(fg_scale),
            step=0.05,
            key="ui_fg_scale",
        )
        if cutout_mode == "手工抠图(画笔)":
            st.caption("流程：中间画布绘制 -> 点击“应用抠图”。")
        if cutout_mode != "不抠图":
            with st.expander("抠图后处理参数", expanded=False):
                post_cfg["alpha_threshold"] = st.slider("Alpha阈值", min_value=1, max_value=80, value=12, step=1, key="ui_alpha_threshold")
                post_cfg["auto_crop"] = st.checkbox("自动裁掉透明边框", value=True, key="ui_auto_crop")
                post_cfg["keep_largest"] = st.checkbox("仅保留最大连通域", value=True, key="ui_keep_largest")
                post_cfg["invert_mask"] = st.checkbox("反选掩码", value=False, key="ui_invert_mask")
                post_cfg["feather_radius"] = st.slider("后处理羽化", min_value=1, max_value=8, value=2, step=1, key="ui_feather_radius")
                post_cfg["crop_padding"] = st.slider("裁边留白", min_value=0, max_value=30, value=6, step=1, key="ui_crop_padding")
        if cutout_mode == "手工抠图(画笔)":
            with st.expander("手工抠图参数", expanded=False):
                st.selectbox("初始化掩码", options=["空白", "智能抠图结果"], index=1, key="ui_manual_seed_mode")
                st.selectbox("画笔模式", options=["保留(绿色)", "擦除(红色)"], index=0, key="ui_manual_brush_mode")
                st.selectbox("绘制方式", options=["freedraw", "polygon", "line"], index=0, key="ui_manual_draw_mode")
                st.selectbox(
                    "手工与智能掩码融合",
                    options=["并集", "交集", "仅手工", "仅智能"],
                    index=0,
                    key="ui_manual_combine_mode",
                )
                st.slider("画笔粗细", min_value=4, max_value=60, value=18, step=2, key="ui_manual_brush")
                st.slider("掩码扩张", min_value=0, max_value=15, value=2, step=1, key="ui_manual_expand_px")
                st.slider("掩码收缩", min_value=0, max_value=15, value=0, step=1, key="ui_manual_shrink_px")
                st.slider("边缘羽化", min_value=1, max_value=12, value=2, step=1, key="ui_manual_feather_px")

    with search_tab:
        top_k = st.slider("展示推荐 Top-K", min_value=1, max_value=8, value=5)
        search_budget = st.slider("搜索预算（候选中心数）", min_value=16, max_value=120, value=48, step=4)
        heat_grid = st.slider("热力图网格密度（每边采样点）", min_value=8, max_value=40, value=18, step=2)
        precompute_heatmap = st.checkbox("推理时同步生成热力图", value=True)
        with st.expander("高级搜索参数", expanded=False):
            enable_scale_search = st.checkbox("启用多尺度搜索（推荐）", value=True)
            parallel_scale_search = st.checkbox(
                "并行执行多尺度搜索",
                value=False,
                disabled=not enable_scale_search,
            )
            max_cpu_compose_workers = max(1, min(8, os.cpu_count() or 4))
            cpu_compose_workers = st.slider(
                "CPU候选合成线程数",
                min_value=1,
                max_value=max_cpu_compose_workers,
                value=1,
                step=1,
            )
            scale_offsets = st.multiselect(
                "额外尝试缩放",
                options=["-20%", "-10%", "+10%", "+20%"],
                default=["-10%", "+10%"],
            )
            run_dual_backend_compare = st.checkbox("同时评测双后端（当前+原始SimOPA）", value=False)

    # Ensure defaults when advanced section is unopened in current rerun.
    if "enable_scale_search" not in locals():
        enable_scale_search = True
    if "parallel_scale_search" not in locals():
        parallel_scale_search = False
    if "cpu_compose_workers" not in locals():
        cpu_compose_workers = 1
    if "scale_offsets" not in locals():
        scale_offsets = ["-10%", "+10%"]
    if "run_dual_backend_compare" not in locals():
        run_dual_backend_compare = False
    st.markdown("</div>", unsafe_allow_html=True)

with center_panel:
    st.markdown('<div class="workspace-panel">', unsafe_allow_html=True)
    tab_preview, tab_cutout, tab_run, tab_result_stage = st.tabs(
        ["1) 预览", "2) 抠图处理", "3) 运行推荐", "4) 结果查看"]
    )
    with tab_preview:
        st.caption("导入背景与前景后，在此确认素材状态。")
        if bg_file is not None:
            try:
                _bg_preview = Image.open(bg_file).convert("RGB")
                if fg_file is not None:
                    _fg_preview = Image.open(fg_file).convert("RGBA")
                    c_bg, c_fg = st.columns(2)
                    with c_bg:
                        st.image(_bg_preview, caption="背景预览（工作画布）", width="stretch")
                    with c_fg:
                        st.image(_fg_preview, caption="前景预览（素材）", width="stretch")
                else:
                    st.image(_bg_preview, caption="背景预览（工作画布）", width="stretch")
            except Exception:
                st.info("背景图预览失败，请重新上传。")
        else:
            st.info("请先在左侧上传背景图与前景图。")

    with tab_run:
        run_status_placeholder = st.empty()
        run_preview_placeholder = st.container()
        run_status = st.session_state.get("run_status", "idle")
        run_message = st.session_state.get("run_message", "请在右侧参数栏点击“开始推荐”。")
        if run_status == "running":
            run_status_placeholder.warning(run_message)
        elif run_status == "success":
            run_status_placeholder.success(run_message)
        elif run_status == "error":
            run_status_placeholder.error(run_message)
        else:
            run_status_placeholder.info(run_message)
        if st.session_state.get("run_preview_bg") is not None and st.session_state.get("run_preview_fg") is not None:
            c_prev_1, c_prev_2 = run_preview_placeholder.columns(2)
            with c_prev_1:
                st.image(st.session_state["run_preview_bg"], caption="本次运行背景", width="stretch")
            with c_prev_2:
                st.image(st.session_state["run_preview_fg"], caption="本次运行前景", width="stretch")
    with tab_result_stage:
        if "last_result" not in st.session_state:
            st.info("尚未生成推荐结果。请先切换到「3) 运行推荐」并执行。")
        else:
            res_tab = st.session_state["last_result"]
            ranked_tab = res_tab["ranked"]
            images_tab = res_tab["images"]
            spread_tab = spread_summary(ranked_tab)
            st.markdown(
                f"""
<div class="workspace-toolbar">
  <span class="workspace-chip">后端: {res_tab.get('model_backend', REFERENCE_BACKEND)}</span>
  <span class="workspace-chip">设备: {res_tab.get('device', 'cpu')}</span>
  <span class="workspace-chip">耗时: {res_tab.get('latency_ms', 0.0):.1f} ms</span>
  <span class="workspace-chip">分数 gap: {spread_tab['gap']:.3f}</span>
</div>
                """,
                unsafe_allow_html=True,
            )
            result_overview, result_candidates, result_heatmap, result_export = st.tabs(
                ["结果总览", "候选结果", "热力图", "导出"]
            )
            with result_overview:
                if images_tab:
                    st.image(
                        images_tab[0],
                        caption=f"Top1 分数={ranked_tab[0]['score']:.3f}（{ranked_tab[0]['level']}）",
                        width="stretch",
                    )
                else:
                    st.info("当前无可展示结果。")
            with result_candidates:
                if not images_tab:
                    st.info("暂无候选结果。")
                else:
                    n_cols_tab = min(3, max(1, len(images_tab)))
                    cols_tab = st.columns(n_cols_tab)
                    for i, (row, img) in enumerate(zip(ranked_tab, images_tab)):
                        with cols_tab[i % n_cols_tab]:
                            st.image(img, caption=f"#{i+1} 分数={row['score']:.3f}", width="stretch")
                            st.caption(f"x={row['x']}, y={row['y']}, scale={row['scale']:.2f}")
            with result_heatmap:
                hm_overlay_tab = st.session_state.get("last_heatmap_overlay")
                hm_tab = st.session_state.get("last_heatmap")
                if hm_overlay_tab is None:
                    st.info("当前没有热力图，请先执行推荐。")
                else:
                    st.image(hm_overlay_tab, caption="热力图 + Top-K 标记", width="stretch")
                    if hm_tab is not None:
                        st.caption(
                            f"热力图统计: min={float(hm_tab.min()):.3f}, max={float(hm_tab.max()):.3f}, "
                            f"gap={float(hm_tab.max()-hm_tab.min()):.3f}"
                        )
            with result_export:
                json_bytes_tab = json.dumps(
                    {
                        "model_backend": res_tab.get("model_backend", REFERENCE_BACKEND),
                        "inference_mode": res_tab.get("inference_mode", "热力图引导搜索（默认）"),
                        "device": res_tab.get("device", "cpu"),
                        "ranked": ranked_tab,
                        "latency_ms": float(res_tab.get("latency_ms", 0.0)),
                        "spread": spread_tab,
                        "compare_report": res_tab.get("compare_report", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8")
                st.download_button(
                    label="下载 ranking.json",
                    data=json_bytes_tab,
                    file_name="ranking.json",
                    mime="application/json",
                    key="result_stage_download_json",
                )
                zip_bytes_tab = _build_export_zip(
                    ranked=ranked_tab,
                    images=images_tab,
                    heatmap_overlay_with_marks=st.session_state.get("last_heatmap_overlay"),
                    raw_heatmap=st.session_state.get("last_heatmap"),
                    model_backend=res_tab.get("model_backend", REFERENCE_BACKEND),
                    device_used=res_tab.get("device", "cpu"),
                    compare_report=res_tab.get("compare_report", []),
                )
                st.download_button(
                    label="下载结果包 (zip)",
                    data=zip_bytes_tab,
                    file_name="simopa_results.zip",
                    mime="application/zip",
                    key="result_stage_download_zip",
                )

    with tab_cutout:
        if cutout_mode == "手工抠图(画笔)":
            if not HAS_DRAWABLE_CANVAS:
                st.warning("未安装 `streamlit-drawable-canvas`，请执行 `pip install streamlit-drawable-canvas` 后重启。")
            elif fg_file is not None:
                fg_source_id = _uploaded_file_id(fg_file)
                st.session_state["manual_fg_source_id"] = fg_source_id
                fg_preview = Image.open(fg_file).convert("RGB")
                max_show_w = 760
                max_show_h = 500
                show_scale = min(
                    1.0,
                    max_show_w / max(1, fg_preview.size[0]),
                    max_show_h / max(1, fg_preview.size[1]),
                )
                show_size = (int(fg_preview.size[0] * show_scale), int(fg_preview.size[1] * show_scale))
                fg_show = fg_preview.resize(show_size, Image.Resampling.BILINEAR)
                manual_seed_mode = st.session_state.get("ui_manual_seed_mode", "智能抠图结果")
                manual_brush_mode = st.session_state.get("ui_manual_brush_mode", "保留(绿色)")
                draw_mode = st.session_state.get("ui_manual_draw_mode", "freedraw")
                combine_mode = st.session_state.get("ui_manual_combine_mode", "并集")

                auto_seed_rgba = None
                if manual_seed_mode == "智能抠图结果":
                    auto_seed_rgba = _get_cached_auto_seed_rgba(
                        fg_preview=fg_preview,
                        fg_source_id=fg_source_id,
                        cutout_target=cutout_target,
                        show_spinner=True,
                    )

                brush = int(st.session_state.get("ui_manual_brush", 18))
                expand_px = int(st.session_state.get("ui_manual_expand_px", 2))
                shrink_px = int(st.session_state.get("ui_manual_shrink_px", 0))
                feather_px = int(st.session_state.get("ui_manual_feather_px", 2))

                if "manual_canvas_uid" not in st.session_state:
                    st.session_state["manual_canvas_uid"] = f"canvas_{time.time_ns()}"

                workspace_col_canvas, workspace_col_preview = st.columns([2.2, 1.1], gap="small")
                with workspace_col_canvas:
                    stroke_color = "#00FF00" if manual_brush_mode.startswith("保留") else "#FF0000"
                    fill_color = "rgba(0,255,0,0.22)" if manual_brush_mode.startswith("保留") else "rgba(255,0,0,0.22)"
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

                    action_col_1, action_col_2, action_col_3 = st.columns([1, 1, 2])
                    with action_col_1:
                        if st.button("清空画布"):
                            st.session_state["manual_canvas_uid"] = f"canvas_{time.time_ns()}"
                            st.session_state["manual_canvas_data"] = None
                            st.session_state["manual_canvas_compat_mode"] = False
                            st.session_state["manual_applied_rgba"] = None
                            st.session_state["manual_applied_info"] = None
                            st.session_state["manual_applied_postprocessed"] = False
                            st.rerun()
                    with action_col_2:
                        apply_clicked = st.button("应用抠图", type="primary")
                    with action_col_3:
                        if st.button("丢弃已应用"):
                            st.session_state["manual_applied_rgba"] = None
                            st.session_state["manual_applied_info"] = None
                            st.session_state["manual_applied_postprocessed"] = False
                            st.rerun()

                if apply_clicked:
                    base_rgba = fg_preview.convert("RGBA")
                    base_keep = None
                    if manual_seed_mode == "智能抠图结果" and auto_seed_rgba is not None:
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
                    st.session_state["manual_applied_rgba"] = fg_manual
                    st.session_state["manual_applied_info"] = (
                        f"已应用手工抠图：融合={combine_mode}，保留笔迹={int((keep_delta>0.1).sum())}，"
                        f"擦除笔迹={int((erase_delta>0.1).sum())}"
                    )
                    st.session_state["manual_applied_source_id"] = fg_source_id
                    st.session_state["manual_applied_postprocessed"] = False

                preview_items: list[tuple[str, Image.Image]] = [("原图", fg_preview.convert("RGBA"))]
                if auto_seed_rgba is not None:
                    preview_items.append(("智能初始化", auto_seed_rgba.convert("RGBA")))
                applied_rgba = st.session_state.get("manual_applied_rgba")
                if applied_rgba is not None and st.session_state.get("manual_applied_source_id") == fg_source_id:
                    preview_items.append(("应用结果", applied_rgba.convert("RGBA")))
                with workspace_col_preview:
                    with st.expander("预览面板", expanded=True):
                        preview_tab_main, preview_tab_thumb = st.tabs(["主预览", "缩略图"])
                        with preview_tab_main:
                            default_label = "应用结果" if any(name == "应用结果" for name, _ in preview_items) else preview_items[0][0]
                            choice = st.radio(
                                "主预览",
                                options=[name for name, _ in preview_items],
                                index=[name for name, _ in preview_items].index(default_label),
                                horizontal=True,
                                label_visibility="collapsed",
                            )
                            selected_image = next(img for name, img in preview_items if name == choice)
                            st.image(_fit_image_to_box(selected_image, max_w=520, max_h=360), width="stretch")
                        with preview_tab_thumb:
                            cols = st.columns(max(1, min(2, len(preview_items))))
                            for idx, (name, image_obj) in enumerate(preview_items):
                                thumb = _fit_image_to_box(image_obj, max_w=220, max_h=150)
                                with cols[idx % len(cols)]:
                                    st.image(thumb, caption=name, width="stretch")
            else:
                st.info("请先上传前景图。")

        elif cutout_mode == "一键智能抠图(U2Net)":
            if fg_file is None:
                st.info("请先上传前景图。")
            else:
                fg_source_id = _uploaded_file_id(fg_file)
                fg_preview = Image.open(fg_file).convert("RGB")
                auto_seed_rgba = _get_cached_auto_seed_rgba(
                    fg_preview=fg_preview,
                    fg_source_id=fg_source_id,
                    cutout_target=cutout_target,
                    show_spinner=False,
                )
                if auto_seed_rgba is None:
                    st.warning("智能抠图结果暂不可用。")
                else:
                    smart_view_col, smart_preview_col = st.columns([2.2, 1.1], gap="small")
                    with smart_view_col:
                        st.image(_fit_image_to_box(auto_seed_rgba, max_w=760, max_h=500), width="stretch")
                    with smart_preview_col:
                        with st.expander("预览面板", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(_fit_image_to_box(fg_preview, max_w=220, max_h=150), caption="原图", width="stretch")
                            with c2:
                                st.image(_fit_image_to_box(auto_seed_rgba, max_w=220, max_h=150), caption="智能结果", width="stretch")
        else:
            if fg_file is not None:
                fg_preview = Image.open(fg_file).convert("RGBA")
                st.image(_fit_image_to_box(fg_preview, max_w=760, max_h=500), width="stretch")
            else:
                st.info("请先上传前景图。")
    st.markdown("</div>", unsafe_allow_html=True)

if clear_results:
    clear_result_state(st.session_state)
    st.rerun()

if start_recommend:
    if bg_file is None or fg_file is None:
        st.session_state["run_status"] = "error"
        st.session_state["run_message"] = "运行失败：请先上传背景图和前景图。"
        if run_status_placeholder is not None:
            run_status_placeholder.error(st.session_state["run_message"])
    else:
        st.session_state["run_status"] = "running"
        st.session_state["run_message"] = "推荐运行中，请稍候..."
        if run_status_placeholder is not None:
            run_status_placeholder.warning(st.session_state["run_message"])
        if not mock_infer_mode:
            backend_ready, backend_msg = _backend_weight_status(model_backend)
            if not backend_ready:
                st.session_state["run_status"] = "error"
                st.session_state["run_message"] = f"运行失败：{backend_msg}"
                if run_status_placeholder is not None:
                    run_status_placeholder.error(st.session_state["run_message"])
                st.stop()
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
                st.session_state["run_status"] = "error"
                st.session_state["run_message"] = "运行失败：请先在中间抠图画布点击“应用抠图”。"
                if run_status_placeholder is not None:
                    run_status_placeholder.error(st.session_state["run_message"])
                st.stop()
            base_fg_rgba = applied.convert("RGBA")
            fg_info = st.session_state.get("manual_applied_info", "已使用手工抠图结果。")
        else:
            base_fg_rgba = raw_fg.convert("RGBA")

        if cutout_mode != "不抠图" and post_cfg:
            manual_already_post = bool(st.session_state.get("manual_applied_postprocessed", False))
            if not (cutout_mode == "手工抠图(画笔)" and manual_already_post):
                base_fg_rgba = _apply_post_cfg_to_cutout(base_fg_rgba, post_cfg)
                fg_info += " 已应用后处理（连通域/羽化/裁边）。"
                if cutout_mode == "手工抠图(画笔)":
                    st.session_state["manual_applied_postprocessed"] = True
        # Keep expected fg/bg ratio when background is auto-resized.
        effective_fg_scale = fg_scale * bg_resize_factor
        fg = resize_foreground(base_fg_rgba, effective_fg_scale)
        fg = fit_foreground_to_background(fg, bg)
        st.session_state["run_preview_bg"] = bg.copy()
        st.session_state["run_preview_fg"] = fg.copy()
        if run_preview_placeholder is not None:
            with run_preview_placeholder:
                c_prev_1, c_prev_2 = st.columns(2)
                with c_prev_1:
                    st.image(bg, caption="本次运行背景", width="stretch")
                with c_prev_2:
                    st.image(fg, caption="本次运行前景", width="stretch")

        t0 = time.time()
        if mock_infer_mode:
            ranked, images, hm, hm_overlay = _mock_inference_outputs(
                bg=bg,
                base_fg_rgba=base_fg_rgba,
                fg_scale=fg_scale,
                bg_resize_factor=bg_resize_factor,
                top_k=top_k,
                heat_grid=heat_grid,
            )
            device_used = "mock-random"
            parallel_enabled = False
            effective_compose_workers = 0
            compare_report = []
        else:
            with st.spinner("本地推理中..."):
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

                scale_values = sorted(set(max(0.3, min(2.5, s)) for s in scales))

                def _run_backend(backend_name: str):
                    scorer = _cached_backend_scorer(model_backend=backend_name, device="auto")
                    parallel_enabled = bool(
                        parallel_scale_search
                        and len(scale_values) > 1
                        and getattr(scorer.device, "type", "cpu") == "cpu"
                    )
                    if parallel_scale_search and len(scale_values) > 1 and not parallel_enabled:
                        st.caption(f"[{backend_name}] 检测到 GPU/MPS 推理，自动关闭多尺度线程并行。")
                    effective_compose_workers = 1 if parallel_enabled else cpu_compose_workers

                    def search_one_scale(sc: float):
                        fg_sc = resize_foreground(base_fg_rgba, sc * bg_resize_factor)
                        fg_sc = fit_foreground_to_background(fg_sc, bg)
                        local_scorer = scorer
                        if inference_mode.startswith("DenseMap"):
                            rows_sc, images_sc, hm_sc = rank_candidates_dense_map(
                                bg,
                                fg_sc,
                                top_k=max(top_k, 6),
                                heatmap_grid=heat_grid,
                                refine_per_point=6,
                                scale_tag=sc,
                                scorer=local_scorer,
                                compose_workers=effective_compose_workers,
                            )
                            return rows_sc, images_sc, hm_sc, fg_sc.size
                        rows_sc, images_sc, hm_sc = rank_candidates_heatmap_guided(
                            bg,
                            fg_sc,
                            top_k=max(top_k, 6),
                            candidate_count=search_budget,
                            heatmap_grid=heat_grid,
                            scale_tag=sc,
                            scorer=local_scorer,
                            compose_workers=effective_compose_workers,
                        )
                        return rows_sc, images_sc, hm_sc, fg_sc.size

                    scale_results = []
                    if parallel_enabled:
                        max_workers = min(4, len(scale_values))
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {executor.submit(search_one_scale, sc): sc for sc in scale_values}
                            for future in as_completed(futures):
                                sc = futures[future]
                                rows_sc, images_sc, hm_sc, fg_size_sc = future.result()
                                scale_results.append((sc, rows_sc, images_sc, hm_sc, fg_size_sc))
                        scale_results.sort(key=lambda item: item[0])
                    else:
                        for sc in scale_values:
                            rows_sc, images_sc, hm_sc, fg_size_sc = search_one_scale(sc)
                            scale_results.append((sc, rows_sc, images_sc, hm_sc, fg_size_sc))

                    ranked, images, scale_heatmaps = merge_ranked_scale_outputs(
                        scale_results=scale_results,
                        top_k=top_k,
                    )
                    hm = None
                    hm_overlay = None
                    if precompute_heatmap and scale_heatmaps:
                        best_scale = float(ranked[0]["scale"]) if ranked else scale_heatmaps[0][0]
                        chosen = min(scale_heatmaps, key=lambda t: abs(t[0] - best_scale))
                        hm = chosen[1]
                        scale_to_fg_size = {
                            round(float(sc), 3): (int(w), int(h))
                            for sc, _hm, w, h in scale_heatmaps
                        }
                        hm_overlay = _draw_topk_markers(
                            _render_heatmap_overlay(bg, hm),
                            ranked,
                            scale_to_fg_size=scale_to_fg_size,
                            default_fg_size=(int(chosen[2]), int(chosen[3])),
                        )
                    return {
                        "ranked": ranked,
                        "images": images,
                        "hm": hm,
                        "hm_overlay": hm_overlay,
                        "device": str(scorer.device),
                        "parallel_enabled": bool(parallel_enabled),
                        "effective_compose_workers": int(effective_compose_workers),
                    }

                    primary = _run_backend(model_backend)
                    compare_report = []
                    if run_dual_backend_compare:
                        compare_backends = [model_backend]
                        if REFERENCE_BACKEND not in compare_backends:
                            compare_backends.append(REFERENCE_BACKEND)
                        for bname in compare_backends:
                            if bname == model_backend:
                                out_cmp = primary
                                cmp_latency = float((time.time() - t0) * 1000.0)
                            else:
                                t_cmp = time.time()
                                out_cmp = _run_backend(bname)
                                cmp_latency = (time.time() - t_cmp) * 1000.0
                            cmp_scores = [float(r["score"]) for r in out_cmp["ranked"]]
                            compare_report.append(
                                {
                                    "backend": bname,
                                    "latency_ms": float(cmp_latency),
                                    "top1_score": float(cmp_scores[0]) if cmp_scores else 0.0,
                                    "gap": float(max(cmp_scores) - min(cmp_scores)) if len(cmp_scores) >= 2 else 0.0,
                                }
                            )

                    ranked = primary["ranked"]
                    images = primary["images"]
                    hm = primary["hm"]
                    hm_overlay = primary["hm_overlay"]
                    device_used = primary["device"]
                    parallel_enabled = primary["parallel_enabled"]
                    effective_compose_workers = primary["effective_compose_workers"]
        latency_ms = (time.time() - t0) * 1000.0
        run_mode_name = "随机后门" if mock_infer_mode else "模型推理"
        st.session_state["run_status"] = "success"
        st.session_state["run_message"] = f"运行完成（{run_mode_name}），耗时 {latency_ms:.1f} ms。"
        if run_status_placeholder is not None:
            run_status_placeholder.success(st.session_state["run_message"])

        st.session_state["last_result"] = {
            "bg": bg,
            "fg": fg,
            "base_fg_rgba": base_fg_rgba,
            "ranked": ranked,
            "images": images,
            "latency_ms": latency_ms,
            "orig_bg_size": orig_bg_size,
            "resized_bg_size": resized_bg_size,
            "device": device_used,
            "model_backend": model_backend,
            "inference_mode": inference_mode,
            "parallel_scale_search": bool(parallel_enabled),
            "cpu_compose_workers": int(cpu_compose_workers),
            "effective_compose_workers": int(effective_compose_workers),
            "compare_report": compare_report,
        }
        st.session_state["manual_score_result"] = None
        st.session_state["last_heatmap"] = hm
        st.session_state["last_heatmap_overlay"] = hm_overlay
        # Force a fresh render so the result tab can read the new state immediately.
        st.rerun()

# Legacy result block kept for fallback debugging; disabled in new tabbed workspace flow.
if False and "last_result" in st.session_state:
    res = st.session_state["last_result"]
    bg = res["bg"]
    fg = res["fg"]
    base_fg_rgba = res["base_fg_rgba"]
    ranked = res["ranked"]
    images = res["images"]
    latency_ms = res["latency_ms"]
    device_used = res.get("device", "cpu")
    model_backend_used = res.get("model_backend", REFERENCE_BACKEND)
    inference_mode_used = res.get("inference_mode", "热力图引导搜索（默认）")
    parallel_used = res.get("parallel_scale_search", False)
    cpu_workers = res.get("cpu_compose_workers", 1)
    effective_workers = res.get("effective_compose_workers", cpu_workers)
    compare_report = res.get("compare_report", [])
    spread = spread_summary(ranked)

    st.markdown(
        f"""
<div class="workspace-toolbar">
  <span class="workspace-chip">后端: {model_backend_used}</span>
  <span class="workspace-chip">策略: {inference_mode_used}</span>
  <span class="workspace-chip">设备: {device_used}</span>
  <span class="workspace-chip">总耗时: {latency_ms:.1f} ms</span>
  <span class="workspace-chip">分数 gap: {spread['gap']:.3f}</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_rankings, tab_manual, tab_heatmap, tab_export = st.tabs(
        ["工作区总览", "候选结果", "手动精修", "热力图", "导出"]
    )

    with tab_overview:
        left, right = st.columns([1.8, 1.2], gap="large")
        with left:
            st.markdown('<div class="workspace-panel">', unsafe_allow_html=True)
            if images:
                st.image(
                    images[0],
                    caption=f"Top1 分数={ranked[0]['score']:.3f}（{ranked[0]['level']}） scale={ranked[0]['scale']:.2f}",
                    width="stretch",
                )
            else:
                st.info("当前无可展示结果。")
            if st.session_state.get("last_heatmap_overlay") is not None:
                st.image(st.session_state["last_heatmap_overlay"], caption="热力图 + Top-K 标记", width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="workspace-panel">', unsafe_allow_html=True)
            st.caption(
                f"多尺度并行：{'开启' if parallel_used else '关闭'}；"
                f"CPU合成线程：{cpu_workers}（实际使用 {effective_workers}）"
            )
            st.caption(
                f"分数分布: min={spread['min']:.3f}, max={spread['max']:.3f}, "
                f"gap={spread['gap']:.3f}, std={spread['std']:.3f}"
            )
            if compare_report:
                st.caption("双后端对比（一次运行）")
                st.dataframe(compare_report, width="stretch", height=180)
            st.dataframe(ranked, width="stretch", height=260)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_rankings:
        if not images:
            st.info("暂无候选结果。")
        else:
            n_cols = min(3, max(1, len(images)))
            cols = st.columns(n_cols)
            for i, (row, img) in enumerate(zip(ranked, images)):
                col = cols[i % n_cols]
                with col:
                    st.image(
                        img,
                        caption=f"#{i+1} 分数={row['score']:.3f}（{row['level']}） | scale={row['scale']:.2f}",
                        width="stretch",
                    )
                    st.write(f"x={row['x']}, y={row['y']}")
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

    with tab_manual:
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

            fg_id_for_drag = _uploaded_file_id(fg_file) if fg_file is not None else "no_fg"
            drag_cache_key = (
                f"{fg_manual_drag.size[0]}x{fg_manual_drag.size[1]}|"
                f"{fg_id_for_drag}|bg:{bg.size[0]}x{bg.size[1]}"
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
            if st.session_state.get("manual_drag_state_key") != drag_cache_key:
                st.session_state["manual_drag_state_key"] = drag_cache_key
                st.session_state["manual_drag_x"] = manual_x
                st.session_state["manual_drag_y"] = manual_y
                st.session_state["manual_drag_canvas_key"] = f"manual_drag_canvas_image_{time.time_ns()}"
                st.session_state["manual_drag_json"] = init_image
                st.session_state["manual_drag_need_init"] = True
            if "manual_drag_canvas_key" not in st.session_state:
                st.session_state["manual_drag_canvas_key"] = f"manual_drag_canvas_image_{time.time_ns()}"
            if "manual_drag_json" not in st.session_state:
                st.session_state["manual_drag_json"] = init_image
            if "manual_drag_need_init" not in st.session_state:
                st.session_state["manual_drag_need_init"] = True

            saved_drag_json = st.session_state.get("manual_drag_json")
            saved_has_object = bool(
                isinstance(saved_drag_json, dict)
                and isinstance(saved_drag_json.get("objects"), list)
                and len(saved_drag_json.get("objects", [])) > 0
            )
            if not saved_has_object:
                saved_drag_json = init_image
                st.session_state["manual_drag_json"] = init_image
                st.session_state["manual_drag_need_init"] = True
            saved_obj = dict(saved_drag_json["objects"][0]) if saved_has_object else dict(init_image["objects"][0])
            saved_obj["src"] = fg_data_url
            saved_obj["width"] = float(fg_manual_drag.size[0])
            saved_obj["height"] = float(fg_manual_drag.size[1])
            saved_obj["lockRotation"] = True
            saved_obj["lockScalingX"] = True
            saved_obj["lockScalingY"] = True
            saved_drag_json = {
                "version": saved_drag_json.get("version", "4.4.0") if isinstance(saved_drag_json, dict) else "4.4.0",
                "objects": [saved_obj],
            }
            st.session_state["manual_drag_json"] = saved_drag_json
            inject_drawing = saved_drag_json if st.session_state.get("manual_drag_need_init", True) else None

            drag_canvas = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=1,
                stroke_color="#FFD400",
                background_image=bg,
                update_streamlit=True,
                width=bg.size[0],
                height=bg.size[1],
                drawing_mode="transform",
                initial_drawing=inject_drawing,
                display_toolbar=False,
                key=st.session_state["manual_drag_canvas_key"],
            )
            st.session_state["manual_drag_need_init"] = False
            current_drag_json = drag_canvas.json_data or st.session_state.get("manual_drag_json")
            if current_drag_json is not None:
                if isinstance(current_drag_json, dict) and isinstance(current_drag_json.get("objects"), list) and len(current_drag_json["objects"]) > 0:
                    first_obj = dict(current_drag_json["objects"][0])
                    first_obj["src"] = fg_data_url
                    first_obj["width"] = float(fg_manual_drag.size[0])
                    first_obj["height"] = float(fg_manual_drag.size[1])
                    first_obj["lockRotation"] = True
                    first_obj["lockScalingX"] = True
                    first_obj["lockScalingY"] = True
                    current_drag_json = {
                        "version": current_drag_json.get("version", "4.4.0"),
                        "objects": [first_obj],
                    }
                st.session_state["manual_drag_json"] = current_drag_json
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
            scorer = _cached_backend_scorer(model_backend=model_backend_used, device="auto")
            fg_manual = resize_foreground(base_fg_rgba, manual_scale * manual_bg_resize_factor)
            fg_manual = fit_foreground_to_background(fg_manual, bg)
            if use_drag_manual:
                current_drag_json = drag_canvas.json_data or st.session_state.get("manual_drag_json")
                manual_x, manual_y = _extract_image_xy_from_canvas(
                    json_data=current_drag_json,
                    default_x=int(st.session_state.get("manual_drag_x", manual_x)),
                    default_y=int(st.session_state.get("manual_drag_y", manual_y)),
                    max_x=max(0, bg.size[0] - fg_manual.size[0]),
                    max_y=max(0, bg.size[1] - fg_manual.size[1]),
                )
                st.session_state["manual_drag_x"] = manual_x
                st.session_state["manual_drag_y"] = manual_y
                if current_drag_json is not None:
                    st.session_state["manual_drag_json"] = current_drag_json
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
                "backend": model_backend_used,
            }

        if st.session_state.get("manual_score_result") is not None:
            cached = st.session_state["manual_score_result"]
            manual = cached["manual"]
            tips = cached["tips"]
            man_scale = float(cached.get("scale", 1.0))
            man_backend = cached.get("backend", model_backend_used)
            st.success(
                f"当前位置分数={manual['score']:.3f}（{manual['level']}），"
                f"x={manual['x']}, y={manual['y']}, scale={man_scale:.3f}，后端={man_backend}"
            )
            st.caption("；".join(tips))

    with tab_heatmap:
        st.subheader("位置热力图")
        if st.session_state.get("last_heatmap_overlay") is not None:
            hm = st.session_state.get("last_heatmap")
            overlay_with_marks = st.session_state["last_heatmap_overlay"]
            st.caption("说明：热力图基于网格粗采样，Top-K 位置经过局部精修，因此标记点可能不在单一网格峰值中心。")
            st.image(
                overlay_with_marks,
                caption=f"红色越强表示估计得分越高，已叠加 Top-K 标记点",
                width="stretch",
            )
            if hm is not None:
                st.caption(
                    f"热力图统计: min={float(hm.min()):.3f}, max={float(hm.max()):.3f}, "
                    f"gap={float(hm.max()-hm.min()):.3f}"
                )
        else:
            st.info("当前没有热力图，请先执行推荐。")

    with tab_export:
        st.subheader("导出结果")
        json_bytes = json.dumps(
            {
                "model_backend": model_backend_used,
                "inference_mode": inference_mode_used,
                "device": device_used,
                "ranked": ranked,
                "latency_ms": float(latency_ms),
                "spread": spread,
                "compare_report": compare_report,
            },
            ensure_ascii=False,
            indent=2,
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
            model_backend=model_backend_used,
            device_used=device_used,
            compare_report=compare_report,
        )
        st.download_button(
            label="下载结果包 (zip)",
            data=zip_bytes,
            file_name="simopa_results.zip",
            mime="application/zip",
        )

st.markdown("---")
st.write("运行入口: `streamlit run app.py`")
