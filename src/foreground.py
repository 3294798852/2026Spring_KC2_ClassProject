import io
from typing import Literal, Tuple

import numpy as np
from PIL import Image, ImageFilter

_REMBG_SESSION = None
_REMBG_INIT_FAILED = False


def _get_rembg_session():
    global _REMBG_SESSION, _REMBG_INIT_FAILED
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    if _REMBG_INIT_FAILED:
        raise RuntimeError("rembg session unavailable")
    try:
        from rembg import new_session

        _REMBG_SESSION = new_session("u2net")
        return _REMBG_SESSION
    except Exception:
        _REMBG_INIT_FAILED = True
        raise


def _soft_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Fallback when segmentation weights are unavailable."""
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    rx, ry = w * 0.42, h * 0.46
    ellipse = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2) < 1.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    local = gray > np.percentile(gray, 40)
    mask = np.where(ellipse & local, 1.0, 0.0).astype(np.float32)
    return mask


def remove_background(
    image: Image.Image, target: Literal["person", "foreground"] = "person", threshold: float = 0.55
) -> Tuple[Image.Image, str]:
    """
    Remove image background and return RGBA image.
    Returns (image, info_message).
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    try:
        from rembg import remove

        session = _get_rembg_session()
        out_obj = remove(rgb, session=session)
        if isinstance(out_obj, Image.Image):
            rgba_img = out_obj.convert("RGBA")
        else:
            rgba_img = Image.open(io.BytesIO(out_obj)).convert("RGBA")
        rgba_arr = np.asarray(rgba_img, dtype=np.float32) / 255.0
        # target=person/foreground currently share same strong model; threshold controls alpha cleanup.
        alpha = (rgba_arr[..., 3] >= threshold).astype(np.float32)
        rgba_arr[..., 3] = alpha
        out = Image.fromarray((np.clip(rgba_arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGBA")
        info = "使用 U2Net(rembg) 自动抠图完成。"
        return out, info
    except Exception:
        alpha = _soft_foreground_mask(arr)
        info = "U2Net(rembg) 不可用，已使用快速备选抠图。"

    rgba = np.concatenate([arr, alpha[..., None]], axis=-1)
    out = Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8), mode="RGBA")
    return out, info


def apply_manual_alpha_mask(image: Image.Image, keep_mask: np.ndarray, feather: int = 1) -> Image.Image:
    """
    keep_mask: float mask in [0,1], 1 means keep foreground.
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    mask = np.clip(keep_mask.astype(np.float32), 0.0, 1.0)
    if mask.shape != rgba.shape[:2]:
        mask = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((rgba.shape[1], rgba.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    if feather > 1:
        mask = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=float(feather))),
            dtype=np.float32,
        ) / 255.0
    rgba[..., 3] = rgba[..., 3] * mask
    return Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8), mode="RGBA")


def _keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest 4-connected component in a binary mask.
    """
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    best_pixels = []

    for yy in range(h):
        for xx in range(w):
            if binary_mask[yy, xx] == 0 or visited[yy, xx] == 1:
                continue
            stack = [(yy, xx)]
            visited[yy, xx] = 1
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                if y > 0 and binary_mask[y - 1, x] == 1 and visited[y - 1, x] == 0:
                    visited[y - 1, x] = 1
                    stack.append((y - 1, x))
                if y < h - 1 and binary_mask[y + 1, x] == 1 and visited[y + 1, x] == 0:
                    visited[y + 1, x] = 1
                    stack.append((y + 1, x))
                if x > 0 and binary_mask[y, x - 1] == 1 and visited[y, x - 1] == 0:
                    visited[y, x - 1] = 1
                    stack.append((y, x - 1))
                if x < w - 1 and binary_mask[y, x + 1] == 1 and visited[y, x + 1] == 0:
                    visited[y, x + 1] = 1
                    stack.append((y, x + 1))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels

    out = np.zeros_like(binary_mask, dtype=np.uint8)
    for y, x in best_pixels:
        out[y, x] = 1
    return out


def crop_rgba_by_alpha(image: Image.Image, alpha_threshold: int = 8, padding: int = 4) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = arr[..., 3]
    ys, xs = np.where(alpha >= int(alpha_threshold))
    if len(xs) == 0 or len(ys) == 0:
        return image.convert("RGBA")
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(arr.shape[1], int(xs.max()) + 1 + padding)
    y2 = min(arr.shape[0], int(ys.max()) + 1 + padding)
    crop = arr[y1:y2, x1:x2]
    return Image.fromarray(crop, mode="RGBA")


def refine_rgba_cutout(
    image: Image.Image,
    alpha_threshold: int = 12,
    keep_largest: bool = True,
    feather_radius: int = 1,
    auto_crop: bool = False,
    crop_padding: int = 4,
    invert_mask: bool = False,
) -> Image.Image:
    # np.asarray(PIL.Image) may return a read-only view; take a writable copy.
    arr = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
    alpha = arr[..., 3]
    mask = (alpha >= int(alpha_threshold)).astype(np.uint8)
    if invert_mask:
        mask = 1 - mask
    if keep_largest and mask.sum() > 0:
        mask = _keep_largest_component(mask)

    if feather_radius > 1:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=float(feather_radius))
        )
        mask = (np.asarray(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
        arr[..., 3] = (arr[..., 3].astype(np.float32) * mask).astype(np.uint8)
    else:
        arr[..., 3] = (arr[..., 3] * mask).astype(np.uint8)

    out = Image.fromarray(arr, mode="RGBA")
    if auto_crop:
        out = crop_rgba_by_alpha(out, alpha_threshold=max(1, alpha_threshold // 2), padding=crop_padding)
    return out


def resize_foreground(foreground_rgba: Image.Image, scale: float) -> Image.Image:
    scale = max(0.1, min(3.0, float(scale)))
    w, h = foreground_rgba.size
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    return foreground_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)


def fit_foreground_to_background(
    foreground_rgba: Image.Image, background_rgb: Image.Image, max_ratio: float = 0.85
) -> Image.Image:
    fg_w, fg_h = foreground_rgba.size
    bg_w, bg_h = background_rgb.size
    max_w = int(bg_w * max_ratio)
    max_h = int(bg_h * max_ratio)
    if fg_w <= max_w and fg_h <= max_h:
        return foreground_rgba
    scale = min(max_w / max(fg_w, 1), max_h / max(fg_h, 1))
    new_w = max(16, int(fg_w * scale))
    new_h = max(16, int(fg_h * scale))
    return foreground_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
