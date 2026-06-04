from typing import Literal, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image


_SEGMENTATION_MODEL = None
_SEGMENTATION_DEVICE = "cpu"
_SEGMENTATION_FAILED = False


def _get_segmentation_model() -> torch.nn.Module:
    global _SEGMENTATION_MODEL, _SEGMENTATION_FAILED
    if _SEGMENTATION_MODEL is not None:
        return _SEGMENTATION_MODEL
    if _SEGMENTATION_FAILED:
        raise RuntimeError("segmentation model unavailable")
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    try:
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(_SEGMENTATION_DEVICE)
        model.eval()
        _SEGMENTATION_MODEL = model
        return model
    except Exception:
        _SEGMENTATION_FAILED = True
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
        model = _get_segmentation_model()
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        tensor = weights.transforms()(rgb).unsqueeze(0).to(_SEGMENTATION_DEVICE)
        with torch.no_grad():
            logits = model(tensor)["out"][0]
        probs = logits.softmax(0).cpu().numpy()
        if target == "person":
            # PASCAL VOC class index for person.
            mask = probs[15]
        else:
            # Any non-background as foreground.
            mask = 1.0 - probs[0]
        if mask.shape != arr.shape[:2]:
            mask = np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (arr.shape[1], arr.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        alpha = (mask >= threshold).astype(np.float32)
        info = "使用 DeepLabV3 自动抠图完成。"
    except Exception:
        alpha = _soft_foreground_mask(arr)
        info = "DeepLabV3 不可用，已使用快速备选抠图。"

    rgba = np.concatenate([arr, alpha[..., None]], axis=-1)
    out = Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8), mode="RGBA")
    return out, info


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
