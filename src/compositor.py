from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class Candidate:
    x: int
    y: int
    score: float = 0.0
    level: str = "未知"


def pil_to_rgb_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def pil_to_rgba_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0


def generate_candidates(bg_w: int, bg_h: int, fg_w: int, fg_h: int, count: int = 12) -> List[Candidate]:
    grid_cols = max(2, int(np.sqrt(count)))
    grid_rows = max(2, int(np.ceil(count / grid_cols)))
    margin_x = max(5, fg_w // 6)
    margin_y = max(5, fg_h // 6)

    max_x = max(1, bg_w - fg_w - margin_x)
    max_y = max(1, bg_h - fg_h - margin_y)
    xs = np.linspace(margin_x, max_x, num=grid_cols, dtype=int)
    ys = np.linspace(margin_y, max_y, num=grid_rows, dtype=int)

    out: List[Candidate] = []
    for y in ys:
        for x in xs:
            out.append(Candidate(x=int(x), y=int(y)))
            if len(out) >= count:
                return out
    return out


def compose_rgba(
    background_rgb: np.ndarray, foreground_rgba: np.ndarray, x: int, y: int
) -> Tuple[np.ndarray, np.ndarray]:
    bg_h, bg_w, _ = background_rgb.shape
    fg_h, fg_w, _ = foreground_rgba.shape

    x = int(np.clip(x, 0, max(0, bg_w - fg_w)))
    y = int(np.clip(y, 0, max(0, bg_h - fg_h)))

    fg_rgb = foreground_rgba[..., :3]
    alpha = foreground_rgba[..., 3:4]
    comp = background_rgb.copy()
    mask = np.zeros((bg_h, bg_w, 1), dtype=np.float32)

    roi = comp[y : y + fg_h, x : x + fg_w]
    comp[y : y + fg_h, x : x + fg_w] = fg_rgb * alpha + roi * (1.0 - alpha)
    mask[y : y + fg_h, x : x + fg_w] = alpha
    return comp, mask
