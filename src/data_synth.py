from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.compositor import compose_rgba


def _random_background(size: int = 256) -> np.ndarray:
    sky = np.random.uniform(0.55, 0.95, size=3)
    ground = np.random.uniform(0.15, 0.75, size=3)
    split = np.random.randint(size // 3, size * 2 // 3)
    bg = np.zeros((size, size, 3), dtype=np.float32)
    bg[:split] = sky
    bg[split:] = ground
    for c in range(3):
        xv = np.linspace(0, np.random.uniform(0.75, 1.25), size)
        yv = np.linspace(0.8, np.random.uniform(0.9, 1.2), size)
        bg[..., c] *= np.outer(yv, xv)
    noise = np.random.normal(0, 0.025, bg.shape).astype(np.float32)
    return np.clip(bg + noise, 0.0, 1.0)


def _random_foreground(size: int = 96) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = tuple(np.random.randint(30, 220, size=3).tolist()) + (255,)
    if np.random.rand() > 0.5:
        draw.ellipse((10, 10, size - 10, size - 10), fill=color)
    else:
        draw.rounded_rectangle((8, 14, size - 8, size - 14), radius=14, fill=color)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    alpha = arr[..., 3:4]
    arr[..., :3] *= np.random.uniform(0.8, 1.2)
    arr[..., :3] = np.clip(arr[..., :3], 0.0, 1.0)
    arr[..., 3:4] = alpha
    return arr


def _placement_score(x: int, y: int, bg_size: int, fg_size: int) -> float:
    center = np.array([bg_size / 2.0, bg_size * 0.68], dtype=np.float32)
    obj = np.array([x + fg_size / 2.0, y + fg_size / 2.0], dtype=np.float32)
    dist = np.linalg.norm((obj - center) / bg_size)
    top_penalty = max(0.0, (bg_size * 0.45 - obj[1]) / bg_size)
    boundary_penalty = 0.0
    if x < 8 or y < 8 or x + fg_size > bg_size - 8 or y + fg_size > bg_size - 8:
        boundary_penalty += 0.4
    score = 1.05 - dist * 1.15 - top_penalty * 0.8 - boundary_penalty
    return float(np.clip(score, 0.0, 1.0))


def generate_synthetic_sample(bg_size: int = 192, fg_size: int = 64) -> Tuple[np.ndarray, float]:
    bg = _random_background(bg_size)
    fg = _random_foreground(fg_size)
    x = int(np.random.randint(0, bg_size - fg_size))
    y = int(np.random.randint(0, bg_size - fg_size))
    comp, mask = compose_rgba(bg, fg, x, y)
    label = _placement_score(x, y, bg_size, fg_size)
    inp = np.concatenate([comp, mask], axis=-1)
    return inp.astype(np.float32), label
