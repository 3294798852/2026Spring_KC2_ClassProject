import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.foreground import fit_foreground_to_background, resize_foreground
from src.infer import rank_candidates
from src.reference_opa import ensure_simopa_weight


def evaluate_pair(bg_path: Path, fg_path: Path, candidate_count: int, top_k: int, scale: float) -> None:
    bg = Image.open(bg_path).convert("RGB")
    fg = Image.open(fg_path).convert("RGBA")
    fg = fit_foreground_to_background(resize_foreground(fg, scale), bg)

    t0 = time.time()
    rows, _ = rank_candidates(bg, fg, top_k=top_k, candidate_count=candidate_count, scale_tag=scale)
    elapsed = (time.time() - t0) * 1000.0
    scores = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    print(f"top-{top_k} latency_ms={elapsed:.1f}")
    print(f"score min={scores.min():.4f} max={scores.max():.4f} gap={(scores.max()-scores.min()):.4f} std={scores.std():.4f}")
    for i, r in enumerate(rows):
        print(f"#{i+1} score={r['score']:.4f} level={r['level']} x={r['x']} y={r['y']} scale={r['scale']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SimOPA ranking quality on a real bg/fg pair.")
    parser.add_argument("--bg", required=True, help="background image path")
    parser.add_argument("--fg", required=True, help="foreground image path (RGBA preferred)")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    ensure_simopa_weight()
    evaluate_pair(
        bg_path=Path(args.bg),
        fg_path=Path(args.fg),
        candidate_count=max(6, int(args.candidate_count)),
        top_k=max(1, int(args.top_k)),
        scale=max(0.3, min(2.5, float(args.scale))),
    )
