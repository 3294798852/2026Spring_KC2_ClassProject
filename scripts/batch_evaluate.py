import argparse
import csv
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
from src.opa import BACKENDS, create_opa_scorer
from src.reference_opa import ensure_simopa_weight


def collect_pairs(bg_dir: Path, fg_dir: Path):
    bg_map = {p.stem: p for p in bg_dir.iterdir() if p.is_file()}
    fg_map = {p.stem: p for p in fg_dir.iterdir() if p.is_file()}
    common = sorted(set(bg_map.keys()) & set(fg_map.keys()))
    return [(k, bg_map[k], fg_map[k]) for k in common]


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate SimOPA ranking and export CSV summary.")
    parser.add_argument("--bg-dir", required=True)
    parser.add_argument("--fg-dir", required=True)
    parser.add_argument("--out-csv", default="batch_eval_results.csv")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--backend", default=BACKENDS[1], choices=BACKENDS)
    parser.add_argument("--compare-both", action="store_true")
    args = parser.parse_args()

    bg_dir = Path(args.bg_dir)
    fg_dir = Path(args.fg_dir)
    out_csv = Path(args.out_csv)

    ensure_simopa_weight()
    pairs = collect_pairs(bg_dir, fg_dir)
    if not pairs:
        raise RuntimeError("no paired samples found. ensure bg/fg filenames share same stem.")

    rows = []
    backends = BACKENDS if args.compare_both else [args.backend]
    for backend in backends:
        scorer = create_opa_scorer(model_backend=backend, device="cpu")
        params_m = sum(p.numel() for p in scorer.model.parameters()) / 1e6 if hasattr(scorer, "model") else 0.0
        for stem, bg_path, fg_path in pairs:
            bg = Image.open(bg_path).convert("RGB")
            fg = Image.open(fg_path).convert("RGBA")
            fg = fit_foreground_to_background(resize_foreground(fg, args.scale), bg)
            t0 = time.time()
            ranked, _ = rank_candidates(
                bg,
                fg,
                top_k=max(1, args.top_k),
                candidate_count=max(6, args.candidate_count),
                scale_tag=args.scale,
                scorer=scorer,
            )
            latency_ms = (time.time() - t0) * 1000.0
            scores = np.array([float(r["score"]) for r in ranked], dtype=np.float32)
            rows.append(
                {
                    "backend": backend,
                    "sample": stem,
                    "latency_ms": f"{latency_ms:.2f}",
                    "params_m": f"{params_m:.3f}",
                    "top1_score": f"{scores[0]:.4f}",
                    "topk_min": f"{float(scores.min()):.4f}",
                    "topk_max": f"{float(scores.max()):.4f}",
                    "topk_gap": f"{float(scores.max() - scores.min()):.4f}",
                    "topk_std": f"{float(scores.std()):.4f}",
                    "top1_x": int(ranked[0]["x"]),
                    "top1_y": int(ranked[0]["y"]),
                    "top1_scale": f"{float(ranked[0]['scale']):.2f}",
                }
            )
            print(
                f"[{backend}::{stem}] top1={scores[0]:.4f} gap={float(scores.max()-scores.min()):.4f} "
                f"latency={latency_ms:.1f}ms params={params_m:.3f}M"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done. {len(rows)} samples saved to {out_csv}")


if __name__ == "__main__":
    main()
