import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.foreground import fit_foreground_to_background, resize_foreground
from src.infer import rank_candidates, rank_candidates_dense_map, rank_candidates_heatmap_guided
from src.opa import BACKENDS, create_opa_scorer
from src.reference_opa import ensure_simopa_weight


def _auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return wins / float(pos.size * neg.size)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum()) + 1e-8
    return float((rx * ry).sum() / denom)


def evaluate_pair(
    bg_path: Path,
    fg_path: Path,
    candidate_count: int,
    top_k: int,
    scale: float,
    model_backend: str,
    search_mode: str,
    heatmap_grid: int,
) -> dict:
    bg = Image.open(bg_path).convert("RGB")
    fg = Image.open(fg_path).convert("RGBA")
    fg = fit_foreground_to_background(resize_foreground(fg, scale), bg)
    scorer = create_opa_scorer(model_backend=model_backend, device="auto")

    t0 = time.time()
    if search_mode == "dense":
        rows, _, _ = rank_candidates_dense_map(
            bg,
            fg,
            top_k=top_k,
            heatmap_grid=max(8, int(heatmap_grid)),
            refine_per_point=6,
            scale_tag=scale,
            scorer=scorer,
        )
    elif search_mode == "legacy":
        rows, _ = rank_candidates(
            bg,
            fg,
            top_k=top_k,
            candidate_count=candidate_count,
            scale_tag=scale,
            scorer=scorer,
        )
    else:
        rows, _, _ = rank_candidates_heatmap_guided(
            bg,
            fg,
            top_k=top_k,
            candidate_count=candidate_count,
            heatmap_grid=max(8, int(heatmap_grid)),
            scale_tag=scale,
            scorer=scorer,
        )
    elapsed = (time.time() - t0) * 1000.0
    scores = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    # Sanity-only metrics (not label-grounded quality metrics).
    y_true = np.zeros_like(scores, dtype=np.int64)
    y_true[np.argsort(scores)[len(scores) // 2 :]] = 1
    y_score = scores
    auc = _auc_binary(y_true, y_score)
    rank_ref = np.arange(len(scores), 0, -1, dtype=np.float32)
    rho = _spearman(scores, rank_ref)
    params_m = sum(p.numel() for p in scorer.model.parameters()) / 1e6 if hasattr(scorer, "model") else 0.0

    print(f"backend={model_backend} top-{top_k} latency_ms={elapsed:.1f}")
    print(
        f"score min={scores.min():.4f} max={scores.max():.4f} gap={(scores.max()-scores.min()):.4f} std={scores.std():.4f}"
    )
    print(f"sanity_auc={auc:.4f} sanity_spearman_vs_rank={rho:.4f} params_m={params_m:.3f}")
    for i, r in enumerate(rows):
        print(f"#{i+1} score={r['score']:.4f} level={r['level']} x={r['x']} y={r['y']} scale={r['scale']:.2f}")
    return {
        "backend": model_backend,
        "latency_ms": float(elapsed),
        "params_m": float(params_m),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_gap": float(scores.max() - scores.min()),
        "score_std": float(scores.std()),
        "sanity_auc": float(auc),
        "sanity_spearman_vs_rank": float(rho),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SimOPA ranking quality on a real bg/fg pair.")
    parser.add_argument("--bg", required=True, help="background image path")
    parser.add_argument("--fg", required=True, help="foreground image path (RGBA preferred)")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--backend", default=BACKENDS[0], choices=BACKENDS)
    parser.add_argument("--search-mode", default="heatmap", choices=["heatmap", "dense", "legacy"])
    parser.add_argument("--heatmap-grid", type=int, default=18)
    parser.add_argument("--compare-both", action="store_true", help="legacy alias, compare all registered backends")
    parser.add_argument("--compare-all", action="store_true", help="run all backends in one report")
    parser.add_argument("--out-json", default=None, help="optional output report json path")
    args = parser.parse_args()

    ensure_simopa_weight()
    print("note: this script is for placement-search sanity. For label-grounded metrics, use scripts/evaluate_dataset_metrics.py")
    targets = BACKENDS if (args.compare_both or args.compare_all) else [args.backend]
    report = []
    for backend in targets:
        report.append(
            evaluate_pair(
                bg_path=Path(args.bg),
                fg_path=Path(args.fg),
                candidate_count=max(6, int(args.candidate_count)),
                top_k=max(1, int(args.top_k)),
                scale=max(0.3, min(2.5, float(args.scale))),
                model_backend=backend,
                search_mode=args.search_mode,
                heatmap_grid=args.heatmap_grid,
            )
        )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report saved to {out}")
