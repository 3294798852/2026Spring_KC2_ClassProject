import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opa import BACKENDS, create_opa_scorer


def _auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return wins / float(pos.size * neg.size)


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    denom = (2.0 * tp + fp + fn)
    if denom <= 1e-8:
        return 0.0
    return (2.0 * tp) / denom


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


def _extract_group_key(img_name: str) -> str:
    stem = Path(img_name).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def _resolve_data_path(dataset_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [
        dataset_root / path,
        dataset_root / Path(*path.parts[1:]) if path.parts and path.parts[0] == "dataset" else None,
        ROOT / path,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve dataset path: {raw_path}")


def evaluate_backend(dataset_root: Path, csv_path: Path, backend: str, device: str, batch_size: int) -> dict:
    scorer = create_opa_scorer(model_backend=backend, device=device)
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")

    y_true: list[int] = []
    y_score: list[float] = []
    group_scores: dict[str, list[tuple[int, float]]] = {}

    for i in range(0, len(rows), max(1, int(batch_size))):
        batch_rows = rows[i : i + max(1, int(batch_size))]
        composites = []
        masks = []
        meta = []
        for row in batch_rows:
            img_path = _resolve_data_path(dataset_root, row["img_name"])
            mask_path = _resolve_data_path(dataset_root, row["mask_name"])
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
            composites.append(img)
            masks.append(m[..., None])
            meta.append((int(row["label"]), _extract_group_key(row["img_name"])))
        scores = scorer.score_batch(composites, masks)
        for (label, gk), score in zip(meta, scores):
            y_true.append(label)
            y_score.append(float(score))
            group_scores.setdefault(gk, []).append((label, float(score)))

    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float32)
    yp = (ys >= 0.5).astype(np.int64)
    acc = float((yp == yt).mean())
    auc = float(_auc_binary(yt, ys))
    f1 = float(_f1_binary(yt, yp))

    pair_total = 0
    pair_ok = 0
    scene_rhos: list[float] = []
    for pairs in group_scores.values():
        if len(pairs) < 2:
            continue
        labels = np.asarray([p[0] for p in pairs], dtype=np.float32)
        scores = np.asarray([p[1] for p in pairs], dtype=np.float32)
        if len(np.unique(labels)) > 1:
            scene_rhos.append(_spearman(scores, labels))
        n = len(pairs)
        for i in range(n):
            for j in range(i + 1, n):
                li, si = pairs[i]
                lj, sj = pairs[j]
                if li == lj:
                    continue
                pair_total += 1
                if (li > lj and si > sj) or (lj > li and sj > si):
                    pair_ok += 1
                elif abs(si - sj) <= 1e-8:
                    pair_ok += 0.5
    pairwise_acc = float(pair_ok / pair_total) if pair_total > 0 else 0.0
    scene_spearman = float(np.mean(scene_rhos)) if scene_rhos else 0.0
    params_m = sum(p.numel() for p in scorer.model.parameters()) / 1e6 if hasattr(scorer, "model") else 0.0

    return {
        "backend": backend,
        "samples": int(len(yt)),
        "acc@0.5": acc,
        "auc": auc,
        "f1@0.5": f1,
        "pairwise_rank_acc": pairwise_acc,
        "scene_spearman": scene_spearman,
        "params_m": float(params_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OPA backends on labeled CSV with grounded metrics.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "new_OPA")
    parser.add_argument("--csv", type=Path, default=ROOT / "new_OPA" / "test_set.csv")
    parser.add_argument("--backend", default=BACKENDS[0], choices=BACKENDS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    targets = BACKENDS if args.compare_all else [args.backend]
    report = [
        evaluate_backend(
            dataset_root=args.data_root,
            csv_path=args.csv,
            backend=backend,
            device=args.device,
            batch_size=args.batch_size,
        )
        for backend in targets
    ]
    for item in report:
        print(
            f"{item['backend']}: acc={item['acc@0.5']:.4f} auc={item['auc']:.4f} "
            f"f1={item['f1@0.5']:.4f} pair_rank={item['pairwise_rank_acc']:.4f} "
            f"scene_rho={item['scene_spearman']:.4f} params={item['params_m']:.3f}M"
        )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report saved to {out}")


if __name__ == "__main__":
    main()

