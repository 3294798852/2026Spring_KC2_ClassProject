from dataclasses import asdict
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from PIL import Image

from src.compositor import Candidate, compose_rgba, generate_candidates, pil_to_rgb_np, pil_to_rgba_np
from src.compress import quantize_head
from src.config import COMPRESSED_PATH, STUDENT_PATH
from src.models import PlacementStudent, classify_score
from src.reference_opa import ReferenceOPAScorer


def _to_tensor(comp: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    inp = np.concatenate([comp, mask], axis=-1)
    inp = np.transpose(inp, (2, 0, 1))
    return torch.tensor(inp, dtype=torch.float32).unsqueeze(0)


def _clip_position(x: int, y: int, max_x: int, max_y: int) -> Tuple[int, int]:
    return int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))


def _generate_seed_positions(
    bg_w: int, bg_h: int, fg_w: int, fg_h: int, candidate_count: int
) -> List[Tuple[int, int]]:
    # Start with broad coverage then add jittered points.
    base_count = max(36, candidate_count * 3)
    base = generate_candidates(bg_w, bg_h, fg_w, fg_h, count=base_count)

    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)
    rng = np.random.default_rng(bg_w * 73856093 + bg_h * 19349663 + fg_w * 83492791 + fg_h * 29791)
    jitter = max(6, min(fg_w, fg_h) // 5)

    out: List[Tuple[int, int]] = []
    for c in base:
        out.append(_clip_position(c.x, c.y, max_x, max_y))
        out.append(
            _clip_position(
                c.x + int(rng.integers(-jitter, jitter + 1)),
                c.y + int(rng.integers(-jitter, jitter + 1)),
                max_x,
                max_y,
            )
        )
    return out


def _select_diverse_top(rows: List[Candidate], top_k: int, min_dist: float) -> List[Candidate]:
    selected: List[Candidate] = []
    for c in sorted(rows, key=lambda z: z.score, reverse=True):
        if len(selected) >= top_k:
            break
        if all(np.hypot(c.x - s.x, c.y - s.y) >= min_dist for s in selected):
            selected.append(c)

    if len(selected) < top_k:
        seen = {(s.x, s.y) for s in selected}
        for c in sorted(rows, key=lambda z: z.score, reverse=True):
            if len(selected) >= top_k:
                break
            if (c.x, c.y) not in seen:
                selected.append(c)
                seen.add((c.x, c.y))
    # Keep final results strictly sorted by score to avoid rank-order confusion.
    selected = sorted(selected, key=lambda z: z.score, reverse=True)
    return selected[:top_k]


def load_legacy_infer_model(prefer_compressed: bool = True) -> torch.nn.Module:
    if prefer_compressed and COMPRESSED_PATH.exists():
        obj = torch.load(COMPRESSED_PATH, map_location="cpu")
        if isinstance(obj, dict) and obj.get("quantized_head", False):
            model = quantize_head(PlacementStudent())
            model.load_state_dict(obj["state_dict"])
            model.eval()
            return model
        if isinstance(obj, dict) and "state_dict" in obj:
            model = PlacementStudent()
            model.load_state_dict(obj["state_dict"])
            model.eval()
            return model

    if not STUDENT_PATH.exists():
        raise FileNotFoundError(
            f"missing model weights: {STUDENT_PATH}. please run `python scripts/bootstrap_and_compress.py` first."
        )

    model = PlacementStudent()
    ckpt = torch.load(STUDENT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def rank_candidates(
    background: Image.Image,
    foreground: Image.Image,
    top_k: int = 5,
    candidate_count: int = 12,
    prefer_compressed: bool = False,
    model_backend: Literal["simopa", "legacy"] = "simopa",
) -> Tuple[List[Dict], List[Image.Image]]:
    bg = pil_to_rgb_np(background)
    fg = pil_to_rgba_np(foreground)
    bg_h, bg_w, _ = bg.shape
    fg_h, fg_w, _ = fg.shape
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)

    score_cache: Dict[Tuple[int, int], float] = {}
    simopa_scorer = ReferenceOPAScorer(device="cpu") if model_backend == "simopa" else None
    legacy_model = load_legacy_infer_model(prefer_compressed=prefer_compressed) if model_backend != "simopa" else None

    def score_positions(positions: List[Tuple[int, int]]) -> List[float]:
        clipped = [_clip_position(x, y, max_x, max_y) for x, y in positions]
        pending = [p for p in clipped if p not in score_cache]
        if pending:
            composites: List[np.ndarray] = []
            masks: List[np.ndarray] = []
            for x, y in pending:
                comp, mask = compose_rgba(bg, fg, x, y)
                composites.append(comp)
                masks.append(mask)
            if simopa_scorer is not None:
                new_scores = simopa_scorer.score_batch(composites, masks)
            else:
                assert legacy_model is not None
                with torch.no_grad():
                    new_scores = [
                        float(legacy_model(_to_tensor(comp, mask)).item()) for comp, mask in zip(composites, masks)
                    ]
            for pos, score in zip(pending, new_scores):
                score_cache[pos] = float(score)
        return [score_cache[p] for p in clipped]

    # Stage 1: broad search.
    seed_positions = _generate_seed_positions(bg_w, bg_h, fg_w, fg_h, candidate_count)
    seed_scores = score_positions(seed_positions)
    ranked_seed = sorted(zip(seed_positions, seed_scores), key=lambda t: t[1], reverse=True)

    # Stage 2: local pattern search around elites.
    elite_count = min(10, max(4, candidate_count // 2), len(ranked_seed))
    elite_positions = [ranked_seed[i][0] for i in range(elite_count)]
    refined_positions: List[Tuple[int, int]] = []
    init_step = max(6, min(bg_w, bg_h) // 10)
    for start in elite_positions:
        cur_x, cur_y = start
        cur_score = score_positions([(cur_x, cur_y)])[0]
        step = init_step
        while step >= 2:
            neighbors = [
                (cur_x + step, cur_y),
                (cur_x - step, cur_y),
                (cur_x, cur_y + step),
                (cur_x, cur_y - step),
                (cur_x + step, cur_y + step),
                (cur_x + step, cur_y - step),
                (cur_x - step, cur_y + step),
                (cur_x - step, cur_y - step),
            ]
            neighbors = [_clip_position(x, y, max_x, max_y) for x, y in neighbors]
            neigh_scores = score_positions(neighbors)
            best_idx = int(np.argmax(neigh_scores))
            if neigh_scores[best_idx] > cur_score + 1e-6:
                cur_x, cur_y = neighbors[best_idx]
                cur_score = neigh_scores[best_idx]
            else:
                step //= 2
        refined_positions.append((cur_x, cur_y))

    all_positions = seed_positions + refined_positions
    all_scores = score_positions(all_positions)

    rows: List[Candidate] = []
    seen = set()
    for (x, y), score in sorted(zip(all_positions, all_scores), key=lambda t: t[1], reverse=True):
        if (x, y) in seen:
            continue
        seen.add((x, y))
        rows.append(Candidate(x=x, y=y, score=float(score), level=classify_score(float(score))))

    rows = rows[: max(candidate_count * 2, 20)]
    min_dist = max(10.0, min(fg_w, fg_h) * 0.35)
    top_rows = _select_diverse_top(rows, top_k=top_k, min_dist=min_dist)
    out = [asdict(r) for r in top_rows]

    top_rendered: List[Image.Image] = []
    for r in top_rows:
        comp, _ = compose_rgba(bg, fg, r.x, r.y)
        top_rendered.append(Image.fromarray((comp * 255).astype(np.uint8)))
    return out, top_rendered
