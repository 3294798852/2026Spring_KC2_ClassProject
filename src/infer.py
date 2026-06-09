from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.compositor import Candidate, compose_rgba, generate_candidates, pil_to_rgb_np, pil_to_rgba_np
from src.reference_opa import ReferenceOPAScorer
from src.scoring import classify_score


def _chunked(items: List[Tuple[int, int]], chunk_size: int) -> List[List[Tuple[int, int]]]:
    chunk_size = max(1, int(chunk_size))
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _clip_position(x: int, y: int, max_x: int, max_y: int) -> Tuple[int, int]:
    return int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))


def _compose_many(
    background_rgb: np.ndarray,
    foreground_rgba: np.ndarray,
    positions: List[Tuple[int, int]],
    max_workers: int = 1,
    executor: ThreadPoolExecutor | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    max_workers = max(1, int(max_workers))
    if max_workers == 1 or len(positions) <= 1:
        pairs = [compose_rgba(background_rgb, foreground_rgba, x, y) for x, y in positions]
    else:
        def compose_one(pos: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
            x, y = pos
            return compose_rgba(background_rgb, foreground_rgba, x, y)

        if executor is None:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(positions))) as local_executor:
                pairs = list(local_executor.map(compose_one, positions))
        else:
            pairs = list(executor.map(compose_one, positions))
    return [p[0] for p in pairs], [p[1] for p in pairs]


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


def rank_candidates(
    background: Image.Image,
    foreground: Image.Image,
    top_k: int = 5,
    candidate_count: int = 12,
    scale_tag: float = 1.0,
    scorer: ReferenceOPAScorer | None = None,
    compose_workers: int = 1,
) -> Tuple[List[Dict], List[Image.Image]]:
    bg = pil_to_rgb_np(background)
    fg = pil_to_rgba_np(foreground)
    bg_h, bg_w, _ = bg.shape
    fg_h, fg_w, _ = fg.shape
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)

    score_cache: Dict[Tuple[int, int], float] = {}
    simopa_scorer = scorer if scorer is not None else ReferenceOPAScorer(device="cpu")
    compose_workers = max(1, int(compose_workers))
    score_chunk_size = max(6, compose_workers * 4)
    compose_executor = (
        ThreadPoolExecutor(max_workers=compose_workers) if compose_workers > 1 else None
    )

    def score_positions(positions: List[Tuple[int, int]], chunk_size: int = score_chunk_size) -> List[float]:
        clipped = [_clip_position(x, y, max_x, max_y) for x, y in positions]
        pending = [p for p in clipped if p not in score_cache]
        if pending:
            for chunk in _chunked(pending, chunk_size):
                composites, masks = _compose_many(
                    bg,
                    fg,
                    chunk,
                    max_workers=compose_workers,
                    executor=compose_executor,
                )
                new_scores = simopa_scorer.score_batch(composites, masks)
                for pos, score in zip(chunk, new_scores):
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
    # Add scale metadata for multi-scale merge.
    for item in out:
        item["scale"] = float(scale_tag)
    
    if compose_executor is not None:
        compose_executor.shutdown(wait=True)
    
    return out, top_rendered


def score_single_position(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    scorer: ReferenceOPAScorer | None = None,
) -> Dict:
    bg = pil_to_rgb_np(background)
    fg = pil_to_rgba_np(foreground)
    bg_h, bg_w, _ = bg.shape
    fg_h, fg_w, _ = fg.shape
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)
    x, y = _clip_position(int(x), int(y), max_x, max_y)

    comp, mask = compose_rgba(bg, fg, x, y)
    sc = scorer if scorer is not None else ReferenceOPAScorer(device="cpu")
    score = float(sc.score_batch([comp], [mask])[0])
    return {
        "x": x,
        "y": y,
        "score": score,
        "level": classify_score(score),
        "image": Image.fromarray((comp * 255).astype(np.uint8)),
    }


def score_heatmap(
    background: Image.Image,
    foreground: Image.Image,
    grid_size: int = 14,
    scorer: ReferenceOPAScorer | None = None,
) -> np.ndarray:
    bg = pil_to_rgb_np(background)
    fg = pil_to_rgba_np(foreground)
    bg_h, bg_w, _ = bg.shape
    fg_h, fg_w, _ = fg.shape
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)
    grid_size = max(6, int(grid_size))

    xs = np.linspace(0, max_x, num=grid_size, dtype=int)
    ys = np.linspace(0, max_y, num=grid_size, dtype=int)
    pos = [(int(x), int(y)) for y in ys for x in xs]

    sc = scorer if scorer is not None else ReferenceOPAScorer(device="cpu")
    scores: List[float] = []
    for chunk in _chunked(pos, chunk_size=4):
        composites: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for x, y in chunk:
            comp, mask = compose_rgba(bg, fg, x, y)
            composites.append(comp)
            masks.append(mask)
        scores.extend(sc.score_batch(composites, masks))
    arr = np.array(scores, dtype=np.float32).reshape(grid_size, grid_size)
    return arr


def rank_candidates_heatmap_guided(
    background: Image.Image,
    foreground: Image.Image,
    top_k: int = 5,
    candidate_count: int = 48,
    heatmap_grid: int = 18,
    scale_tag: float = 1.0,
    scorer: ReferenceOPAScorer | None = None,
    compose_workers: int = 1,
) -> Tuple[List[Dict], List[Image.Image], np.ndarray]:
    """
    Unified pipeline:
    1) compute heatmap samples first;
    2) use high-score heatmap points as search seeds;
    3) add supplemental seeds when budget is large;
    4) return ranking + rendered top-k + heatmap.
    """
    bg = pil_to_rgb_np(background)
    fg = pil_to_rgba_np(foreground)
    bg_h, bg_w, _ = bg.shape
    fg_h, fg_w, _ = fg.shape
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)
    heatmap_grid = max(6, int(heatmap_grid))
    candidate_count = max(12, int(candidate_count))

    score_cache: Dict[Tuple[int, int], float] = {}
    simopa_scorer = scorer if scorer is not None else ReferenceOPAScorer(device="cpu")
    compose_workers = max(1, int(compose_workers))
    score_chunk_size = max(6, compose_workers * 4)
    compose_executor = (
        ThreadPoolExecutor(max_workers=compose_workers) if compose_workers > 1 else None
    )

    def score_positions(positions: List[Tuple[int, int]], chunk_size: int = score_chunk_size) -> List[float]:
        clipped = [_clip_position(x, y, max_x, max_y) for x, y in positions]
        pending = [p for p in clipped if p not in score_cache]
        if pending:
            for chunk in _chunked(pending, chunk_size):
                composites, masks = _compose_many(
                    bg,
                    fg,
                    chunk,
                    max_workers=compose_workers,
                    executor=compose_executor,
                )
                new_scores = simopa_scorer.score_batch(composites, masks)
                for pos, score in zip(chunk, new_scores):
                    score_cache[pos] = float(score)
        return [score_cache[p] for p in clipped]

    xs = np.linspace(0, max_x, num=heatmap_grid, dtype=int)
    ys = np.linspace(0, max_y, num=heatmap_grid, dtype=int)
    grid_positions = [(int(x), int(y)) for y in ys for x in xs]
    grid_scores = score_positions(grid_positions, chunk_size=6)
    heatmap = np.array(grid_scores, dtype=np.float32).reshape(heatmap_grid, heatmap_grid)
    ranked_grid = sorted(zip(grid_positions, grid_scores), key=lambda t: t[1], reverse=True)

    elite_count = min(len(ranked_grid), max(8, candidate_count // 2))
    elite_positions = [ranked_grid[i][0] for i in range(elite_count)]
    step_x = max(2, max_x // max(1, heatmap_grid - 1))
    step_y = max(2, max_y // max(1, heatmap_grid - 1))
    init_step = max(4, int(max(step_x, step_y) * 0.8))

    refined_positions: List[Tuple[int, int]] = []
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

    all_positions = grid_positions + refined_positions
    if len(all_positions) < candidate_count:
        all_positions.extend(_generate_seed_positions(bg_w, bg_h, fg_w, fg_h, candidate_count))
    all_scores = score_positions(all_positions, chunk_size=6)

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
    for item in out:
        item["scale"] = float(scale_tag)
    
    if compose_executor is not None:
        compose_executor.shutdown(wait=True)
    
    return out, top_rendered, heatmap
