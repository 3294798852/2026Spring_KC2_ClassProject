from __future__ import annotations

from typing import Any


def merge_ranked_scale_outputs(
    scale_results: list[tuple[float, list[dict], list[Any], Any, tuple[int, int]]],
    top_k: int,
) -> tuple[list[dict], list[Any], list[tuple[float, Any, int, int]]]:
    merged_rows: list[dict] = []
    merged_images: list[Any] = []
    scale_heatmaps: list[tuple[float, Any, int, int]] = []
    seen = set()
    for sc, rows_sc, images_sc, hm_sc, fg_size_sc in scale_results:
        for r, img in zip(rows_sc, images_sc):
            key = (int(r["x"]), int(r["y"]), round(float(r["scale"]), 2))
            if key in seen:
                continue
            seen.add(key)
            merged_rows.append(r)
            merged_images.append(img)
        scale_heatmaps.append((float(sc), hm_sc, int(fg_size_sc[0]), int(fg_size_sc[1])))
    merged = sorted(
        list(zip(merged_rows, merged_images)),
        key=lambda t: float(t[0]["score"]),
        reverse=True,
    )[: max(1, int(top_k))]
    ranked = [x[0] for x in merged]
    images = [x[1] for x in merged]
    return ranked, images, scale_heatmaps

