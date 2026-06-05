from typing import Dict, List


def analyze_candidate(
    x: int,
    y: int,
    fg_w: int,
    fg_h: int,
    bg_w: int,
    bg_h: int,
    score: float,
) -> List[str]:
    tips: List[str] = []
    margin = max(8, min(fg_w, fg_h) // 6)
    cx = x + fg_w / 2.0
    cy = y + fg_h / 2.0

    if x <= margin or y <= margin or (x + fg_w) >= (bg_w - margin) or (y + fg_h) >= (bg_h - margin):
        tips.append("目标贴近边缘，可能造成越界/裁切不自然。")

    if cy < bg_h * 0.35:
        tips.append("目标位置偏高，常见场景中容易缺少支撑关系。")
    elif cy > bg_h * 0.9:
        tips.append("目标位置过低，建议上移避免贴底。")

    horizontal_dist = abs(cx - bg_w / 2.0) / max(bg_w, 1)
    if horizontal_dist > 0.42:
        tips.append("目标偏离主体区域较远，可尝试向中心或语义主体附近移动。")

    if score < 0.4:
        tips.append("当前分数偏低，建议同时调整位置与缩放。")
    elif score < 0.66:
        tips.append("当前分数中等，可小步微调寻找更优位置。")

    if not tips:
        tips.append("当前位置较合理，可进一步尝试邻近位置做微调对比。")
    return tips


def spread_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {"min": 0.0, "max": 0.0, "gap": 0.0, "std": 0.0}
    scores = [float(r["score"]) for r in rows]
    s_min = min(scores)
    s_max = max(scores)
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {"min": s_min, "max": s_max, "gap": s_max - s_min, "std": var**0.5}
