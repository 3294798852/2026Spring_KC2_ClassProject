def classify_score(score: float) -> str:
    if score >= 0.66:
        return "推荐"
    if score >= 0.4:
        return "可接受"
    return "不推荐"
