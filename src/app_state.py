RESULT_SESSION_KEYS = [
    "last_result",
    "last_heatmap",
    "last_heatmap_overlay",
    "manual_score_result",
    "manual_drag_x",
    "manual_drag_y",
    "manual_drag_json",
    "manual_drag_need_init",
    "manual_applied_rgba",
    "manual_applied_info",
    "manual_applied_postprocessed",
]


def clear_result_state(session_state) -> None:
    for key in RESULT_SESSION_KEYS:
        session_state.pop(key, None)

