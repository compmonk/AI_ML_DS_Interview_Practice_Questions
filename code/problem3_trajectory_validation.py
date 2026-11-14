"""
Problem 3: Sliding-Window Tool-Call Trajectory Validation

Write a function `validate_agent_trajectory(trace: list[dict]) -> bool` that validates
an ordered sequence of agent tool calls. Each item is a dict:
    {"step": int, "tool": str, "status": str}

Validation Rules:
1. Empty trajectories are invalid (return False).
2. Step indices must be strictly sequential integers starting at 0 (0, 1, 2, ...).
3. The sequence must start (step 0) with `tool == "intent_classifier"`.
4. Any step with `status == "FAILED"` must be immediately followed by a step whose
   `tool` is either `"error_recovery"` or `"fallback_handler"`. A `"FAILED"` status on
   the final step is invalid.
5. The final step in the trajectory must have `status` in `{"SUCCESS", "TERMINATED"}`.
"""


def validate_agent_trajectory(trace: list[dict]) -> bool:
    # 1
    if len(trace) == 0:
        return False

    for i in range(len(trace)):
        # 2
        if trace[i]["step"] != i:
            return False

        # 3
        if i == 0 and trace[i]["tool"] != "intent_classifier":
            return False

        # 4
        if trace[i]["status"] == "FAILED":
            if i == len(trace) - 1:
                return False

            if i + 1 < len(trace) and trace[i + 1]["tool"] not in ("error_recovery", "fallback_handler"):
                return False

        # 5
        if i == len(trace) - 1 and trace[i]["status"] not in {"SUCCESS", "TERMINATED"}:
            return False

    return True

