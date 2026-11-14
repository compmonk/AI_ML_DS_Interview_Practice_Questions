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
    # TODO: Implement your solution here
    raise NotImplementedError("Implement validate_agent_trajectory")
