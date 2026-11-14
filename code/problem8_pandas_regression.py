"""
Problem 8 (Pandas): Model Checkpoint A/B Evaluation & Regression Gate

Given `eval_df` with columns:
    - `prompt_id`: str
    - `domain`: str
    - `baseline_score`: float (0.0 to 1.0)
    - `candidate_score`: float (0.0 to 1.0)
    - `candidate_latency_ms`: float

Write `evaluate_checkpoint_regression(eval_df: pd.DataFrame, max_regression: float = 0.02, p95_latency_cap_ms: float = 250.0) -> pd.DataFrame`:
1. If `eval_df` is empty, return an empty DataFrame with columns:
   ["domain", "sample_count", "baseline_mean", "candidate_mean", "score_delta",
    "win_rate", "p95_latency_ms", "release_status"]
2. Deduplicate `eval_df` by `prompt_id`, keeping the **last** occurrence (`keep="last"`).
3. Compute `delta = candidate_score - baseline_score` per prompt. A candidate "wins" a prompt
   if `candidate_score > baseline_score` (strictly greater).
4. Group by `domain` (sorted ascending) and compute:
   - `sample_count` (int)
   - `baseline_mean` (float, rounded to 4 decimals)
   - `candidate_mean` (float, rounded to 4 decimals)
   - `score_delta` (float, `candidate_mean - baseline_mean` computed from unrounded means, rounded to 4 decimals)
   - `win_rate` (float, fraction of prompts where `candidate_score > baseline_score`, rounded to 4 decimals)
   - `p95_latency_ms` (float, 95th percentile of `candidate_latency_ms` using linear interpolation, rounded to 2 decimals)
   - `release_status` (str): `"PASS"` if `(unrounded_score_delta >= -max_regression)` AND
     `(unrounded_p95_latency <= p95_latency_cap_ms)`, otherwise `"BLOCK"`.
"""
import numpy as np
import pandas as pd


def evaluate_checkpoint_regression(
    eval_df: pd.DataFrame,
    max_regression: float = 0.02,
    p95_latency_cap_ms: float = 250.0,
) -> pd.DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement evaluate_checkpoint_regression")
