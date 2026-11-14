"""
Problem 9 (Pandas): Stratified Eval Dataset Sampling & Leakage Filtering

Before running an expensive LLM-as-a-Judge benchmark, we need to remove contaminated
training prompts and construct a stratified, balanced sample across domains and difficulty tiers.

Given:
    - `pool_df`: DataFrame with columns `["prompt_id", "domain", "difficulty", "prompt_text"]`
    - `train_prompt_ids`: set[str] of prompt IDs seen during training (contamination set)
    - `max_per_stratum`: int (maximum rows to sample per `(domain, difficulty)` pair)
    - `seed`: int = 42

Write `build_stratified_eval_set(pool_df: pd.DataFrame, train_prompt_ids: set[str], max_per_stratum: int, seed: int = 42) -> pd.DataFrame`:
1. Filter out rows where `prompt_id` is in `train_prompt_ids`.
2. Drop duplicate `prompt_id` rows (keeping first) and rows with null/blank `prompt_text`.
3. Fill missing `difficulty` values with `"medium"`.
4. For each `(domain, difficulty)` stratum:
   - Sort rows within the stratum by `prompt_id` ascending first (for deterministic ordering).
   - If the stratum has `<= max_per_stratum` rows, keep all rows.
   - Otherwise, sample `max_per_stratum` rows without replacement using `random_state=seed`.
5. Return the resulting DataFrame sorted by `["domain", "difficulty", "prompt_id"]` ascending,
   with reset integer index (`drop=True`), preserving columns `["prompt_id", "domain", "difficulty", "prompt_text"]`.
"""
import pandas as pd


def build_stratified_eval_set(
    pool_df: pd.DataFrame,
    train_prompt_ids: set[str],
    max_per_stratum: int,
    seed: int = 42,
) -> pd.DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement build_stratified_eval_set")
