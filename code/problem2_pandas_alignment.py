"""
Problem 2: Multi-File CSV Data Alignment & Validation (Pandas)

Write a function `compute_alignment_metrics(pred_csv: str, truth_csv: str) -> pd.DataFrame`
using Pandas that:
1. Loads both CSV files and performs an inner join on `request_id`.
2. Fills missing `predicted_intent` values with "UNKNOWN" and missing `ground_truth_intent`
   values with "UNLABELED".
3. Computes a boolean column `is_correct` where `predicted_intent == ground_truth_intent`.
4. Groups by `domain` (sorted alphabetically) and returns a DataFrame with columns:
   - `domain`
   - `sample_count` (int)
   - `accuracy` (float, fraction of correct predictions rounded to 4 decimal places)
   - `avg_confidence` (float, mean confidence rounded to 4 decimal places)
"""
import pandas as pd


def compute_alignment_metrics(pred_csv: str, truth_csv: str) -> pd.DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement compute_alignment_metrics")
