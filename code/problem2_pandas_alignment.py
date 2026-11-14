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

    pred_df = pd.read_csv(pred_csv)
    truth_df = pd.read_csv(truth_csv)

    # 1
    result_df = pd.merge(pred_df, truth_df, left_on="request_id", right_on="request_id", how="inner")

    # 2
    result_df["predicted_intent"] = result_df["predicted_intent"].fillna("UNKNOWN")
    result_df["ground_truth_intent"] = result_df["ground_truth_intent"].fillna("UNLABELED")

    # 3
    result_df["is_correct"] = result_df["predicted_intent"] == result_df["ground_truth_intent"]

    # 4
    result_df = result_df.groupby("domain").agg({"request_id": "count", "is_correct": "mean", "confidence": "mean"}).sort_index()
    result_df = result_df.rename(columns={"is_correct": "accuracy", "confidence": "avg_confidence", "request_id": "sample_count"})
    result_df["accuracy"] = result_df["accuracy"].round(4)
    result_df["avg_confidence"] = result_df["avg_confidence"].round(4)
    return result_df.reset_index()


