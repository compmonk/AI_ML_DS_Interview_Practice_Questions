"""
Problem 1: Streaming JSONL & Metric Aggregation

Write a function `evaluate_model_logs(file_path: str, min_confidence: float) -> dict`
that reads a JSONL file line-by-line (streaming, without loading the whole file into memory).

Requirements:
1. Skip empty/whitespace lines.
2. Filter out records where `confidence < min_confidence`.
3. Group remaining records by `model_version`.
4. For each `model_version`, compute:
   - `event_count`: total valid records
   - `avg_latency_ms`: average latency rounded to 2 decimal places
   - `total_tokens`: sum of prompt + completion tokens across valid records
5. Return a dictionary sorted alphabetically by `model_version`.
"""
import json


def evaluate_model_logs(file_path: str, min_confidence: float) -> dict:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement evaluate_model_logs")
