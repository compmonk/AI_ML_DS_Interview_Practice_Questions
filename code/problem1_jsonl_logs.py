"""
Problem 1: Streaming JSONL & Metric Aggregation

Write a function `evaluate_model_logs(file_path: str, min_confidence: float) -> dict`
that reads a JSONL file line-by-line (streaming, without loading the whole file into memory).

Sample Data:
{"event_id": "e_101", "model_version": "siri-v2.1", "confidence": 0.89, "latency_ms": 142, "tokens": {"prompt": 45, "completion": 12}}

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
from collections import OrderedDict


def evaluate_model_logs(file_path: str, min_confidence: float) -> dict:
    result = dict()
    with open(file_path, "r") as file:
        for line in file:

            # Skip empty lines or whitspaces
            if line is None or len(line.strip()) == 0:
                continue
            else:
                jd = json.loads(line.strip())

                # Filter out records based on min_confidence threshold
                if jd["confidence"] >= min_confidence:

                    # Initialize an empty dict for new model_version if not seen before
                    model_version = jd["model_version"]
                    if model_version not in result:
                        result[model_version] = {
                            "event_count": 0,
                            "avg_latency_ms": 0,
                            "total_tokens": 0,
                        }

                    result[model_version]["avg_latency_ms"] += jd["latency_ms"]
                    result[model_version]["event_count"] += 1
                    result[model_version]["total_tokens"] = result[model_version]["total_tokens"] + jd["tokens"][
                        "prompt"] + jd["tokens"]["completion"]

    for model_version, v in result.items():
        v["avg_latency_ms"] = round(v["avg_latency_ms"] / v["event_count"], 2)

    return OrderedDict(sorted(result.items()))
