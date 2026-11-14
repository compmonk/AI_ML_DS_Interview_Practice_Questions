"""
Problem 12 (PySpark): Agent Tool-Call Array Explosion & Co-Occurrence Extraction

Siri multi-step trajectories are stored in Parquet as nested arrays of tool invocations.

Given a PySpark DataFrame `trajectories_df` with schema:
    - `trace_id`: string
    - `steps`: array<struct<tool_name: string, duration_ms: double, success: boolean>>

Write `extract_tool_metrics(trajectories_df: DataFrame) -> DataFrame`:
1. Filter out traces where `steps` is null or empty (`size(steps) == 0`).
2. Explode `steps` so each row represents a single tool invocation.
3. Group by `tool_name` and compute:
   - `invocation_count`: total number of times the tool was invoked (long/int)
   - `success_rate`: `round(avg(cast(success as double)), 4)`
   - `total_duration_ms`: `round(sum(duration_ms), 2)`
   - `max_duration_ms`: `round(max(duration_ms), 2)`
4. Return the DataFrame ordered by `invocation_count` descending, then `tool_name` ascending.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def extract_tool_metrics(trajectories_df: DataFrame) -> DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement extract_tool_metrics")
