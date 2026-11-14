"""
Problem 10 (PySpark): Distributed Evaluation Event Deduplication & Domain Summary

In distributed Spark ingestion pipelines, duplicate evaluation events arrive out-of-order.

Given a PySpark DataFrame `df` with schema:
    - `event_id`: string
    - `domain`: string
    - `event_ts`: long (epoch milliseconds)
    - `latency_ms`: double
    - `passed`: boolean

Write `summarize_deduped_evals(df: DataFrame) -> DataFrame`:
1. Deduplicate `df` by `event_id`, keeping the record with the **highest `event_ts`**.
   (If two records have identical `event_id` and `event_ts`, break ties by preferring `passed == True`,
   then lowest `latency_ms`).
2. Group the deduplicated records by `domain` and compute:
   - `total_events`: count of deduplicated events (long/int)
   - `pass_rate`: `round(avg(cast(passed as double)), 4)`
   - `avg_latency_ms`: `round(avg(latency_ms), 2)`
3. Return the aggregated DataFrame ordered by `domain` ascending.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def summarize_deduped_evals(df: DataFrame) -> DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement summarize_deduped_evals")
