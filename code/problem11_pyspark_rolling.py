"""
Problem 11 (PySpark): Rolling 3-Request Failure Streak & Rolling Latency per Model

Given a PySpark DataFrame `logs_df` with schema:
    - `model_version`: string
    - `request_seq`: int (monotonically increasing sequence number per model)
    - `latency_ms`: double
    - `is_error`: boolean

Write `compute_rolling_health(logs_df: DataFrame) -> DataFrame`:
1. Partition by `model_version` and order by `request_seq` ascending.
2. Using a window of the current row and the 2 preceding rows (`rowsBetween(-2, 0)`):
   - `rolling_3_avg_latency`: average of `latency_ms` over the window, rounded to 2 decimals.
   - `rolling_3_error_count`: sum of `cast(is_error as int)` over the window (int/long).
   - `alert_triggered`: boolean, `True` if `rolling_3_error_count >= 2`, else `False`.
3. Return a DataFrame with columns:
   `["model_version", "request_seq", "rolling_3_avg_latency", "rolling_3_error_count", "alert_triggered"]`
   ordered by `["model_version", "request_seq"]` ascending.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def compute_rolling_health(logs_df: DataFrame) -> DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement compute_rolling_health")
