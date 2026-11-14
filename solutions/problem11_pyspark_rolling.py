from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def compute_rolling_health(logs_df: DataFrame) -> DataFrame:
    w = Window.partitionBy("model_version").orderBy(F.col("request_seq").asc()).rowsBetween(-2, 0)
    return (
        logs_df.withColumn("rolling_3_avg_latency", F.round(F.avg("latency_ms").over(w), 2))
        .withColumn("rolling_3_error_count", F.sum(F.col("is_error").cast("int")).over(w))
        .withColumn("alert_triggered", F.col("rolling_3_error_count") >= 2)
        .select(
            "model_version",
            "request_seq",
            "rolling_3_avg_latency",
            "rolling_3_error_count",
            "alert_triggered",
        )
        .orderBy(F.col("model_version").asc(), F.col("request_seq").asc())
    )
