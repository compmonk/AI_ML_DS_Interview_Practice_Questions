from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def summarize_deduped_evals(df: DataFrame) -> DataFrame:
    w = Window.partitionBy("event_id").orderBy(
        F.col("event_ts").desc(),
        F.col("passed").desc(),
        F.col("latency_ms").asc(),
    )
    deduped = (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    return (
        deduped.groupBy("domain")
        .agg(
            F.count("event_id").alias("total_events"),
            F.round(F.avg(F.col("passed").cast("double")), 4).alias("pass_rate"),
            F.round(F.avg("latency_ms"), 2).alias("avg_latency_ms"),
        )
        .orderBy(F.col("domain").asc())
    )
