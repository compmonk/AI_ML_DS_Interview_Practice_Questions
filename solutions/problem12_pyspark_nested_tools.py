from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def extract_tool_metrics(trajectories_df: DataFrame) -> DataFrame:
    valid = trajectories_df.filter(F.col("steps").isNotNull() & (F.size(F.col("steps")) > 0))
    exploded = valid.select(F.explode(F.col("steps")).alias("step"))
    return (
        exploded.select(
            F.col("step.tool_name").alias("tool_name"),
            F.col("step.duration_ms").alias("duration_ms"),
            F.col("step.success").alias("success"),
        )
        .groupBy("tool_name")
        .agg(
            F.count("*").alias("invocation_count"),
            F.round(F.avg(F.col("success").cast("double")), 4).alias("success_rate"),
            F.round(F.sum("duration_ms"), 2).alias("total_duration_ms"),
            F.round(F.max("duration_ms"), 2).alias("max_duration_ms"),
        )
        .orderBy(F.col("invocation_count").desc(), F.col("tool_name").asc())
    )
