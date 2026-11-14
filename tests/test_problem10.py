import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, BooleanType
from code.problem10_pyspark_dedup import summarize_deduped_evals


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test_problem10")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


SCHEMA = StructType([
    StructField("event_id", StringType(), True),
    StructField("domain", StringType(), True),
    StructField("event_ts", LongType(), True),
    StructField("latency_ms", DoubleType(), True),
    StructField("passed", BooleanType(), True),
])


def test_p10_dedup_keeps_latest_timestamp(spark):
    rows = [
        ("e1", "media", 1000, 200.0, False),
        ("e1", "media", 2000, 100.0, True),
        ("e2", "media", 1500, 150.0, False),
    ]
    res = summarize_deduped_evals(spark.createDataFrame(rows, SCHEMA)).collect()
    assert len(res) == 1
    assert res[0]["domain"] == "media"
    assert res[0]["total_events"] == 2
    assert res[0]["pass_rate"] == 0.5
    assert res[0]["avg_latency_ms"] == 125.0


def test_p10_tie_breaker_prefers_passed_true(spark):
    rows = [
        ("e1", "weather", 1000, 50.0, False),
        ("e1", "weather", 1000, 80.0, True),
    ]
    res = summarize_deduped_evals(spark.createDataFrame(rows, SCHEMA)).collect()
    assert res[0]["pass_rate"] == 1.0
    assert res[0]["avg_latency_ms"] == 80.0


def test_p10_tie_breaker_prefers_lowest_latency(spark):
    rows = [
        ("e1", "alarms", 1000, 120.0, True),
        ("e1", "alarms", 1000, 90.0, True),
    ]
    res = summarize_deduped_evals(spark.createDataFrame(rows, SCHEMA)).collect()
    assert res[0]["avg_latency_ms"] == 90.0


def test_p10_multiple_domains_ordered_alphabetically(spark):
    rows = [
        ("e1", "navigation", 1000, 100.0, True),
        ("e2", "calendar", 1000, 200.0, False),
    ]
    res = summarize_deduped_evals(spark.createDataFrame(rows, SCHEMA)).collect()
    assert [r["domain"] for r in res] == ["calendar", "navigation"]


def test_p10_empty_dataframe(spark):
    res = summarize_deduped_evals(spark.createDataFrame([], SCHEMA)).collect()
    assert res == []
