import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
from code.problem11_pyspark_rolling import compute_rolling_health


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test_problem11")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


SCHEMA = StructType([
    StructField("model_version", StringType(), True),
    StructField("request_seq", IntegerType(), True),
    StructField("latency_ms", DoubleType(), True),
    StructField("is_error", BooleanType(), True),
])


def test_p11_rolling_3_window_and_alert_trigger(spark):
    rows = [
        ("v1", 1, 100.0, True),   # window [1]: err=1 -> False
        ("v1", 2, 200.0, False),  # window [1,2]: err=1 -> False
        ("v1", 3, 300.0, True),   # window [1,2,3]: err=2 -> True, avg=200.0
        ("v1", 4, 100.0, False),  # window [2,3,4]: err=1 -> False, avg=200.0
    ]
    res = compute_rolling_health(spark.createDataFrame(rows, SCHEMA)).collect()
    assert [r["rolling_3_error_count"] for r in res] == [1, 1, 2, 1]
    assert [r["alert_triggered"] for r in res] == [False, False, True, False]
    assert res[2]["rolling_3_avg_latency"] == 200.0


def test_p11_partitions_isolated_across_models(spark):
    rows = [
        ("v1", 1, 50.0, True),
        ("v2", 1, 150.0, True),
    ]
    res = compute_rolling_health(spark.createDataFrame(rows, SCHEMA)).collect()
    assert all(r["rolling_3_error_count"] == 1 for r in res)
    assert all(r["alert_triggered"] is False for r in res)


def test_p11_unsorted_input_sequences(spark):
    rows = [
        ("v1", 3, 30.0, True),
        ("v1", 1, 10.0, False),
        ("v1", 2, 20.0, True),
    ]
    res = compute_rolling_health(spark.createDataFrame(rows, SCHEMA)).collect()
    assert [r["request_seq"] for r in res] == [1, 2, 3]
    assert res[-1]["rolling_3_avg_latency"] == 20.0
    assert res[-1]["alert_triggered"] is True


def test_p11_all_three_errors_in_window(spark):
    rows = [
        ("v1", 1, 10.0, True),
        ("v1", 2, 20.0, True),
        ("v1", 3, 30.0, True),
    ]
    res = compute_rolling_health(spark.createDataFrame(rows, SCHEMA)).collect()
    assert res[-1]["rolling_3_error_count"] == 3
    assert res[-1]["alert_triggered"] is True


def test_p11_empty_dataframe(spark):
    res = compute_rolling_health(spark.createDataFrame([], SCHEMA)).collect()
    assert res == []
