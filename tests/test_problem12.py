import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType, BooleanType
from code.problem12_pyspark_nested_tools import extract_tool_metrics


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test_problem12")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


STEP_STRUCT = StructType([
    StructField("tool_name", StringType(), True),
    StructField("duration_ms", DoubleType(), True),
    StructField("success", BooleanType(), True),
])
SCHEMA = StructType([
    StructField("trace_id", StringType(), True),
    StructField("steps", ArrayType(STEP_STRUCT), True),
])


def test_p12_explodes_and_aggregates_tool_metrics(spark):
    rows = [
        ("t1", [("intent", 10.0, True), ("weather", 120.5, True)]),
        ("t2", [("intent", 15.0, True), ("weather", 200.0, False)]),
    ]
    res = extract_tool_metrics(spark.createDataFrame(rows, SCHEMA)).collect()
    assert len(res) == 2
    # Tied invocation_count=2 -> alphabetical ("intent", then "weather")
    assert res[0]["tool_name"] == "intent"
    assert res[0]["invocation_count"] == 2
    assert res[0]["success_rate"] == 1.0
    assert res[0]["total_duration_ms"] == 25.0
    assert res[0]["max_duration_ms"] == 15.0

    assert res[1]["tool_name"] == "weather"
    assert res[1]["success_rate"] == 0.5
    assert res[1]["total_duration_ms"] == 320.5
    assert res[1]["max_duration_ms"] == 200.0


def test_p12_orders_by_invocation_count_descending(spark):
    rows = [
        ("t1", [("rare_tool", 50.0, True), ("common_tool", 10.0, True)]),
        ("t2", [("common_tool", 20.0, True)]),
    ]
    res = extract_tool_metrics(spark.createDataFrame(rows, SCHEMA)).collect()
    assert [r["tool_name"] for r in res] == ["common_tool", "rare_tool"]


def test_p12_ignores_null_and_empty_step_arrays(spark):
    rows = [
        ("t1", None),
        ("t2", []),
        ("t3", [("maps", 45.0, True)]),
    ]
    res = extract_tool_metrics(spark.createDataFrame(rows, SCHEMA)).collect()
    assert len(res) == 1
    assert res[0]["tool_name"] == "maps"


def test_p12_rounding_to_four_and_two_decimals(spark):
    rows = [
        ("t1", [("calc", 10.123, True), ("calc", 10.124, True), ("calc", 10.125, False)]),
    ]
    res = extract_tool_metrics(spark.createDataFrame(rows, SCHEMA)).collect()
    assert res[0]["success_rate"] == 0.6667
    assert res[0]["total_duration_ms"] == 30.37


def test_p12_empty_dataframe(spark):
    res = extract_tool_metrics(spark.createDataFrame([], SCHEMA)).collect()
    assert res == []
