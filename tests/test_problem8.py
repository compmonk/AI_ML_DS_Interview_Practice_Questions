import pandas as pd
from code.problem8_pandas_regression import evaluate_checkpoint_regression


def test_p8_empty_dataframe():
    df = pd.DataFrame(columns=["prompt_id", "domain", "baseline_score", "candidate_score", "candidate_latency_ms"])
    res = evaluate_checkpoint_regression(df)
    assert res.empty
    assert list(res.columns) == [
        "domain", "sample_count", "baseline_mean", "candidate_mean",
        "score_delta", "win_rate", "p95_latency_ms", "release_status"
    ]


def test_p8_pass_and_block_by_score_regression():
    df = pd.DataFrame([
        {"prompt_id": "1", "domain": "media", "baseline_score": 0.80, "candidate_score": 0.85, "candidate_latency_ms": 120.0},
        {"prompt_id": "2", "domain": "media", "baseline_score": 0.80, "candidate_score": 0.80, "candidate_latency_ms": 140.0},
        {"prompt_id": "3", "domain": "navigation", "baseline_score": 0.90, "candidate_score": 0.85, "candidate_latency_ms": 110.0},
    ])
    res = evaluate_checkpoint_regression(df, max_regression=0.02, p95_latency_cap_ms=250.0)
    media = res[res["domain"] == "media"].iloc[0]
    nav = res[res["domain"] == "navigation"].iloc[0]
    assert media["release_status"] == "PASS"
    assert media["win_rate"] == 0.5
    assert media["score_delta"] == 0.025
    assert nav["release_status"] == "BLOCK"
    assert nav["score_delta"] == -0.05


def test_p8_block_by_latency_cap_despite_score_gain():
    df = pd.DataFrame([
        {"prompt_id": "1", "domain": "search", "baseline_score": 0.70, "candidate_score": 0.95, "candidate_latency_ms": 300.0},
    ])
    res = evaluate_checkpoint_regression(df, max_regression=0.02, p95_latency_cap_ms=250.0)
    assert res.iloc[0]["release_status"] == "BLOCK"
    assert res.iloc[0]["p95_latency_ms"] == 300.0


def test_p8_deduplicates_prompt_id_keeping_last():
    df = pd.DataFrame([
        {"prompt_id": "p1", "domain": "calendar", "baseline_score": 0.90, "candidate_score": 0.10, "candidate_latency_ms": 900.0},
        {"prompt_id": "p1", "domain": "calendar", "baseline_score": 0.90, "candidate_score": 0.92, "candidate_latency_ms": 100.0},
    ])
    res = evaluate_checkpoint_regression(df)
    assert res.iloc[0]["sample_count"] == 1
    assert res.iloc[0]["candidate_mean"] == 0.92
    assert res.iloc[0]["release_status"] == "PASS"


def test_p8_exact_regression_boundary_passes():
    df = pd.DataFrame([
        {"prompt_id": "p1", "domain": "timers", "baseline_score": 0.80, "candidate_score": 0.78, "candidate_latency_ms": 250.0},
    ])
    res = evaluate_checkpoint_regression(df, max_regression=0.02, p95_latency_cap_ms=250.0)
    assert res.iloc[0]["score_delta"] == -0.02
    assert res.iloc[0]["release_status"] == "PASS"
