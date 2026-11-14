import json
import pandas as pd
import pytest

from code.problem1_jsonl_logs import evaluate_model_logs
from code.problem2_pandas_alignment import compute_alignment_metrics
from code.problem3_trajectory_validation import validate_agent_trajectory
from code.problem4_dedup_curation import curate_eval_prompts
from code.problem5_judge_calibration import audit_judge_calibration
from code.problem6_latency_percentiles import analyze_endpoint_slas


# ==============================================================================
# PROBLEM 1 TESTS: Streaming JSONL & Metric Aggregation (5 test cases)
# ==============================================================================
def test_problem1_sample_logs():
    res = evaluate_model_logs("data/eval_logs.jsonl", min_confidence=0.75)
    assert list(res.keys()) == ["siri-v2.0", "siri-v2.1"]
    assert res["siri-v2.0"] == {"event_count": 2, "avg_latency_ms": 150.0, "total_tokens": 119}
    assert res["siri-v2.1"] == {"event_count": 2, "avg_latency_ms": 130.0, "total_tokens": 97}


def test_problem1_empty_and_whitespace_lines(tmp_path):
    f = tmp_path / "blank.jsonl"
    f.write_text("\n   \n\t\n")
    assert evaluate_model_logs(str(f), 0.0) == {}


def test_problem1_all_filtered_out():
    assert evaluate_model_logs("data/eval_logs.jsonl", min_confidence=0.99) == {}


def test_problem1_exact_threshold_boundary(tmp_path):
    f = tmp_path / "boundary.jsonl"
    rows = [
        {"event_id": "1", "model_version": "m1", "confidence": 0.80, "latency_ms": 100, "tokens": {"prompt": 10, "completion": 5}},
        {"event_id": "2", "model_version": "m1", "confidence": 0.7999, "latency_ms": 200, "tokens": {"prompt": 20, "completion": 5}},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    res = evaluate_model_logs(str(f), min_confidence=0.80)
    assert res == {"m1": {"event_count": 1, "avg_latency_ms": 100.0, "total_tokens": 15}}


def test_problem1_alphabetical_sorting_and_rounding(tmp_path):
    f = tmp_path / "sorting.jsonl"
    rows = [
        {"event_id": "1", "model_version": "z-model", "confidence": 0.9, "latency_ms": 10, "tokens": {"prompt": 1, "completion": 1}},
        {"event_id": "2", "model_version": "a-model", "confidence": 0.9, "latency_ms": 10, "tokens": {"prompt": 2, "completion": 3}},
        {"event_id": "3", "model_version": "a-model", "confidence": 0.9, "latency_ms": 11, "tokens": {"prompt": 4, "completion": 1}},
        {"event_id": "4", "model_version": "a-model", "confidence": 0.9, "latency_ms": 11, "tokens": {"prompt": 1, "completion": 1}},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    res = evaluate_model_logs(str(f), min_confidence=0.5)
    assert list(res.keys()) == ["a-model", "z-model"]
    assert res["a-model"] == {"event_count": 3, "avg_latency_ms": 10.67, "total_tokens": 12}


# ==============================================================================
# PROBLEM 2 TESTS: Multi-File CSV Data Alignment & Validation (5 test cases)
# ==============================================================================
def test_problem2_sample_alignment():
    df = compute_alignment_metrics("data/predictions.csv", "data/ground_truth.csv")
    assert list(df.columns) == ["domain", "sample_count", "accuracy", "avg_confidence"]
    assert list(df["domain"]) == ["home_control", "media", "navigation"]
    media_row = df[df["domain"] == "media"].iloc[0]
    assert media_row["sample_count"] == 2
    assert media_row["accuracy"] == 0.5
    assert media_row["avg_confidence"] == 0.73


def test_problem2_missing_intents_imputed_differently(tmp_path):
    p_file = tmp_path / "p.csv"
    t_file = tmp_path / "t.csv"
    pd.DataFrame([{"request_id": "1", "domain": "weather", "predicted_intent": None, "confidence": 0.5}]).to_csv(p_file, index=False)
    pd.DataFrame([{"request_id": "1", "ground_truth_intent": None}]).to_csv(t_file, index=False)
    df = compute_alignment_metrics(str(p_file), str(t_file))
    # UNKNOWN != UNLABELED -> 0.0 accuracy
    assert df.iloc[0]["accuracy"] == 0.0


def test_problem2_no_overlapping_request_ids(tmp_path):
    p_file = tmp_path / "p.csv"
    t_file = tmp_path / "t.csv"
    pd.DataFrame([{"request_id": "1", "domain": "weather", "predicted_intent": "get_weather", "confidence": 0.9}]).to_csv(p_file, index=False)
    pd.DataFrame([{"request_id": "2", "ground_truth_intent": "get_weather"}]).to_csv(t_file, index=False)
    df = compute_alignment_metrics(str(p_file), str(t_file))
    assert df.empty
    assert list(df.columns) == ["domain", "sample_count", "accuracy", "avg_confidence"]


def test_problem2_perfect_and_zero_accuracy_domains(tmp_path):
    p_file = tmp_path / "p.csv"
    t_file = tmp_path / "t.csv"
    pd.DataFrame([
        {"request_id": "1", "domain": "alarms", "predicted_intent": "set_alarm", "confidence": 0.99},
        {"request_id": "2", "domain": "alarms", "predicted_intent": "cancel_alarm", "confidence": 0.91},
        {"request_id": "3", "domain": "timers", "predicted_intent": "set_timer", "confidence": 0.40},
    ]).to_csv(p_file, index=False)
    pd.DataFrame([
        {"request_id": "1", "ground_truth_intent": "set_alarm"},
        {"request_id": "2", "ground_truth_intent": "cancel_alarm"},
        {"request_id": "3", "ground_truth_intent": "pause_timer"},
    ]).to_csv(t_file, index=False)
    df = compute_alignment_metrics(str(p_file), str(t_file))
    assert df.loc[df["domain"] == "alarms", "accuracy"].iloc[0] == 1.0
    assert df.loc[df["domain"] == "timers", "accuracy"].iloc[0] == 0.0


def test_problem2_four_decimal_rounding(tmp_path):
    p_file = tmp_path / "p.csv"
    t_file = tmp_path / "t.csv"
    pd.DataFrame([
        {"request_id": "1", "domain": "math", "predicted_intent": "calc", "confidence": 0.33333},
        {"request_id": "2", "domain": "math", "predicted_intent": "calc", "confidence": 0.33333},
        {"request_id": "3", "domain": "math", "predicted_intent": "other", "confidence": 0.33334},
    ]).to_csv(p_file, index=False)
    pd.DataFrame([
        {"request_id": "1", "ground_truth_intent": "calc"},
        {"request_id": "2", "ground_truth_intent": "calc"},
        {"request_id": "3", "ground_truth_intent": "calc"},
    ]).to_csv(t_file, index=False)
    df = compute_alignment_metrics(str(p_file), str(t_file))
    assert df.iloc[0]["accuracy"] == 0.6667
    assert df.iloc[0]["avg_confidence"] == 0.3333


# ==============================================================================
# PROBLEM 3 TESTS: Sliding-Window Tool-Call Trajectory Validation (6 test cases)
# ==============================================================================
def test_problem3_valid_with_error_recovery():
    trace = [
        {"step": 0, "tool": "intent_classifier", "status": "SUCCESS"},
        {"step": 1, "tool": "database_lookup", "status": "FAILED"},
        {"step": 2, "tool": "error_recovery", "status": "SUCCESS"},
    ]
    assert validate_agent_trajectory(trace) is True


def test_problem3_valid_with_fallback_and_terminated():
    trace = [
        {"step": 0, "tool": "intent_classifier", "status": "SUCCESS"},
        {"step": 1, "tool": "calendar_api", "status": "FAILED"},
        {"step": 2, "tool": "fallback_handler", "status": "TERMINATED"},
    ]
    assert validate_agent_trajectory(trace) is True


def test_problem3_unhandled_failure():
    trace = [
        {"step": 0, "tool": "intent_classifier", "status": "SUCCESS"},
        {"step": 1, "tool": "database_lookup", "status": "FAILED"},
        {"step": 2, "tool": "response_generator", "status": "SUCCESS"},
    ]
    assert validate_agent_trajectory(trace) is False


def test_problem3_failure_on_final_step():
    trace = [
        {"step": 0, "tool": "intent_classifier", "status": "SUCCESS"},
        {"step": 1, "tool": "error_recovery", "status": "FAILED"},
    ]
    assert validate_agent_trajectory(trace) is False


def test_problem3_empty_and_wrong_start():
    assert validate_agent_trajectory([]) is False
    assert validate_agent_trajectory([{"step": 0, "tool": "calendar_tool", "status": "SUCCESS"}]) is False


def test_problem3_non_sequential_steps_and_bad_terminal():
    assert validate_agent_trajectory([
        {"step": 0, "tool": "intent_classifier", "status": "SUCCESS"},
        {"step": 2, "tool": "calendar_tool", "status": "SUCCESS"},
    ]) is False
    assert validate_agent_trajectory([
        {"step": 0, "tool": "intent_classifier", "status": "PENDING"},
    ]) is False


# ==============================================================================
# PROBLEM 4 TESTS: Prompt Deduplication & PII Sanitization (5 test cases)
# ==============================================================================
def test_problem4_pii_redaction_email_and_phone():
    records = [
        {"id": "p1", "prompt": "Email john.doe@apple.com or call (408) 555-1234", "quality_score": 0.9, "source": "synthetic"},
        {"id": "p2", "prompt": "Text 617-555-9876 about meeting", "quality_score": 0.85, "source": "synthetic"},
    ]
    res = curate_eval_prompts(records)
    assert res[0]["prompt"] == "Email [EMAIL] or call [PHONE]"
    assert res[1]["prompt"] == "Text [PHONE] about meeting"


def test_problem4_dedup_highest_quality_wins():
    records = [
        {"id": "p2", "prompt": "  What is the   weather today? ", "quality_score": 0.72, "source": "synthetic"},
        {"id": "p1", "prompt": "what is the weather today?", "quality_score": 0.95, "source": "synthetic"},
    ]
    res = curate_eval_prompts(records)
    assert len(res) == 1
    assert res[0]["id"] == "p1"
    assert res[0]["quality_score"] == 0.95


def test_problem4_dedup_tie_breaker_human_curated():
    records = [
        {"id": "p1", "prompt": "Set an alarm for 7 AM", "quality_score": 0.88, "source": "synthetic"},
        {"id": "p2", "prompt": "set an alarm for 7 am", "quality_score": 0.88, "source": "human_curated"},
    ]
    res = curate_eval_prompts(records)
    assert len(res) == 1
    assert res[0]["id"] == "p2"
    assert res[0]["source"] == "human_curated"


def test_problem4_dedup_full_tie_keeps_first_and_sorts_by_id():
    records = [
        {"id": "p9", "prompt": "Play jazz music", "quality_score": 0.80, "source": "synthetic"},
        {"id": "p1", "prompt": "play jazz music", "quality_score": 0.80, "source": "synthetic"},
        {"id": "p0", "prompt": "Navigate home", "quality_score": 0.90, "source": "human_curated"},
    ]
    res = curate_eval_prompts(records)
    assert [r["id"] for r in res] == ["p0", "p9"]


def test_problem4_skips_empty_and_none_prompts():
    records = [
        {"id": "p1", "prompt": None, "quality_score": 1.0, "source": "human_curated"},
        {"id": "p2", "prompt": "   \n\t  ", "quality_score": 1.0, "source": "human_curated"},
    ]
    assert curate_eval_prompts(records) == []


# ==============================================================================
# PROBLEM 5 TESTS: LLM-as-a-Judge Calibration & Position Bias (5 test cases)
# ==============================================================================
def test_problem5_empty_dataframe():
    df = pd.DataFrame(columns=["sample_id", "human_winner", "judge_winner_forward", "judge_winner_swapped"])
    assert audit_judge_calibration(df) == {
        "total_samples": 0,
        "forward_agreement": 0.0,
        "consistent_agreement": 0.0,
        "position_flip_rate": 0.0,
        "cohens_kappa": 0.0,
    }


def test_problem5_perfect_consistent_judge():
    df = pd.DataFrame([
        {"sample_id": "1", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "A"},
        {"sample_id": "2", "human_winner": "B", "judge_winner_forward": "B", "judge_winner_swapped": "B"},
        {"sample_id": "3", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "A"},
        {"sample_id": "4", "human_winner": "B", "judge_winner_forward": "B", "judge_winner_swapped": "B"},
    ])
    res = audit_judge_calibration(df)
    assert res == {
        "total_samples": 4,
        "forward_agreement": 1.0,
        "consistent_agreement": 1.0,
        "position_flip_rate": 0.0,
        "cohens_kappa": 1.0,
    }


def test_problem5_severe_position_bias():
    # Judge always picks whichever response is shown first (forward="A", swapped="B")
    df = pd.DataFrame([
        {"sample_id": "1", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "B"},
        {"sample_id": "2", "human_winner": "B", "judge_winner_forward": "A", "judge_winner_swapped": "B"},
        {"sample_id": "3", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "B"},
        {"sample_id": "4", "human_winner": "B", "judge_winner_forward": "A", "judge_winner_swapped": "B"},
    ])
    res = audit_judge_calibration(df)
    assert res["forward_agreement"] == 0.5
    assert res["consistent_agreement"] == 0.0
    assert res["position_flip_rate"] == 1.0
    assert res["cohens_kappa"] == 0.0


def test_problem5_mixed_calibration_metrics():
    df = pd.DataFrame([
        {"sample_id": "1", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "A"},
        {"sample_id": "2", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "B"},
        {"sample_id": "3", "human_winner": "B", "judge_winner_forward": "B", "judge_winner_swapped": "B"},
        {"sample_id": "4", "human_winner": "B", "judge_winner_forward": "A", "judge_winner_swapped": "A"},
    ])
    res = audit_judge_calibration(df)
    # p_o = 3/4 = 0.75; p_h_A = 0.5, p_h_B = 0.5; p_j_A = 0.75, p_j_B = 0.25
    # p_e = 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa = (0.75 - 0.5) / 0.5 = 0.5
    assert res["total_samples"] == 4
    assert res["forward_agreement"] == 0.75
    assert res["consistent_agreement"] == 0.5
    assert res["position_flip_rate"] == 0.25
    assert res["cohens_kappa"] == 0.5


def test_problem5_single_class_edge_case():
    df = pd.DataFrame([
        {"sample_id": "1", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "A"},
        {"sample_id": "2", "human_winner": "A", "judge_winner_forward": "A", "judge_winner_swapped": "A"},
    ])
    res = audit_judge_calibration(df)
    assert res["cohens_kappa"] == 1.0


# ==============================================================================
# PROBLEM 6 TESTS: Agent Endpoint Latency Percentiles & SLA Detection (5 test cases)
# ==============================================================================
def test_problem6_percentiles_and_sla_breaches():
    events = [
        {"endpoint": "weather_api", "latency_ms": 10.0, "status": "OK"},
        {"endpoint": "weather_api", "latency_ms": 20.0, "status": "OK"},
        {"endpoint": "weather_api", "latency_ms": 30.0, "status": "OK"},
        {"endpoint": "weather_api", "latency_ms": 40.0, "status": "OK"},
        {"endpoint": "weather_api", "latency_ms": 100.0, "status": "OK"},
    ]
    res = analyze_endpoint_slas(events, {"weather_api": 35.0})
    assert res["weather_api"]["count"] == 5
    assert res["weather_api"]["p50_ms"] == 30.0
    assert res["weather_api"]["p95_ms"] == 88.0
    assert res["weather_api"]["p99_ms"] == 97.6
    assert res["weather_api"]["sla_breach_rate"] == 0.4


def test_problem6_filters_errors_and_negative_latencies():
    events = [
        {"endpoint": "maps_api", "latency_ms": 50.0, "status": "OK"},
        {"endpoint": "maps_api", "latency_ms": 999.0, "status": "ERROR"},
        {"endpoint": "maps_api", "latency_ms": -5.0, "status": "OK"},
        {"endpoint": "maps_api", "latency_ms": None, "status": "OK"},
    ]
    res = analyze_endpoint_slas(events, {"maps_api": 40.0})
    assert res["maps_api"] == {
        "count": 1,
        "p50_ms": 50.0,
        "p95_ms": 50.0,
        "p99_ms": 50.0,
        "sla_breach_rate": 1.0,
    }


def test_problem6_missing_sla_threshold_defaults_to_zero_breaches():
    events = [
        {"endpoint": "notes_api", "latency_ms": 250.0, "status": "OK"},
        {"endpoint": "notes_api", "latency_ms": 500.0, "status": "OK"},
    ]
    res = analyze_endpoint_slas(events, {})
    assert res["notes_api"]["sla_breach_rate"] == 0.0


def test_problem6_strictly_greater_sla_boundary():
    events = [
        {"endpoint": "timer_api", "latency_ms": 50.0, "status": "OK"},
        {"endpoint": "timer_api", "latency_ms": 50.01, "status": "OK"},
    ]
    res = analyze_endpoint_slas(events, {"timer_api": 50.0})
    assert res["timer_api"]["sla_breach_rate"] == 0.5


def test_problem6_empty_or_no_valid_events():
    assert analyze_endpoint_slas([], {"timer_api": 50.0}) == {}
    assert analyze_endpoint_slas([{"endpoint": "x", "latency_ms": 10.0, "status": "TIMEOUT"}], {}) == {}
