import pandas as pd
from code.problem7_pandas_sessions import analyze_user_sessions


def test_p7_empty_dataframe():
    df = pd.DataFrame(columns=["user_id", "timestamp", "intent", "tool_success"])
    res = analyze_user_sessions(df)
    assert res.empty
    assert list(res.columns) == ["user_id", "total_sessions", "avg_turns_per_session", "session_success_rate"]


def test_p7_single_user_multiple_sessions_and_recovery():
    df = pd.DataFrame([
        # Session 1: fails turn 1, recovers on turn 2 -> success! (2 turns)
        {"user_id": "u1", "timestamp": "2026-11-01T10:00:00", "intent": "weather", "tool_success": False},
        {"user_id": "u1", "timestamp": "2026-11-01T10:05:00", "intent": "weather", "tool_success": True},
        # Session 2: 20 mins later (>15m timeout), fails final turn -> failed! (1 turn)
        {"user_id": "u1", "timestamp": "2026-11-01T10:25:00", "intent": "alarm", "tool_success": False},
    ])
    res = analyze_user_sessions(df, timeout_minutes=15)
    assert len(res) == 1
    row = res.iloc[0]
    assert row["user_id"] == "u1"
    assert row["total_sessions"] == 2
    assert row["avg_turns_per_session"] == 1.5
    assert row["session_success_rate"] == 0.5


def test_p7_exact_timeout_boundary_stays_in_same_session():
    df = pd.DataFrame([
        {"user_id": "u1", "timestamp": "2026-11-01T12:00:00", "intent": "a", "tool_success": False},
        {"user_id": "u1", "timestamp": "2026-11-01T12:15:00", "intent": "b", "tool_success": True},
    ])
    res = analyze_user_sessions(df, timeout_minutes=15)
    assert res.iloc[0]["total_sessions"] == 1
    assert res.iloc[0]["avg_turns_per_session"] == 2.0
    assert res.iloc[0]["session_success_rate"] == 1.0


def test_p7_unsorted_input_timestamps():
    df = pd.DataFrame([
        {"user_id": "u2", "timestamp": "2026-11-01T09:02:00", "intent": "b", "tool_success": False},
        {"user_id": "u2", "timestamp": "2026-11-01T09:00:00", "intent": "a", "tool_success": True},
    ])
    res = analyze_user_sessions(df, timeout_minutes=15)
    # Chronologically: 09:00 (True) -> 09:02 (False), so session ends False!
    assert res.iloc[0]["total_sessions"] == 1
    assert res.iloc[0]["session_success_rate"] == 0.0


def test_p7_multiple_users_sorted_output():
    df = pd.DataFrame([
        {"user_id": "user_z", "timestamp": "2026-11-01T08:00:00", "intent": "x", "tool_success": True},
        {"user_id": "user_a", "timestamp": "2026-11-01T08:00:00", "intent": "y", "tool_success": False},
        {"user_id": "user_a", "timestamp": "2026-11-01T09:00:00", "intent": "y", "tool_success": True},
        {"user_id": "user_a", "timestamp": "2026-11-01T10:00:00", "intent": "y", "tool_success": True},
    ])
    res = analyze_user_sessions(df, timeout_minutes=15)
    assert list(res["user_id"]) == ["user_a", "user_z"]
    assert res.iloc[0]["session_success_rate"] == 0.6667
    assert res.iloc[1]["session_success_rate"] == 1.0
