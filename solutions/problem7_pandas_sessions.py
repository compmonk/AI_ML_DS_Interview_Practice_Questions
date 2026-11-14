import pandas as pd


def analyze_user_sessions(events_df: pd.DataFrame, timeout_minutes: int = 15) -> pd.DataFrame:
    cols = ["user_id", "total_sessions", "avg_turns_per_session", "session_success_rate"]
    if events_df.empty:
        return pd.DataFrame(columns=cols)

    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    time_diff = df.groupby("user_id")["timestamp"].diff()
    new_session = time_diff.isna() | (time_diff > pd.Timedelta(minutes=timeout_minutes))
    df["session_id"] = new_session.groupby(df["user_id"]).cumsum()

    sessions = (
        df.groupby(["user_id", "session_id"], as_index=False)
        .agg(
            turns=("intent", "count"),
            session_succeeded=("tool_success", "last"),
        )
    )

    summary = (
        sessions.groupby("user_id", as_index=False)
        .agg(
            total_sessions=("session_id", "count"),
            avg_turns_per_session=("turns", "mean"),
            session_success_rate=("session_succeeded", "mean"),
        )
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    summary["avg_turns_per_session"] = summary["avg_turns_per_session"].round(2)
    summary["session_success_rate"] = summary["session_success_rate"].round(4)
    return summary[cols]
