"""
Problem 7 (Pandas): Siri Multi-Turn Sessionization & Failure Recovery Metrics

Given a DataFrame `events_df` of user interactions with columns:
    - `user_id`: str
    - `timestamp`: str or datetime64 (ISO-8601 timestamps)
    - `intent`: str
    - `tool_success`: bool

Task:
Write `analyze_user_sessions(events_df: pd.DataFrame, timeout_minutes: int = 15) -> pd.DataFrame`:
1. If `events_df` is empty, return an empty DataFrame with columns:
   ["user_id", "total_sessions", "avg_turns_per_session", "session_success_rate"]
2. Convert `timestamp` to datetime (`pd.to_datetime`) and sort by `["user_id", "timestamp"]` ascending.
3. Group events into sessions per `user_id`: a new session starts on a user's first event OR
   whenever the time elapsed since that user's previous event is strictly greater than
   `timeout_minutes` (`> pd.Timedelta(minutes=timeout_minutes)`).
4. A session is considered successful (`session_succeeded = True`) if the **final turn**
   in that session has `tool_success == True`.
5. Aggregate per `user_id` (sorted ascending by `user_id`) and return a DataFrame with:
   - `user_id` (str)
   - `total_sessions` (int)
   - `avg_turns_per_session` (float, rounded to 2 decimals)
   - `session_success_rate` (float, fraction of succeeded sessions rounded to 4 decimals)
"""
import pandas as pd


def analyze_user_sessions(events_df: pd.DataFrame, timeout_minutes: int = 15) -> pd.DataFrame:
    # TODO: Implement your solution here
    raise NotImplementedError("Implement analyze_user_sessions")
