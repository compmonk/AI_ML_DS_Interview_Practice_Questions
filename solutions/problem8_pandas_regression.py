import numpy as np
import pandas as pd


def evaluate_checkpoint_regression(
    eval_df: pd.DataFrame,
    max_regression: float = 0.02,
    p95_latency_cap_ms: float = 250.0,
) -> pd.DataFrame:
    cols = [
        "domain",
        "sample_count",
        "baseline_mean",
        "candidate_mean",
        "score_delta",
        "win_rate",
        "p95_latency_ms",
        "release_status",
    ]
    if eval_df.empty:
        return pd.DataFrame(columns=cols)

    df = eval_df.drop_duplicates(subset=["prompt_id"], keep="last").copy()
    df["is_win"] = df["candidate_score"] > df["baseline_score"]

    rows = []
    for domain, grp in df.groupby("domain", sort=True):
        b_mean = float(grp["baseline_score"].mean())
        c_mean = float(grp["candidate_score"].mean())
        delta = c_mean - b_mean
        win_rate = float(grp["is_win"].mean())
        p95 = float(np.percentile(grp["candidate_latency_ms"].to_numpy(dtype=float), 95, method="linear"))
        status = "PASS" if (delta >= -max_regression - 1e-12 and p95 <= p95_latency_cap_ms + 1e-12) else "BLOCK"
        rows.append(
            {
                "domain": domain,
                "sample_count": int(len(grp)),
                "baseline_mean": round(b_mean, 4),
                "candidate_mean": round(c_mean, 4),
                "score_delta": round(delta, 4),
                "win_rate": round(win_rate, 4),
                "p95_latency_ms": round(p95, 2),
                "release_status": status,
            }
        )

    return pd.DataFrame(rows, columns=cols)
