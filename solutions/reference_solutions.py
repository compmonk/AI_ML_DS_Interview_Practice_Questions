import json
import re
import numpy as np
import pandas as pd


# --- Problem 1 ---
def evaluate_model_logs(file_path: str, min_confidence: float) -> dict:
    metrics = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("confidence", 0.0) < min_confidence:
                continue

            mv = record["model_version"]
            if mv not in metrics:
                metrics[mv] = {"count": 0, "total_latency": 0, "total_tokens": 0}

            metrics[mv]["count"] += 1
            metrics[mv]["total_latency"] += record["latency_ms"]
            tokens = record.get("tokens", {})
            metrics[mv]["total_tokens"] += tokens.get("prompt", 0) + tokens.get("completion", 0)

    result = {}
    for mv in sorted(metrics.keys()):
        d = metrics[mv]
        result[mv] = {
            "event_count": d["count"],
            "avg_latency_ms": round(d["total_latency"] / d["count"], 2),
            "total_tokens": d["total_tokens"],
        }
    return result


# --- Problem 2 ---
def compute_alignment_metrics(pred_csv: str, truth_csv: str) -> pd.DataFrame:
    preds = pd.read_csv(pred_csv)
    truth = pd.read_csv(truth_csv)

    merged = pd.merge(preds, truth, on="request_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["domain", "sample_count", "accuracy", "avg_confidence"])

    merged["predicted_intent"] = merged["predicted_intent"].fillna("UNKNOWN")
    merged["ground_truth_intent"] = merged["ground_truth_intent"].fillna("UNLABELED")
    merged["is_correct"] = merged["predicted_intent"] == merged["ground_truth_intent"]

    summary = (
        merged.groupby("domain", as_index=False)
        .agg(
            sample_count=("request_id", "count"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .sort_values("domain")
        .reset_index(drop=True)
    )
    summary["accuracy"] = summary["accuracy"].round(4)
    summary["avg_confidence"] = summary["avg_confidence"].round(4)
    return summary


# --- Problem 3 ---
def validate_agent_trajectory(trace: list[dict]) -> bool:
    if not trace:
        return False

    if trace[0].get("tool") != "intent_classifier":
        return False

    if trace[-1].get("status") not in {"SUCCESS", "TERMINATED"}:
        return False

    recovery_tools = {"error_recovery", "fallback_handler"}

    for idx, item in enumerate(trace):
        if item.get("step") != idx:
            return False

        if item.get("status") == "FAILED":
            if idx + 1 >= len(trace):
                return False
            if trace[idx + 1].get("tool") not in recovery_tools:
                return False

    return True


# --- Problem 4 ---
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\(\d{3}\)\s*|\b\d{3}-)\d{3}-\d{4}\b")


def curate_eval_prompts(records: list[dict]) -> list[dict]:
    best_by_norm = {}

    for rec in records:
        raw_prompt = rec.get("prompt")
        if raw_prompt is None or not str(raw_prompt).strip():
            continue

        redacted = EMAIL_RE.sub("[EMAIL]", str(raw_prompt))
        redacted = PHONE_RE.sub("[PHONE]", redacted)

        norm_key = " ".join(redacted.lower().split())
        cleaned_rec = {
            "id": rec["id"],
            "prompt": redacted,
            "quality_score": rec["quality_score"],
            "source": rec["source"],
        }

        if norm_key not in best_by_norm:
            best_by_norm[norm_key] = cleaned_rec
        else:
            existing = best_by_norm[norm_key]
            new_rank = (cleaned_rec["quality_score"], 1 if cleaned_rec["source"] == "human_curated" else 0)
            old_rank = (existing["quality_score"], 1 if existing["source"] == "human_curated" else 0)
            if new_rank > old_rank:
                best_by_norm[norm_key] = cleaned_rec

    return sorted(best_by_norm.values(), key=lambda x: x["id"])


# --- Problem 5 ---
def audit_judge_calibration(eval_df: pd.DataFrame) -> dict:
    n = len(eval_df)
    if n == 0:
        return {
            "total_samples": 0,
            "forward_agreement": 0.0,
            "consistent_agreement": 0.0,
            "position_flip_rate": 0.0,
            "cohens_kappa": 0.0,
        }

    fwd_match = eval_df["judge_winner_forward"] == eval_df["human_winner"]
    swap_match = eval_df["judge_winner_swapped"] == eval_df["human_winner"]
    flipped = eval_df["judge_winner_forward"] != eval_df["judge_winner_swapped"]

    p_o = float(fwd_match.mean())
    consistent = float((fwd_match & swap_match).mean())
    flip_rate = float(flipped.mean())

    p_h_a = float((eval_df["human_winner"] == "A").mean())
    p_h_b = float((eval_df["human_winner"] == "B").mean())
    p_j_a = float((eval_df["judge_winner_forward"] == "A").mean())
    p_j_b = float((eval_df["judge_winner_forward"] == "B").mean())

    p_e = (p_h_a * p_j_a) + (p_h_b * p_j_b)
    if 1.0 - p_e == 0.0:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "total_samples": int(n),
        "forward_agreement": round(p_o, 4),
        "consistent_agreement": round(consistent, 4),
        "position_flip_rate": round(flip_rate, 4),
        "cohens_kappa": round(float(kappa), 4),
    }


# --- Problem 6 ---
def analyze_endpoint_slas(events: list[dict], sla_thresholds_ms: dict[str, float]) -> dict:
    grouped = {}
    for ev in events:
        if ev.get("status") != "OK":
            continue
        lat = ev.get("latency_ms")
        if lat is None or lat < 0:
            continue
        ep = ev["endpoint"]
        grouped.setdefault(ep, []).append(float(lat))

    result = {}
    for ep in sorted(grouped.keys()):
        arr = np.array(grouped[ep], dtype=float)
        thresh = sla_thresholds_ms.get(ep, float("inf"))
        breaches = np.sum(arr > thresh)
        result[ep] = {
            "count": int(len(arr)),
            "p50_ms": round(float(np.percentile(arr, 50, method="linear")), 2),
            "p95_ms": round(float(np.percentile(arr, 95, method="linear")), 2),
            "p99_ms": round(float(np.percentile(arr, 99, method="linear")), 2),
            "sla_breach_rate": round(float(breaches / len(arr)), 4),
        }
    return result
