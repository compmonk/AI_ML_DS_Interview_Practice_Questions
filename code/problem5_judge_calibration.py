"""
Problem 5: LLM-as-a-Judge Calibration & Position Bias Audit

Write a function `audit_judge_calibration(eval_df: pd.DataFrame) -> dict` that evaluates how well
an automated LLM judge aligns with human ground-truth labels and detects position bias.

Input DataFrame `eval_df` columns:
    - `sample_id`: str
    - `human_winner`: str ("A" or "B")
    - `judge_winner_forward`: str ("A" or "B")  # Order: (Response A, Response B)
    - `judge_winner_swapped`: str ("A" or "B")  # Order: (Response B, Response A), labeled by original ID

Requirements:
1. If `eval_df` is empty, return:
   {"total_samples": 0, "forward_agreement": 0.0, "consistent_agreement": 0.0,
    "position_flip_rate": 0.0, "cohens_kappa": 0.0}
2. `forward_agreement`: fraction of rows where `judge_winner_forward == human_winner` (rounded to 4 decimals).
3. `position_flip_rate`: fraction of rows where `judge_winner_forward != judge_winner_swapped` (rounded to 4 decimals).
4. `consistent_agreement`: fraction of rows where BOTH `judge_winner_forward == human_winner` AND
   `judge_winner_swapped == human_winner` (rounded to 4 decimals).
5. `cohens_kappa`: Cohen's Kappa between `human_winner` and `judge_winner_forward` across categories {"A", "B"}:
      kappa = (p_o - p_e) / (1 - p_e)
   where `p_o` is observed agreement and `p_e` is expected chance agreement:
      p_e = P(human="A")*P(judge="A") + P(human="B")*P(judge="B")
   If `1 - p_e == 0`, return 1.0 if `p_o == 1.0` else 0.0. Round to 4 decimals.
"""
import pandas as pd


def audit_judge_calibration(eval_df: pd.DataFrame) -> dict:
    result = {
        "total_samples": 0,
        "forward_agreement": 0.0,
        "consistent_agreement": 0.0,
        "position_flip_rate": 0.0,
        "cohens_kappa": 0.0
    }

    # 1
    if len(eval_df) == 0:
        return result

    # 2
    forward_agreement = round((eval_df["human_winner"] == eval_df["judge_winner_forward"]).sum() / len(eval_df), 4)

    # 3
    position_flip_rate = round((eval_df["judge_winner_forward"] != eval_df["judge_winner_swapped"]).sum() / len(eval_df), 4)

    # 4
    consistent_agreement = round(((eval_df["judge_winner_forward"] == eval_df["human_winner"]) &
                                  (eval_df["judge_winner_swapped"] == eval_df["human_winner"])).sum() / len(eval_df), 4)


    # 5
    p_o = forward_agreement

    # p_e = P(human="A") * P(judge="A") + P(human="B") * P(judge="B")
    p_e = ((eval_df["human_winner"] == "A").mean() * (eval_df["judge_winner_forward"] == "A").mean() +
           (eval_df["human_winner"] == "B").mean() * (eval_df["judge_winner_forward"] == "B").mean())

    if p_e == 1:
        cohens_kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        cohens_kappa = round((p_o - p_e) / (1 - p_e), 4)


    result["forward_agreement"] = forward_agreement
    result["position_flip_rate"] = position_flip_rate
    result["consistent_agreement"] = consistent_agreement
    result["cohens_kappa"] = cohens_kappa
    result["total_samples"] = len(eval_df)

    return result
