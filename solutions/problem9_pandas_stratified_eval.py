import pandas as pd


def build_stratified_eval_set(
    pool_df: pd.DataFrame,
    train_prompt_ids: set[str],
    max_per_stratum: int,
    seed: int = 42,
) -> pd.DataFrame:
    cols = ["prompt_id", "domain", "difficulty", "prompt_text"]
    if pool_df.empty or max_per_stratum <= 0:
        return pd.DataFrame(columns=cols)

    df = pool_df[cols].copy()
    df = df[~df["prompt_id"].isin(train_prompt_ids)]
    df = df.drop_duplicates(subset=["prompt_id"], keep="first")
    df = df[df["prompt_text"].notna() & (df["prompt_text"].astype(str).str.strip() != "")]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["difficulty"] = df["difficulty"].fillna("medium")

    sampled_parts = []
    for (_, _), grp in df.groupby(["domain", "difficulty"], sort=True):
        grp_sorted = grp.sort_values("prompt_id", ascending=True)
        if len(grp_sorted) <= max_per_stratum:
            sampled_parts.append(grp_sorted)
        else:
            sampled_parts.append(grp_sorted.sample(n=max_per_stratum, replace=False, random_state=seed))

    result = (
        pd.concat(sampled_parts, ignore_index=True)
        .sort_values(["domain", "difficulty", "prompt_id"], ascending=True)
        .reset_index(drop=True)
    )
    return result[cols]
