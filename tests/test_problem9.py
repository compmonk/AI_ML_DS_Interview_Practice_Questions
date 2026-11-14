import pandas as pd
from code.problem9_pandas_stratified_eval import build_stratified_eval_set


def test_p9_filters_contaminated_train_ids():
    pool = pd.DataFrame([
        {"prompt_id": "p1", "domain": "media", "difficulty": "easy", "prompt_text": "Play pop"},
        {"prompt_id": "p2", "domain": "media", "difficulty": "easy", "prompt_text": "Play rock"},
    ])
    res = build_stratified_eval_set(pool, train_prompt_ids={"p1"}, max_per_stratum=5)
    assert list(res["prompt_id"]) == ["p2"]


def test_p9_fills_null_difficulty_with_medium_and_drops_blank_prompts():
    pool = pd.DataFrame([
        {"prompt_id": "p1", "domain": "media", "difficulty": None, "prompt_text": "Play jazz"},
        {"prompt_id": "p2", "domain": "media", "difficulty": "easy", "prompt_text": "   "},
        {"prompt_id": "p3", "domain": "media", "difficulty": "easy", "prompt_text": None},
    ])
    res = build_stratified_eval_set(pool, train_prompt_ids=set(), max_per_stratum=5)
    assert len(res) == 1
    assert res.iloc[0]["prompt_id"] == "p1"
    assert res.iloc[0]["difficulty"] == "medium"


def test_p9_stratified_sampling_caps_large_stratum_deterministically():
    pool = pd.DataFrame([
        {"prompt_id": f"p{i}", "domain": "weather", "difficulty": "hard", "prompt_text": f"Query {i}"}
        for i in range(10)
    ])
    res1 = build_stratified_eval_set(pool, train_prompt_ids=set(), max_per_stratum=3, seed=42)
    res2 = build_stratified_eval_set(pool, train_prompt_ids=set(), max_per_stratum=3, seed=42)
    assert len(res1) == 3
    assert list(res1["prompt_id"]) == list(res2["prompt_id"])
    assert list(res1["prompt_id"]) == sorted(res1["prompt_id"])


def test_p9_deduplicates_prompt_ids_before_sampling():
    pool = pd.DataFrame([
        {"prompt_id": "p1", "domain": "alarms", "difficulty": "easy", "prompt_text": "First version"},
        {"prompt_id": "p1", "domain": "alarms", "difficulty": "easy", "prompt_text": "Duplicate version"},
    ])
    res = build_stratified_eval_set(pool, train_prompt_ids=set(), max_per_stratum=5)
    assert len(res) == 1
    assert res.iloc[0]["prompt_text"] == "First version"


def test_p9_all_filtered_out_returns_empty_with_schema():
    pool = pd.DataFrame([
        {"prompt_id": "p1", "domain": "alarms", "difficulty": "easy", "prompt_text": "Set alarm"},
    ])
    res = build_stratified_eval_set(pool, train_prompt_ids={"p1"}, max_per_stratum=5)
    assert res.empty
    assert list(res.columns) == ["prompt_id", "domain", "difficulty", "prompt_text"]
