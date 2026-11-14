"""
Problem 4: Prompt Deduplication & PII Sanitization (Dataset Curation)

Write a function `curate_eval_prompts(records: list[dict]) -> list[dict]` that cleans and
deduplicates evaluation prompt records prior to model benchmarking.

Each input record has keys:
    {"id": str, "prompt": str, "quality_score": float, "source": str}

Requirements:
1. Skip records where `prompt` is None or empty after stripping whitespace.
2. Redact sensitive PII in `prompt`:
   - Replace email addresses (e.g., `alice@example.com`) with `[EMAIL]`.
   - Replace US phone numbers in formats `123-456-7890` or `(123) 456-7890` with `[PHONE]`.
3. Normalize prompts for deduplication by lowercasing, collapsing multiple internal whitespace
   characters into a single space, and stripping leading/trailing whitespace.
4. Deduplicate records that share the same normalized prompt:
   - Keep the record with the higher `quality_score`.
   - If `quality_score` is tied, prefer `"human_curated"` over any other `source`.
   - If still tied, keep the record that appeared first in the input list.
5. Return the deduplicated list of dicts (preserving the redacted `prompt` and original fields),
   sorted by `id` ascending.
"""
import re


def curate_eval_prompts(records: list[dict]) -> list[dict]:
    filtered_records = {}

    def redact_pii(prompt: str) -> str:
        rp = prompt
        rp = re.sub(r"[1-9][0-9][0-9]-\d{3}-\d{4}", "[PHONE]", rp)
        rp = re.sub(r"\([1-9][0-9][0-9]\) \d{3}-\d{4}", "[PHONE]", rp)
        rp = re.sub(r"\S+@\S+\.\S{2,}", "[EMAIL]", rp)
        return rp

    for record in records:
        # 1
        if record["prompt"] is None or len(record["prompt"].strip()) == 0:
            continue

        # 2
        redacted_prompt = redact_pii(record["prompt"])
        record["prompt"] = redacted_prompt

        # 3
        redacted_prompt = " ".join(redacted_prompt.lower().split())

        # 4
        if redacted_prompt not in filtered_records:
            filtered_records[redacted_prompt] = record
        else:
            previous = filtered_records[redacted_prompt]

            new_score = (record["quality_score"], record["source"] == "human_curated")
            old_score = (previous["quality_score"], previous["source"] == "human_curated")
            if new_score > old_score:
                filtered_records[redacted_prompt] = record


            # if record["quality_score"] > previous["quality_score"]:
            #     filtered_records[redacted_prompt] = record
            # elif record["quality_score"] == previous["quality_score"]:
            #     if previous["source"] == "human_curated":
            #         continue
            #     elif record["source"] == "human_curated":
            #         filtered_records[redacted_prompt] = record

    return sorted(filtered_records.values(), key=lambda x: x["id"])







