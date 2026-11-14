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
    # TODO: Implement your solution here
    raise NotImplementedError("Implement curate_eval_prompts")
