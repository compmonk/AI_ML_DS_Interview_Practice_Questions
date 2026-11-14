# Apple ML Data & Eval Engineer — Coding Practice Suite (Expanded)

This workspace contains 6 production-style Python coding problems tailored to the **ML Data & Eval Engineer (Siri Agent Modeling)** onsite interview, complete with expanded test suites (5+ test cases per problem).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Problems Included
1. **Problem 1 (`problem1_jsonl_logs.py`)**: Streaming JSONL audit log parsing, filtering, and metric aggregation.
2. **Problem 2 (`problem2_pandas_alignment.py`)**: Multi-file CSV data alignment, missing value imputation, and domain-level accuracy reporting using Pandas.
3. **Problem 3 (`problem3_trajectory_validation.py`)**: Sliding-window agentic tool-call sequence validation with strict state transition rules.
4. **Problem 4 (`problem4_dedup_curation.py`)**: Exact & normalized prompt deduplication with priority tie-breaking and PII redaction for dataset curation.
5. **Problem 5 (`problem5_judge_calibration.py`)**: LLM-as-a-Judge vs. Human annotation calibration metrics (accuracy, precision/recall, Cohen's Kappa, and position-bias detection).
6. **Problem 6 (`problem6_latency_percentiles.py`)**: Streaming/sliding-window latency SLA & percentile (p50, p95, p99) tracker per model endpoint.

## Running Tests
Run the full pytest suite from the root folder:
```bash
PYTHONPATH=. pytest -v
```

Run tests for a single problem (e.g., Problem 4):
```bash
PYTHONPATH=. pytest -k "problem4" -v
```
