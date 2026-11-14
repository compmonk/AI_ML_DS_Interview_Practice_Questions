# Apple ML Data & Eval Engineer — Practice Suite (Part 2: Pandas & PySpark)

6 advanced Pandas & PySpark interview problems focused on large-scale ML data curation, sessionization, A/B checkpoint evaluation, and distributed feature aggregation.

## Folder Structure
```text
├── code/           # Starter files (implement your solutions here)
├── tests/          # Separate pytest test files for each problem (5+ tests each)
├── solutions/      # Reference implementations
├── pytest.ini      # Automatically adds project root to PYTHONPATH
└── requirements.txt
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests
Because `pytest.ini` sets `pythonpath = .`, you can run `pytest` directly:
```bash
# Run all tests
pytest -v

# Run tests for a single problem
pytest tests/test_problem7.py -v
```
