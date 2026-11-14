"""
Problem 6: Agent Endpoint Latency Percentiles & SLA Breach Detection

Write a function `analyze_endpoint_slas(events: list[dict], sla_thresholds_ms: dict[str, float]) -> dict`
that computes latency percentiles and SLA breach rates per tool/endpoint.

Each event in `events` has:
    {"endpoint": str, "latency_ms": float, "status": str}

Requirements:
1. Only include events where `status == "OK"` and `latency_ms >= 0`.
2. Group valid latencies by `endpoint`.
3. For each endpoint (sorted alphabetically), compute:
   - `count`: number of valid events
   - `p50_ms`: 50th percentile latency (using linear interpolation, i.e., `numpy.percentile(..., method="linear")`), rounded to 2 decimals
   - `p95_ms`: 95th percentile latency, rounded to 2 decimals
   - `p99_ms`: 99th percentile latency, rounded to 2 decimals
   - `sla_breach_rate`: fraction of valid events strictly exceeding `sla_thresholds_ms[endpoint]`, rounded to 4 decimals.
     (If `endpoint` is not present in `sla_thresholds_ms`, default threshold is `float("inf")`, so breach rate is `0.0`).
4. Return a dict mapping `endpoint -> metrics_dict`.
"""
from collections import defaultdict

import numpy as np


def analyze_endpoint_slas(events: list[dict], sla_thresholds_ms: dict[str, float]) -> dict:
    endpoint_latencies = defaultdict(list)
    
    for event in events:
        # 1
        if event["status"] != "OK" or event["latency_ms"] is None or event["latency_ms"] < 0:
            continue
        # 2
        else:
            endpoint_latencies[event["endpoint"]].append(event["latency_ms"])

    # 3
    result = {}
    for endpoint, ls in sorted(endpoint_latencies.items()):
        result[endpoint] = {
            "count": len(ls),
            "p50_ms": round(np.percentile(ls, 50, method="linear"), 2),
            "p95_ms": round(np.percentile(ls, 95, method="linear"), 2),
            "p99_ms": round(np.percentile(ls, 99, method="linear"), 2),
            "sla_breach_rate": round(sum([1 if l > sla_thresholds_ms.get(endpoint, float("inf")) else 0 for l in ls]) / len(ls), 4)
        }

    # 4
    return result

    
