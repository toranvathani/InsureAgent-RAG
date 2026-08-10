"""
Lightweight observability: tracks request counts, latency, and errors in memory
and exposes them via /metrics. Swap for Prometheus client if you want to go further.
"""
import time
from collections import deque
from statistics import mean

_request_log = deque(maxlen=500)  # rolling window of recent requests
_error_count = 0
_request_count = 0


def record_request(endpoint: str, latency_ms: float, success: bool):
    global _error_count, _request_count
    _request_count += 1
    if not success:
        _error_count += 1
    _request_log.append({"endpoint": endpoint, "latency_ms": latency_ms, "success": success, "ts": time.time()})


def get_metrics() -> dict:
    latencies = [r["latency_ms"] for r in _request_log]
    return {
        "total_requests": _request_count,
        "total_errors": _error_count,
        "error_rate": round(_error_count / _request_count, 4) if _request_count else 0.0,
        "avg_latency_ms": round(mean(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1) if latencies else 0.0,
        "sample_size": len(latencies),
    }
