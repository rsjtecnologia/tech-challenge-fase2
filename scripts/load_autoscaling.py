"""Bounded HTTP load against an explicitly selected local/cluster API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
import statistics
import time
from urllib.parse import urlsplit


def run_load(url: str, seconds: float, concurrency: int) -> dict:
    target = urlsplit(url)
    if (
        target.scheme != "http"
        or not target.hostname
        or target.username is not None
        or target.password is not None
        or target.fragment
    ):
        raise ValueError("Informe uma URL HTTP sem credenciais ou fragmento.")
    if not 0 < seconds <= 300 or not 1 <= concurrency <= 64:
        raise ValueError("seconds deve estar em (0, 300] e concurrency em [1, 64].")

    started = time.monotonic()
    deadline = started + seconds
    path = target.path or "/"
    if target.query:
        path += "?" + target.query

    def worker() -> tuple[list[float], int, dict[str, int]]:
        latencies: list[float] = []
        successes = 0
        errors: dict[str, int] = {}
        while time.monotonic() < deadline:
            connection = http.client.HTTPConnection(
                target.hostname, target.port or 80, timeout=10
            )
            request_started = time.monotonic()
            try:
                connection.request("GET", path, headers={"Connection": "close"})
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    successes += 1
                else:
                    reason = f"HTTP {response.status}"
                    errors[reason] = errors.get(reason, 0) + 1
            except (OSError, http.client.HTTPException) as error:
                reason = type(error).__name__
                errors[reason] = errors.get(reason, 0) + 1
            finally:
                latencies.append((time.monotonic() - request_started) * 1000)
                connection.close()
        return latencies, successes, errors

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: worker(), range(concurrency)))
    latencies = sorted(value for values, _, _ in results for value in values)
    errors: dict[str, int] = {}
    for _, _, worker_errors in results:
        for reason, count in worker_errors.items():
            errors[reason] = errors.get(reason, 0) + count
    successful_requests = sum(successes for _, successes, _ in results)
    elapsed = time.monotonic() - started
    return {
        "url": url,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 3),
        "requests": len(latencies),
        "successful_requests": successful_requests,
        "failed_requests": sum(errors.values()),
        "errors": errors,
        "requests_per_second": round(len(latencies) / elapsed, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p95_latency_ms": (
            round(latencies[math.ceil(len(latencies) * 0.95) - 1], 2)
            if latencies else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    result = run_load(args.url, args.seconds, args.concurrency)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result["successful_requests"] > 0 and not result["failed_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
