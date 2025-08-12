"""Simple benchmark for metadata caching.

This script compares repeated metadata lookups with and without
``functools.lru_cache`` to illustrate the potential speed-up from caching
frequently accessed metadata. It uses a tiny artificial delay to simulate an
expensive retrieval such as a database query.
"""

from functools import lru_cache
import time
import timeit


def slow_metadata_lookup(code: str) -> dict:
    """Simulate a slow metadata fetch."""
    time.sleep(0.001)
    return {"code": code}


@lru_cache(maxsize=None)
def cached_metadata_lookup(code: str) -> dict:
    """Same lookup but cached."""
    time.sleep(0.001)
    return {"code": code}


def benchmark(repeats: int = 500) -> None:
    """Run a simple timing comparison."""
    uncached = timeit.timeit(
        "slow_metadata_lookup('A')", number=repeats, globals=globals()
    )
    cached = timeit.timeit(
        "cached_metadata_lookup('A')", number=repeats, globals=globals()
    )
    print(f"Uncached: {uncached:.4f}s for {repeats} calls")
    print(f"Cached:   {cached:.4f}s for {repeats} calls")


if __name__ == "__main__":  # pragma: no cover - benchmark script
    benchmark()
