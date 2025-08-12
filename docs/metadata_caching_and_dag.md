# Metadata Retrieval and Processing Lineage

This document explores options for improving metadata performance and handling
complex processing parentage.

## Caching strategies

- **Function-level caches** – Decorators such as
  `functools.lru_cache` can memoize expensive lookups. This approach is simple
  to add and works well for metadata that changes infrequently.
- **Precomputed views** – Materialized tables or serialized JSON files can store
  merged metadata for common queries. They trade freshness for speed and require
  an invalidation strategy when source data changes.
- **External caches** – Tools like Redis or Memcached provide centralized caches
  with configurable expiration and cross-process sharing. They introduce extra
  infrastructure but scale beyond a single process.

## DAG representation for processing chains

The current parent chain is linear, but real workflows may fork or merge. A
Directed Acyclic Graph (DAG) captures these relationships explicitly. Python
libraries such as `networkx` model DAGs in-memory, while graph databases like
Neo4j or ArangoDB offer persistent storage and efficient traversal queries. A
DAG allows querying shared ancestors, detecting cycles, and attaching metadata to
edges for transformation steps.

## Prototype benchmark

`prototypes/metadata_cache_benchmark.py` simulates an expensive metadata lookup
with a 1 ms delay and compares repeated calls with and without caching. Running
the script showed a dramatic improvement:

```
Uncached: 0.5393s for 500 calls
Cached:   0.0012s for 500 calls
```

While artificial, the example demonstrates how caching can eliminate redundant
work when metadata is accessed repeatedly.
