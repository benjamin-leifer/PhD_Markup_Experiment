# Metadata Retrieval and Processing Lineage

This document explores options for improving metadata performance and handling
complex processing parentage.

## Caching strategies

- **Function-level caches** – Decorators such as
  `functools.lru_cache` can memoize expensive lookups. This approach is simple
  to add and works well for metadata that changes infrequently.
- **Precomputed views** – Materialized tables or serialized JSON files can store
  merged metadata for common queries. They trade freshness for speed and require
  an invalidation strategy when source data changes. A background task can
  rebuild the materialized view whenever new tests are inserted.
- **External caches** – Tools like Redis or Memcached provide centralized caches
  with configurable expiration and cross-process sharing. They introduce extra
  infrastructure but scale beyond a single process. These caches are useful when
  multiple dashboard workers need to share a warm metadata cache.

## DAG representation for processing chains

The current parent chain is linear, but real workflows may fork or merge. A
Directed Acyclic Graph (DAG) captures these relationships explicitly. Python
libraries such as `networkx` model DAGs in-memory, while graph databases like
Neo4j or ArangoDB offer persistent storage and efficient traversal queries. A
DAG allows querying shared ancestors, detecting cycles, and attaching metadata to
edges for transformation steps. The prototype `prototypes/dag_metadata_prototype.py`
illustrates merging metadata from multiple parents and includes a tiny timing
comparison between a linear chain and a DAG traversal.

## Prototype benchmark

`prototypes/metadata_cache_benchmark.py` simulates an expensive metadata lookup
with a 1 ms delay and compares repeated calls with and without caching. Running
the script showed a dramatic improvement:

```
Uncached: 0.5862s for 500 calls
Cached:   0.0012s for 500 calls
```

While artificial, the example demonstrates how caching can eliminate redundant
work when metadata is accessed repeatedly. The DAG prototype yields slower
lookups for this tiny graph:

```
Linear chain: 0.0012s for 1000 calls
DAG traversal: 0.0151s for 1000 calls
```

The overhead comes from computing ancestor sets, but the approach unlocks more
flexible parentage when needed.
