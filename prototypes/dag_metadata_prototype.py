"""Prototype representing processing lineage as a DAG.

This script builds a small directed acyclic graph using ``networkx`` where each
node stores its own metadata. The ``merged_metadata`` function collects metadata
from all ancestors, allowing multiple parents to contribute to a child's
metadata. A tiny benchmark compares this DAG traversal with a simple linear
parent chain.
"""

from __future__ import annotations

import timeit

import networkx as nx


def build_dag() -> nx.DiGraph:
    """Create a DAG with branching parentage."""
    g = nx.DiGraph()
    g.add_node("material", metadata={"material": "NMC"})
    g.add_node("slurry", metadata={"slurry": "S"})
    g.add_node("electrode_a", metadata={"electrode": "E1"})
    g.add_node("electrode_b", metadata={"electrode": "E2"})
    g.add_node("cell", metadata={"cell": "C"})
    # edges
    g.add_edge("material", "slurry")
    g.add_edge("slurry", "electrode_a")
    g.add_edge("slurry", "electrode_b")
    g.add_edge("electrode_a", "cell")
    g.add_edge("electrode_b", "cell")
    return g


def merged_metadata(g: nx.DiGraph, node: str) -> dict:
    """Merge metadata from ``node`` and all of its ancestors."""
    merged: dict = {}
    for ancestor in nx.ancestors(g, node):
        merged.update(g.nodes[ancestor].get("metadata", {}))
    merged.update(g.nodes[node].get("metadata", {}))
    return merged


# --- Linear chain for comparison ------------------------------------------------

class Node:
    def __init__(self, name: str, metadata: dict | None = None, parent: "Node | None" = None):
        self.name = name
        self.metadata = metadata or {}
        self.parent = parent


def build_linear_chain() -> Node:
    material = Node("material", {"material": "NMC"})
    slurry = Node("slurry", {"slurry": "S"}, material)
    electrode = Node("electrode", {"electrode": "E"}, slurry)
    cell = Node("cell", {"cell": "C"}, electrode)
    return cell


def linear_merged_metadata(node: Node) -> dict:
    merged: dict = {}
    current = node
    while current is not None:
        merged = {**current.metadata, **merged}
        current = current.parent
    return merged


def benchmark(repeats: int = 1000) -> None:
    cell = build_linear_chain()
    g = build_dag()
    linear_time = timeit.timeit(lambda: linear_merged_metadata(cell), number=repeats)
    dag_time = timeit.timeit(lambda: merged_metadata(g, "cell"), number=repeats)
    print(f"Linear chain: {linear_time:.4f}s for {repeats} calls")
    print(f"DAG traversal: {dag_time:.4f}s for {repeats} calls")


if __name__ == "__main__":  # pragma: no cover - benchmark script
    benchmark()
