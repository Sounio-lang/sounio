"""Provenance chain — directed acyclic graph tracking Knowledge computation history.

Each arithmetic operation on a tracked Knowledge value creates a new ProvenanceNode
linking back to its parent nodes. The full ancestry of any value can be traversed,
exported to JSON, or rendered as a Graphviz DOT diagram.

Usage
-----
>>> from sounio.knowledge import Knowledge
>>> a = Knowledge.tracked(500.0, 2.5, "calibration_batch")
>>> b = Knowledge.tracked(1.2, 0.05, "correction_factor")
>>> c = a * b
>>> print(c.chain.summary())
ProvenanceChain: 3 nodes (2 sources, 1 operations)
>>> print(c.chain.to_dot())
digraph ProvenanceChain { ... }
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class ProvenanceNode:
    """A single node in the provenance DAG.

    Attributes
    ----------
    id : str
        UUID identifying this node.
    label : str
        Human-readable name (e.g., "lipinski_screen", "pk_half_life").
    operation : str
        The operation that produced this node.
        One of: "source", "add", "sub", "mul", "div", "scale", "neg", "abs".
    value : float
        Central value at this node.
    epsilon : float
        Standard uncertainty at this node.
    timestamp : float
        Unix timestamp of creation.
    parent_ids : list[str]
        IDs of parent nodes (empty for source nodes).
    metadata : dict
        Optional extra data (units, version tags, etc.).
    """

    id: str
    label: str
    operation: str
    value: float
    epsilon: float
    timestamp: float
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "operation": self.operation,
            "value": self.value,
            "epsilon": self.epsilon,
            "timestamp": self.timestamp,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
        }


class ProvenanceChain:
    """A directed acyclic graph of ProvenanceNode objects.

    Nodes are added as Knowledge values are created and combined. The chain
    is automatically merged when two tracked Knowledge values are combined
    arithmetically.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProvenanceNode] = {}

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def add_source(
        self,
        label: str,
        value: float,
        epsilon: float,
        **metadata,
    ) -> ProvenanceNode:
        """Add a root node (a measurement source)."""
        node = ProvenanceNode(
            id=str(uuid.uuid4()),
            label=label,
            operation="source",
            value=value,
            epsilon=epsilon,
            timestamp=time.time(),
            parent_ids=[],
            metadata=dict(metadata),
        )
        self._nodes[node.id] = node
        return node

    def add_node(
        self,
        label: str,
        operation: str,
        value: float,
        epsilon: float,
        parent_ids: List[str],
        **metadata,
    ) -> ProvenanceNode:
        """Add a derived node produced by an operation."""
        node = ProvenanceNode(
            id=str(uuid.uuid4()),
            label=label,
            operation=operation,
            value=value,
            epsilon=epsilon,
            timestamp=time.time(),
            parent_ids=[pid for pid in parent_ids if pid in self._nodes],
            metadata=dict(metadata),
        )
        self._nodes[node.id] = node
        return node

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def parents_of(self, node_id: str) -> List[ProvenanceNode]:
        """Return direct parents of a node."""
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[pid] for pid in node.parent_ids if pid in self._nodes]

    def ancestors_of(self, node_id: str) -> List[ProvenanceNode]:
        """Return all ancestors of a node (full DAG walk, breadth-first)."""
        visited: set = set()
        result: List[ProvenanceNode] = []
        queue = list(self.parents_of(node_id))
        while queue:
            node = queue.pop(0)
            if node.id in visited:
                continue
            visited.add(node.id)
            result.append(node)
            queue.extend(self.parents_of(node.id))
        return result

    def node(self, node_id: str) -> Optional[ProvenanceNode]:
        """Look up a node by ID."""
        return self._nodes.get(node_id)

    def __iter__(self) -> Iterator[ProvenanceNode]:
        return iter(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Export the full chain as a JSON-serialisable dict."""
        return {"nodes": [n.to_dict() for n in self._nodes.values()]}

    def to_json(self, indent: int = 2) -> str:
        """Export the full chain as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceChain":
        """Reconstruct a ProvenanceChain from a dict produced by to_dict()."""
        chain = cls()
        for nd in data.get("nodes", []):
            node = ProvenanceNode(
                id=nd["id"],
                label=nd["label"],
                operation=nd["operation"],
                value=nd["value"],
                epsilon=nd["epsilon"],
                timestamp=nd["timestamp"],
                parent_ids=nd.get("parent_ids", []),
                metadata=nd.get("metadata", {}),
            )
            chain._nodes[node.id] = node
        return chain

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        """Export the chain as a Graphviz DOT string.

        Render with: echo '...' | dot -Tpng -o chain.png
        """
        lines = [
            "digraph ProvenanceChain {",
            "  rankdir=LR;",
            "  node [fontname=Helvetica fontsize=10];",
        ]
        for node in self._nodes.values():
            label = f"{node.label}\\n{node.value:.4g} ± {node.epsilon:.4g}"
            shape = "ellipse" if node.operation == "source" else "box"
            color = "#d0f0c0" if node.operation == "source" else "#c0d8f0"
            lines.append(
                f'  "{node.id}" [label="{label}" shape={shape}'
                f' style=filled fillcolor="{color}"];'
            )
        for node in self._nodes.values():
            for pid in node.parent_ids:
                lines.append(
                    f'  "{pid}" -> "{node.id}" [label="{node.operation}"];'
                )
        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> str:
        sources = sum(1 for n in self._nodes.values() if n.operation == "source")
        ops = len(self._nodes) - sources
        return (
            f"ProvenanceChain: {len(self._nodes)} nodes "
            f"({sources} sources, {ops} operations)"
        )

    def __repr__(self) -> str:
        return self.summary()


# ------------------------------------------------------------------
# Internal helper used by Knowledge arithmetic operators
# ------------------------------------------------------------------

def _merge_chains(
    a_chain: Optional[ProvenanceChain],
    a_node_id: Optional[str],
    b_chain: Optional[ProvenanceChain],
    b_node_id: Optional[str],
    label: str,
    operation: str,
    value: float,
    epsilon: float,
) -> tuple:
    """Merge two parent chains and add a new result node.

    Returns (merged_chain, result_node_id). If both chains are None,
    returns (None, None).
    """
    if a_chain is None and b_chain is None:
        return None, None

    merged = ProvenanceChain()
    if a_chain is not None:
        merged._nodes.update(a_chain._nodes)
    if b_chain is not None:
        merged._nodes.update(b_chain._nodes)

    parent_ids = []
    if a_node_id is not None:
        parent_ids.append(a_node_id)
    if b_node_id is not None:
        parent_ids.append(b_node_id)

    node = merged.add_node(
        label=label,
        operation=operation,
        value=value,
        epsilon=epsilon,
        parent_ids=parent_ids,
    )
    return merged, node.id


def _unary_chain(
    parent_chain: Optional[ProvenanceChain],
    parent_node_id: Optional[str],
    label: str,
    operation: str,
    value: float,
    epsilon: float,
) -> tuple:
    """Create a derived chain for a unary operation.

    Returns (new_chain, result_node_id). If parent_chain is None, returns (None, None).
    """
    if parent_chain is None:
        return None, None

    new_chain = ProvenanceChain()
    new_chain._nodes.update(parent_chain._nodes)
    parent_ids = [parent_node_id] if parent_node_id is not None else []
    node = new_chain.add_node(
        label=label,
        operation=operation,
        value=value,
        epsilon=epsilon,
        parent_ids=parent_ids,
    )
    return new_chain, node.id
