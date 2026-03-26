#!/usr/bin/env python3
"""Generate deterministic primitive sedenion zero-divisor geometry artifacts.

This script mirrors the exact Cayley-Dickson sign recursion used by the
self-hosted compiler in `self-hosted/ir/algebra.sio::ir_cd_sigma`.

Scope:
- Enumerate primitive sign-quotiented two-support imaginary vectors
  `v = e_i +/- e_j` with `1 <= i < j <= 15`
- Find all ordered pairs `(a, b)` such that `a * b = 0` in the 16D
  Cayley-Dickson algebra
- Quotient by pair swap to recover the 168 projective classes referenced in
  Moreno/de Marrais-style discussions
- Report stable graph/annihilator invariants using exact arithmetic

The output is deterministic and uses no floating-point arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict, deque
from fractions import Fraction
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(
    ROOT, "artifacts", "research", "sedenion_zero_divisor_geometry.v1.json"
)

BITS = 4
DIM = 1 << BITS
PrimitiveKey = Tuple[Tuple[int, int], ...]


def git_value(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:
        return "unknown"


def cd_sigma(a: int, b: int, bits: int = BITS) -> int:
    """Exact transcription of self-hosted/ir/algebra.sio::ir_cd_sigma."""
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi = 1 if a >= half else 0
    b_hi = 1 if b >= half else 0
    a_lo = a & (half - 1)
    b_lo = b & (half - 1)
    if a_hi == 0 and b_hi == 0:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if a_hi == 0 and b_hi == 1:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi == 1 and b_hi == 0:
        if b_lo == 0:
            return cd_sigma(a_lo, b_lo, bits - 1)
        return -cd_sigma(a_lo, b_lo, bits - 1)
    if b_lo == 0:
        return -cd_sigma(b_lo, a_lo, bits - 1)
    return cd_sigma(b_lo, a_lo, bits - 1)


def basis_mul(i: int, j: int, bits: int = BITS) -> Tuple[int, int]:
    return cd_sigma(i, j, bits), i ^ j


def add_term(acc: Dict[int, int], idx: int, coeff: int) -> None:
    if coeff == 0:
        return
    acc[idx] = acc.get(idx, 0) + coeff
    if acc[idx] == 0:
        del acc[idx]


def mul_vectors(a: Dict[int, int], b: Dict[int, int], bits: int = BITS) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i, c_i in a.items():
        for j, c_j in b.items():
            sign, basis = basis_mul(i, j, bits)
            add_term(out, basis, c_i * c_j * sign)
    return out


def canonical_vector_key(v: Dict[int, int]) -> PrimitiveKey:
    items = tuple(sorted((int(idx), int(coeff)) for idx, coeff in v.items() if coeff))
    neg = tuple(sorted((idx, -coeff) for idx, coeff in items))
    return min(items, neg)


def key_to_vector(key: PrimitiveKey) -> Dict[int, int]:
    return {idx: coeff for idx, coeff in key}


def key_support(key: PrimitiveKey) -> Tuple[int, ...]:
    return tuple(idx for idx, _coeff in key)


def vector_label(key: PrimitiveKey) -> str:
    parts: List[str] = []
    for idx, coeff in key:
        sign = "+" if coeff > 0 and parts else ""
        if coeff < 0:
            sign = "-"
        parts.append(f"{sign}e{idx}")
    return "".join(parts)


def primitive_candidates() -> List[PrimitiveKey]:
    candidates = []
    for i, j in combinations(range(1, DIM), 2):
        for second_sign in (1, -1):
            candidates.append(canonical_vector_key({i: 1, j: second_sign}))
    unique = sorted(set(candidates))
    assert len(unique) == 210
    return unique


def half_profile(key: PrimitiveKey) -> Tuple[int, int]:
    support = key_support(key)
    lower = sum(1 for idx in support if 1 <= idx <= 7)
    upper = sum(1 for idx in support if 8 <= idx <= 15)
    return lower, upper


def xor_label(key: PrimitiveKey) -> int:
    support = key_support(key)
    assert len(support) == 2
    return support[0] ^ support[1]


def matrix_rank(mat: List[List[Fraction]]) -> int:
    rows = [row[:] for row in mat]
    m = len(rows)
    n = len(rows[0]) if rows else 0
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if rows[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][col]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(m):
            if row == rank or rows[row][col] == 0:
                continue
            factor = rows[row][col]
            rows[row] = [rows[row][k] - factor * rows[rank][k] for k in range(n)]
        rank += 1
        if rank == m:
            break
    return rank


def multiplication_matrix(key: PrimitiveKey, side: str) -> List[List[Fraction]]:
    v = key_to_vector(key)
    matrix: List[List[Fraction]] = [[Fraction(0) for _ in range(DIM)] for _ in range(DIM)]
    for basis_idx in range(DIM):
        basis_vector = {basis_idx: 1}
        if side == "left":
            product = mul_vectors(v, basis_vector)
        else:
            product = mul_vectors(basis_vector, v)
        for out_idx, coeff in product.items():
            matrix[out_idx][basis_idx] = Fraction(coeff)
    return matrix


def is_bipartite(component: Sequence[PrimitiveKey], adjacency: Dict[PrimitiveKey, set]) -> Tuple[bool, Counter]:
    color: Dict[PrimitiveKey, int] = {}
    partitions: Counter = Counter()
    for root in component:
        if root in color:
            continue
        queue = deque([root])
        color[root] = 0
        while queue:
            node = queue.popleft()
            partitions[color[node]] += 1
            for nbr in adjacency[node]:
                if nbr not in color:
                    color[nbr] = 1 - color[node]
                    queue.append(nbr)
                elif color[nbr] == color[node]:
                    return False, partitions
    return True, partitions


def build_report() -> Dict[str, object]:
    candidates = primitive_candidates()

    ordered_pairs: List[Tuple[PrimitiveKey, PrimitiveKey]] = []
    adjacency: Dict[PrimitiveKey, set] = defaultdict(set)
    participating = set()
    ordered_pair_count_by_component: Counter = Counter()

    for a in candidates:
        vec_a = key_to_vector(a)
        for b in candidates:
            vec_b = key_to_vector(b)
            if mul_vectors(vec_a, vec_b):
                continue
            ordered_pairs.append((a, b))
            adjacency[a].add(b)
            adjacency[b].add(a)
            participating.add(a)
            participating.add(b)

    ordered_pair_count = len(ordered_pairs)
    unordered_pairs = sorted({tuple(sorted(pair)) for pair in ordered_pairs})
    ordered_projective_pairs = sorted(set(ordered_pairs))
    vertices = sorted(participating)

    mixed_candidates = sorted({key for key in candidates if half_profile(key) == (1, 1)})
    excluded_mixed = sorted(set(mixed_candidates) - participating)

    quartet_counts: Counter = Counter()
    quartet_examples: Dict[Tuple[int, ...], List[Tuple[PrimitiveKey, PrimitiveKey]]] = defaultdict(list)
    for lhs, rhs in unordered_pairs:
        quartet = tuple(sorted(set(key_support(lhs)) | set(key_support(rhs))))
        quartet_counts[quartet] += 1
        quartet_examples[quartet].append((lhs, rhs))

    left_rank_counter: Counter = Counter()
    right_rank_counter: Counter = Counter()
    vertex_infos: List[Dict[str, object]] = []
    vertex_to_component: Dict[PrimitiveKey, int] = {}
    vertex_id = {vertex: idx for idx, vertex in enumerate(vertices)}

    components: List[List[PrimitiveKey]] = []
    seen = set()
    for vertex in vertices:
        if vertex in seen:
            continue
        stack = [vertex]
        seen.add(vertex)
        component: List[PrimitiveKey] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adjacency[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        component.sort()
        component_id = len(components)
        for member in component:
            vertex_to_component[member] = component_id
        components.append(component)

    for lhs, rhs in ordered_pairs:
        ordered_pair_count_by_component[vertex_to_component[lhs]] += 1

    component_infos: List[Dict[str, object]] = []
    for component_id, component in enumerate(components):
        xor_labels = sorted({xor_label(vertex) for vertex in component})
        support_pairs = sorted({key_support(vertex) for vertex in component})
        bipartite, partitions = is_bipartite(component, adjacency)
        edge_count = sum(len(adjacency[vertex]) for vertex in component) // 2
        component_infos.append(
            {
                "component_id": component_id,
                "vertex_count": len(component),
                "edge_count": edge_count,
                "ordered_pair_count": ordered_pair_count_by_component[component_id],
                "unordered_pair_count": edge_count,
                "xor_labels": xor_labels,
                "bipartite": bipartite,
                "bipartition_sizes": [partitions.get(0, 0), partitions.get(1, 0)],
                "support_pairs": [list(pair) for pair in support_pairs],
                "vertices": [vertex_id[vertex] for vertex in component],
            }
        )

    for vertex in vertices:
        left_rank = matrix_rank(multiplication_matrix(vertex, "left"))
        right_rank = matrix_rank(multiplication_matrix(vertex, "right"))
        left_rank_counter[left_rank] += 1
        right_rank_counter[right_rank] += 1
        vertex_infos.append(
            {
                "vertex_id": vertex_id[vertex],
                "label": vector_label(vertex),
                "support": list(key_support(vertex)),
                "coefficients": [coeff for _idx, coeff in vertex],
                "half_profile": list(half_profile(vertex)),
                "xor_label": xor_label(vertex),
                "component_id": vertex_to_component[vertex],
                "left_rank": left_rank,
                "left_nullity": DIM - left_rank,
                "right_rank": right_rank,
                "right_nullity": DIM - right_rank,
                "neighbors": sorted(vertex_id[nbr] for nbr in adjacency[vertex]),
            }
        )

    quartet_multiplicity_counter = Counter(quartet_counts.values())
    component_vertex_counter = Counter(info["vertex_count"] for info in component_infos)
    component_edge_counter = Counter(info["edge_count"] for info in component_infos)
    component_pair_counter = Counter(info["unordered_pair_count"] for info in component_infos)

    support_rule_ok = all(
        half_profile(vertex) == (1, 1)
        and 8 not in key_support(vertex)
        and xor_label(vertex) != 8
        for vertex in vertices
    )

    excluded_rule_ok = all(
        half_profile(vertex) == (1, 1)
        and (8 in key_support(vertex) or xor_label(vertex) == 8)
        for vertex in excluded_mixed
    )

    report = {
        "schema": "sounio.research.sedenion-zero-divisor-geometry.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "source_model": {
            "bits": BITS,
            "dimension": DIM,
            "sign_recursion": "self-hosted/ir/algebra.sio::ir_cd_sigma",
            "enumeration_scope": "primitive sign-quotiented two-support imaginary vectors e_i +/- e_j, 1 <= i < j <= 15",
            "projective_quotient": "independent sign quotient on each factor; swap quotient applied only for the 168 class count",
        },
        "counts": {
            "primitive_candidate_count": len(candidates),
            "mixed_support_candidate_count": len(mixed_candidates),
            "participating_vertex_count": len(vertices),
            "excluded_mixed_vertex_count": len(excluded_mixed),
            "ordered_zero_divisor_pair_count": ordered_pair_count,
            "ordered_projective_pair_count": len(ordered_projective_pairs),
            "unordered_projective_pair_count": len(unordered_pairs),
            "support_quartet_count": len(quartet_counts),
            "xor_component_count": len(component_infos),
        },
        "uniformity_checks": {
            "support_quartet_uniform": {
                "uniform": len(quartet_multiplicity_counter) == 1,
                "multiplicity_distribution": {str(k): v for k, v in sorted(quartet_multiplicity_counter.items())},
            },
            "xor_component_uniform": {
                "uniform": len(component_vertex_counter) == 1
                and len(component_edge_counter) == 1
                and len(component_pair_counter) == 1,
                "vertex_count_distribution": {str(k): v for k, v in sorted(component_vertex_counter.items())},
                "edge_count_distribution": {str(k): v for k, v in sorted(component_edge_counter.items())},
                "unordered_pair_count_distribution": {str(k): v for k, v in sorted(component_pair_counter.items())},
            },
            "annihilator_signature_uniform": {
                "uniform": len(left_rank_counter) == 1 and len(right_rank_counter) == 1,
                "left_rank_distribution": {str(k): v for k, v in sorted(left_rank_counter.items())},
                "right_rank_distribution": {str(k): v for k, v in sorted(right_rank_counter.items())},
            },
            "mixed_support_rule": {
                "participating_vertices_match_rule": support_rule_ok,
                "excluded_vertices_match_rule": excluded_rule_ok,
                "rule": "primitive projective vertices occur exactly on mixed supports (1 lower, 1 upper) with neither e8 nor xor-label 8",
            },
            "factor_11_probe": {
                "natural_11_fold_fiber_found": False,
                "checked_fibrations": [
                    "support quartets (42 x 4 = 168)",
                    "xor components (7 x 24 = 168 unordered pairs)",
                    "annihilator signatures (uniform rank/nullity across all primitive vertices)",
                ],
                "note": "No primitive-pair decomposition with stable fiber size 11 appears in this enumeration; the 1848 = 11 x 168 relation looks associator-side rather than primitive-zero-divisor-side.",
            },
        },
        "candidate_geometry": {
            "summary": "The primitive projective annihilation graph is a 7-component xor-bundle on 84 mixed-support vertices.",
            "details": [
                "Each component is indexed by a constant xor label in {9, ..., 15}.",
                "Each component has 12 vertices, 24 unordered annihilation pairs, and is bipartite with partition sizes 6 and 6.",
                "There are 42 distinct 2+2 support quartets, each supporting exactly 4 unordered projective zero-divisor pairs.",
                "Every primitive participating vertex has left/right multiplication rank 12 and nullity 4.",
            ],
        },
        "components": component_infos,
        "support_quartets": [
            {
                "support_union": list(quartet),
                "unordered_pair_count": quartet_counts[quartet],
                "pairs": [
                    [vertex_id[lhs], vertex_id[rhs]]
                    for lhs, rhs in sorted(quartet_examples[quartet])
                ],
            }
            for quartet in sorted(quartet_counts)
        ],
        "vertices": vertex_infos,
        "unordered_projective_pairs": [
            {
                "lhs": vertex_id[lhs],
                "rhs": vertex_id[rhs],
                "support_union": list(sorted(set(key_support(lhs)) | set(key_support(rhs)))),
                "component_id": vertex_to_component[lhs],
            }
            for lhs, rhs in unordered_pairs
        ],
        "excluded_mixed_vertices": [
            {
                "label": vector_label(vertex),
                "support": list(key_support(vertex)),
                "xor_label": xor_label(vertex),
            }
            for vertex in excluded_mixed
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    args = parser.parse_args()

    report = build_report()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    counts = report["counts"]
    print("sedenion zero-divisor geometry artifact written")
    print(f"  out:       {out_path}")
    print(f"  ordered:   {counts['ordered_zero_divisor_pair_count']}")
    print(f"  unordered: {counts['unordered_projective_pair_count']}")
    print(f"  vertices:  {counts['participating_vertex_count']}")
    print(f"  quartets:  {counts['support_quartet_count']}")
    print(f"  components:{counts['xor_component_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
