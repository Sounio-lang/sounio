#!/usr/bin/env python3
"""Degree-preserving graph shuffle fixtures for SWOW/CPC graph-edge probes.

This consumes the graph-edge multifixture manifest and creates deterministic
degree-preserving shuffles of each 16-node induced support. The output is a
structural fixture manifest only: it is not a hypothesis test, not a graph null
model for inference, not exact ORC, and not a biomarker artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import semantic_orc_swow16_fixture as base
import semantic_orc_swow16_graph_edge_fixture as graph_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multifixture-manifest", type=Path, required=True)
    parser.add_argument("--shuffle-count", type=int, default=16)
    parser.add_argument("--swap-attempts", type=int, default=256)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def edge_key(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise RuntimeError("self-loop edge key")
    return (left, right) if left < right else (right, left)


def induced_edges(selected: list[int], adjacency: dict[int, dict[int, float]]) -> set[tuple[int, int]]:
    support = set(selected)
    edges: set[tuple[int, int]] = set()
    for left in selected:
        for right in adjacency[left]:
            if right in support and left < right:
                edges.add((left, right))
    return edges


def degree_sequence(selected: list[int], edges: set[tuple[int, int]]) -> list[int]:
    degrees = {node_id: 0 for node_id in selected}
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    return [degrees[node_id] for node_id in selected]


def adjacency_from_edges(selected: list[int], edges: set[tuple[int, int]]) -> dict[int, dict[int, float]]:
    adjacency = {node_id: {} for node_id in selected}
    for left, right in edges:
        adjacency[left][right] = 1.0
        adjacency[right][left] = 1.0
    return adjacency


def connected_components(selected: list[int], edges: set[tuple[int, int]]) -> int:
    adjacency = {node_id: [] for node_id in selected}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    remaining = set(selected)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def deterministic_shuffle_edges(
    original_edges: set[tuple[int, int]],
    *,
    seed_text: str,
    swap_attempts: int,
) -> tuple[set[tuple[int, int]], int]:
    rng = random.Random(int.from_bytes(seed_text.encode("utf-8"), "little", signed=False))
    edges = set(original_edges)
    accepted = 0
    for _ in range(swap_attempts):
        if len(edges) < 2:
            break
        first, second = rng.sample(sorted(edges), 2)
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            new_first = edge_key(a, d)
            new_second = edge_key(c, b)
        else:
            new_first = edge_key(a, c)
            new_second = edge_key(b, d)
        if new_first == new_second:
            continue
        if new_first in edges or new_second in edges:
            continue
        edges.remove(first)
        edges.remove(second)
        edges.add(new_first)
        edges.add(new_second)
        accepted += 1
    return edges, accepted


def transport_probe_for_edges(
    selected: list[int],
    source: int,
    target: int,
    edges: set[tuple[int, int]],
    features: dict[int, dict[str, str]],
) -> dict[str, float]:
    rows = [features[node_id] for node_id in selected]
    cost = base.semantic_cost(rows)
    adjacency = adjacency_from_edges(selected, edges)
    mu, mu_stats = graph_edge.lazy_walk_measure(source, selected, adjacency)
    nu, nu_stats = graph_edge.lazy_walk_measure(target, selected, adjacency)
    k_base2 = [base.f32((-value / 0.05) / math.log(2.0)) for value in cost]
    la_log2 = [base.f32(math.log2(value)) for value in mu]
    lb_log2 = [base.f32(math.log2(value)) for value in nu]
    oracle = base.sinkhorn_oracle_from_logs(
        mu,
        nu,
        la_log2,
        lb_log2,
        k_base2,
        iterations=16,
        base2=True,
    )
    transport = base.plan_transport_cost(oracle["plan"], cost)
    endpoint_cost = cost[1]
    return {
        "transport_cost": transport,
        "one_minus_transport_over_endpoint_cost_probe": 1.0 - transport / endpoint_cost,
        "oracle_mass": float(oracle["mass"]),
        "oracle_max_row_err": float(oracle["max_row_err"]),
        "oracle_max_col_err": float(oracle["max_col_err"]),
        "mu_floor_mass_fraction": float(mu_stats["floor_mass_fraction"]),
        "nu_floor_mass_fraction": float(nu_stats["floor_mass_fraction"]),
    }


def main() -> int:
    args = parse_args()
    if args.shuffle_count <= 0:
        raise RuntimeError("shuffle-count must be positive")
    if args.swap_attempts <= 0:
        raise RuntimeError("swap-attempts must be positive")
    multifixture = json.loads(args.multifixture_manifest.read_text())
    if multifixture.get("schema") != "sounio.semantic_orc.swow16_graph_edge_multifixture_manifest.v1":
        raise RuntimeError("unexpected graph-edge multifixture manifest schema")

    records = []
    for record in multifixture["records"]:
        fixture_manifest_path = Path(record["fixture_manifest"])
        fixture_manifest = json.loads(fixture_manifest_path.read_text())
        feature_path = Path(fixture_manifest["feature_file"])
        edge_file = Path(fixture_manifest["edge_file"])
        features = base.load_features(feature_path)
        name_to_id = {row["node"]: node_id for node_id, row in features.items()}
        _, adjacency, graph_stats = graph_edge.load_graph_edges(edge_file, name_to_id)
        selected = [int(value) for value in fixture_manifest["selected_node_ids"]]
        source = int(fixture_manifest["edge_source_node_id"])
        target = int(fixture_manifest["edge_target_node_id"])
        original_edges = induced_edges(selected, adjacency)
        original_degrees = degree_sequence(selected, original_edges)
        original_probe = transport_probe_for_edges(selected, source, target, original_edges, features)
        shuffle_records = []
        for shuffle_index in range(args.shuffle_count):
            shuffled_edges, accepted_swaps = deterministic_shuffle_edges(
                original_edges,
                seed_text=(
                    "sounio-semantic-orc-degree-preserving-shuffle-v1:"
                    f"{record['edge_index']}:{record['edge_offset']}:{shuffle_index}"
                ),
                swap_attempts=args.swap_attempts,
            )
            shuffled_degrees = degree_sequence(selected, shuffled_edges)
            if shuffled_degrees != original_degrees:
                raise RuntimeError("degree-preserving shuffle changed the degree sequence")
            probe = transport_probe_for_edges(selected, source, target, shuffled_edges, features)
            if abs(probe["oracle_mass"] - 1.0) >= 0.02:
                raise RuntimeError("shuffled probe oracle mass outside tolerance")
            if probe["oracle_max_row_err"] >= 0.015 or probe["oracle_max_col_err"] >= 0.015:
                raise RuntimeError("shuffled probe oracle marginal error outside tolerance")
            overlap = len(original_edges.intersection(shuffled_edges))
            union = len(original_edges.union(shuffled_edges))
            shuffle_records.append(
                {
                    "shuffle_index": shuffle_index,
                    "accepted_edge_swaps": accepted_swaps,
                    "edge_count": len(shuffled_edges),
                    "degree_sequence": shuffled_degrees,
                    "degree_sequence_preserved": True,
                    "connected_components": connected_components(selected, shuffled_edges),
                    "edge_jaccard_vs_observed": overlap / union if union else 1.0,
                    **probe,
                }
            )
        records.append(
            {
                "edge_index": int(record["edge_index"]),
                "edge_offset": int(record["edge_offset"]),
                "edge_source_node_name": record["edge_source_node_name"],
                "edge_target_node_name": record["edge_target_node_name"],
                "selected_node_ids": selected,
                "selected_node_names": fixture_manifest["selected_node_names"],
                "fixture_manifest": str(fixture_manifest_path),
                "observed_induced_edge_count": len(original_edges),
                "observed_connected_components": connected_components(selected, original_edges),
                "observed_degree_sequence": original_degrees,
                "observed_transport_probe": original_probe,
                "shuffle_count": args.shuffle_count,
                "shuffle_records": shuffle_records,
                "source_graph_stats": graph_stats,
            }
        )

    all_shuffle_records = [
        shuffle for record in records for shuffle in record["shuffle_records"]
    ]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "schema": "sounio.semantic_orc.swow16_graph_degree_shuffle_fixture_manifest.v1",
                "status": "pass",
                "source_multifixture_manifest": str(args.multifixture_manifest),
                "source_multifixture_manifest_sha256": base.sha256(args.multifixture_manifest),
                "edge_fixture_count": len(records),
                "shuffle_count_per_edge_fixture": args.shuffle_count,
                "swap_attempts_per_shuffle": args.swap_attempts,
                "fixture_kind": "degree_preserving_induced_support_shuffle",
                "runtime_role": "structural fixture over graph-edge Sounio/K-AXI pack manifests; not a standalone GPU run",
                "degree_preservation_scope": "degree sequence is preserved only inside each induced 16-node support; full-graph degree distribution is not preserved or claimed",
                "degree_sequence_preserved_for_all_shuffles": all(
                    shuffle["degree_sequence_preserved"]
                    for shuffle in all_shuffle_records
                ),
                "accepted_edge_swap_range": [
                    min(shuffle["accepted_edge_swaps"] for shuffle in all_shuffle_records),
                    max(shuffle["accepted_edge_swaps"] for shuffle in all_shuffle_records),
                ],
                "edge_jaccard_vs_observed_range": [
                    min(shuffle["edge_jaccard_vs_observed"] for shuffle in all_shuffle_records),
                    max(shuffle["edge_jaccard_vs_observed"] for shuffle in all_shuffle_records),
                ],
                "transport_cost_range": [
                    min(shuffle["transport_cost"] for shuffle in all_shuffle_records),
                    max(shuffle["transport_cost"] for shuffle in all_shuffle_records),
                ],
                "one_minus_transport_over_endpoint_cost_probe_range": [
                    min(
                        shuffle["one_minus_transport_over_endpoint_cost_probe"]
                        for shuffle in all_shuffle_records
                    ),
                    max(
                        shuffle["one_minus_transport_over_endpoint_cost_probe"]
                        for shuffle in all_shuffle_records
                    ),
                ],
                "boundaries": [
                    "degree_preserving_structural_fixture_only",
                    "not_a_statistical_null_model",
                    "statistical_inference_claims_not_enforced",
                    "not_a_hypothesis_test",
                    "not_exact_ollivier_ricci_curvature",
                    "not_an_orc_estimator",
                    "not_a_biomarker",
                    "not_a_clinical_claim",
                    "not_a_generalizability_claim",
                ],
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"semantic_orc_swow16_graph_shuffle_fixture: PASS artifact={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
