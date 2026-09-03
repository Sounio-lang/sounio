#!/usr/bin/env python3
"""Generate a graph-edge SWOW/CPC N=16 entropic-transport fixture.

This is the first graph-aware rung above the trajectory-support bridge. It uses
a real exported SWOW edge list, selects one semantic graph edge, builds two
strictly-positive lazy random-walk measures around the edge endpoints on a
bounded 16-node support, and emits the same K-AXI sinkhorn16 pack shape used by
the current runtime lane. It is still an entropic fixture, not exact ORC and
not a biomarker or clinical artifact. Sinkhorn16 here is a fixed-iteration,
entropy-regularized fixture path; it is not an exact optimal-transport solver,
not exact Wasserstein-1, and it records no convergence or error-bound
certificate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import semantic_orc_swow16_fixture as base


DEFAULT_INPUT_DIR = Path(
    "/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input"
)
DEFAULT_EDGE_FILE = Path(
    "/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv"
)
N = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--edge-file", type=Path, default=DEFAULT_EDGE_FILE)
    parser.add_argument("--edge-index", type=int, default=0)
    parser.add_argument("--edge-stride", type=int, default=1)
    parser.add_argument("--tile-index", type=int, default=0)
    parser.add_argument("--tile-stride", type=int, default=14)
    parser.add_argument("--sounio-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kaxi-pack", type=Path, required=True)
    return parser.parse_args()


def load_graph_edges(
    edge_file: Path, name_to_id: dict[str, int]
) -> tuple[list[tuple[int, int, float]], dict[int, dict[int, float]], dict[str, object]]:
    combined: dict[tuple[int, int], float] = {}
    skipped_unknown = 0
    skipped_self = 0
    raw_rows = 0
    with edge_file.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            source = row["source"]
            target = row["target"]
            if source not in name_to_id or target not in name_to_id:
                skipped_unknown += 1
                continue
            left = name_to_id[source]
            right = name_to_id[target]
            if left == right:
                skipped_self += 1
                continue
            if left > right:
                left, right = right, left
            weight = float(row.get("weight", "1.0"))
            if not math.isfinite(weight) or weight <= 0.0:
                weight = 1.0
            combined[(left, right)] = combined.get((left, right), 0.0) + weight

    edges = [(left, right, weight) for (left, right), weight in combined.items()]
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    adjacency: dict[int, dict[int, float]] = {node_id: {} for node_id in name_to_id.values()}
    for left, right, weight in edges:
        adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
        adjacency[right][left] = adjacency[right].get(left, 0.0) + weight
    stats = {
        "raw_rows": raw_rows,
        "usable_undirected_edges": len(edges),
        "skipped_unknown_endpoint_rows": skipped_unknown,
        "skipped_self_loop_rows": skipped_self,
        "covered_feature_nodes": sum(1 for values in adjacency.values() if values),
    }
    return edges, adjacency, stats


def feature_vector(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row["feature_poincare_x"]),
        float(row["feature_poincare_y"]),
        0.5 * float(row["feature_valence"]),
        0.25 * float(row["feature_c_ent"]),
    )


def squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def select_edge(edges: list[tuple[int, int, float]], edge_index: int, edge_stride: int) -> tuple[int, int, float, int]:
    if edge_index < 0:
        raise RuntimeError("edge-index must be non-negative")
    if edge_stride <= 0:
        raise RuntimeError("edge-stride must be positive")
    offset = edge_index * edge_stride
    if offset >= len(edges):
        raise RuntimeError(f"edge selection offset {offset} exceeds {len(edges)} usable edges")
    return (*edges[offset], offset)


def add_node(selected: list[int], seen: set[int], node_id: int) -> None:
    if node_id not in seen and len(selected) < N:
        selected.append(node_id)
        seen.add(node_id)


def select_support(
    source: int,
    target: int,
    adjacency: dict[int, dict[int, float]],
    features: dict[int, dict[str, str]],
    *,
    tile_index: int,
    tile_stride: int,
) -> tuple[list[int], dict[str, object]]:
    if tile_index < 0:
        raise RuntimeError("tile-index must be non-negative")
    if tile_stride <= 0:
        raise RuntimeError("tile-stride must be positive")
    selected: list[int] = []
    seen: set[int] = set()
    add_node(selected, seen, source)
    add_node(selected, seen, target)

    def ranked_neighbors(center: int) -> list[int]:
        return [
            node_id
            for node_id, _ in sorted(
                adjacency[center].items(),
                key=lambda item: (-item[1], features[item[0]]["node"], item[0]),
            )
        ]

    source_neighbors = ranked_neighbors(source)
    target_neighbors = ranked_neighbors(target)
    candidates: list[int] = []
    candidate_seen: set[int] = {source, target}
    for idx in range(max(len(source_neighbors), len(target_neighbors))):
        if idx < len(source_neighbors):
            node_id = source_neighbors[idx]
            if node_id not in candidate_seen:
                candidates.append(node_id)
                candidate_seen.add(node_id)
        if idx < len(target_neighbors):
            node_id = target_neighbors[idx]
            if node_id not in candidate_seen:
                candidates.append(node_id)
                candidate_seen.add(node_id)

    source_vec = feature_vector(features[source])
    target_vec = feature_vector(features[target])
    midpoint = tuple((a + b) * 0.5 for a, b in zip(source_vec, target_vec))
    fill = sorted(
        (node_id for node_id in features if node_id not in candidate_seen),
        key=lambda node_id: (
            squared_distance(feature_vector(features[node_id]), midpoint),
            features[node_id]["node"],
            node_id,
        ),
    )
    candidates.extend(fill)
    extra_start = tile_index * tile_stride
    extra_end = extra_start + (N - 2)
    extra_nodes = candidates[extra_start:extra_end]
    if len(extra_nodes) != N - 2:
        raise RuntimeError(
            f"could only select {len(extra_nodes)} extra tile nodes at "
            f"tile_index={tile_index} tile_stride={tile_stride} from {len(candidates)} candidates"
        )
    for node_id in extra_nodes:
        add_node(selected, seen, node_id)

    if len(selected) != N:
        raise RuntimeError(f"could only select {len(selected)} graph support nodes")
    support_set = set(selected)
    induced_edges = 0
    incident_source = 0
    incident_target = 0
    for left in support_set:
        for right in adjacency[left]:
            if right in support_set and left < right:
                induced_edges += 1
            if left == source and right in support_set:
                incident_source += 1
            if left == target and right in support_set:
                incident_target += 1
    return selected, {
        "selector": "edge endpoints plus alternating top-weighted endpoint neighbors, filled by semantic-vector nearest nodes if needed",
        "tile_index": tile_index,
        "tile_stride": tile_stride,
        "tile_extra_candidate_offset": extra_start,
        "tile_extra_candidate_count": len(extra_nodes),
        "candidate_count": len(candidates),
        "induced_undirected_edges": induced_edges,
        "source_incident_support_neighbors": incident_source,
        "target_incident_support_neighbors": incident_target,
    }


def lazy_walk_measure(
    center: int,
    support: list[int],
    adjacency: dict[int, dict[int, float]],
) -> tuple[list[float], dict[str, object]]:
    floor = 0.08
    laziness = 0.45
    values = [floor for _ in support]
    support_index = {node_id: idx for idx, node_id in enumerate(support)}
    values[support_index[center]] += laziness
    neighbor_weights = [
        (neighbor, weight)
        for neighbor, weight in adjacency[center].items()
        if neighbor in support_index and neighbor != center
    ]
    total_neighbor_weight = sum(weight for _, weight in neighbor_weights)
    if total_neighbor_weight <= 0.0:
        raise RuntimeError("selected graph edge endpoint has no support neighbor")
    for neighbor, weight in neighbor_weights:
        values[support_index[neighbor]] += (1.0 - laziness) * weight / total_neighbor_weight
    total = sum(values)
    normalized = [value / total for value in values]
    return normalized, {
        "floor": floor,
        "laziness": laziness,
        "support_neighbor_count": len(neighbor_weights),
        "support_neighbor_weight": total_neighbor_weight,
        "floor_mass_fraction": floor * len(support) / total,
    }


def emit_graph_sounio_source(
    mu: list[float],
    nu: list[float],
    cost: list[float],
    node_ids: list[int],
    node_names: list[str],
    source_name: str,
    target_name: str,
) -> str:
    text = base.emit_sounio_source(mu, nu, cost, node_ids, node_names)
    return (
        text.replace(
            "Generated SWOW/CPC N=16 entropic-transport runtime gate.",
            "Generated SWOW/CPC graph-edge N=16 entropic-transport runtime gate.",
        )
        .replace(
            "This is not exact W1, not a convergence theorem, and not a clinical claim.",
            f"Graph edge fixture: {source_name} -> {target_name}. This is not exact ORC, not a convergence theorem, and not a clinical or biomarker claim.",
        )
        .replace(
            'println("SOUNIO_SWOW16_ENTROPIC_TRANSPORT_PASS")',
            'println("SOUNIO_SWOW16_GRAPH_EDGE_TRANSPORT_PASS")',
        )
        .replace(
            'println("claim_level=swow16_fixture_runtime_gate")',
            'println("claim_level=swow16_graph_edge_fixture_runtime_gate")',
        )
        .replace(
            'println("fixture_source=cpc2026_sounio_input_node_features_parity")',
            'println("fixture_source=swow_english_final_graph_edge_neighborhood")',
        )
        .replace(
            'println("swow16_fixture_nodes=16")',
            'println("swow16_graph_edge_fixture_nodes=16")',
        )
    )


def main() -> int:
    args = parse_args()
    feature_path = args.input_dir / "node_features.csv"
    features = base.load_features(feature_path)
    name_to_id = {row["node"]: node_id for node_id, row in features.items()}
    edges, adjacency, graph_stats = load_graph_edges(args.edge_file, name_to_id)
    source, target, edge_weight, edge_offset = select_edge(edges, args.edge_index, args.edge_stride)
    selected, support_stats = select_support(
        source,
        target,
        adjacency,
        features,
        tile_index=args.tile_index,
        tile_stride=args.tile_stride,
    )
    rows = [features[node_id] for node_id in selected]
    names = [row["node"] for row in rows]
    mu, mu_stats = lazy_walk_measure(source, selected, adjacency)
    nu, nu_stats = lazy_walk_measure(target, selected, adjacency)
    cost = base.semantic_cost(rows)
    base.validate_fixture_arrays(mu, nu, cost)

    epsilon = 0.05
    k_nat = [(-c / epsilon) for c in cost]
    k_base2 = [base.f32(value / math.log(2.0)) for value in k_nat]
    la_log2 = [base.f32(math.log2(x)) for x in mu]
    lb_log2 = [base.f32(math.log2(x)) for x in nu]
    oracle_nat = base.sinkhorn_oracle(mu, nu, k_nat, iterations=16, base2=False)
    oracle_kaxi = base.sinkhorn_oracle_from_logs(
        mu,
        nu,
        la_log2,
        lb_log2,
        k_base2,
        iterations=16,
        base2=True,
    )
    init_mem = la_log2 + lb_log2 + k_base2 + [0.0] * 32

    source_name = features[source]["node"]
    target_name = features[target]["node"]
    args.sounio_out.parent.mkdir(parents=True, exist_ok=True)
    args.sounio_out.write_text(
        emit_graph_sounio_source(mu, nu, cost, selected, names, source_name, target_name)
    )
    manifest = {
        "schema": "sounio.semantic_orc.swow16_graph_edge_fixture_manifest.v1",
        "status": "pass",
        "generator": "scripts/research/semantic_orc_swow16_graph_edge_fixture.py",
        "generator_sha256": base.sha256(Path(__file__)),
        "input_dir": str(args.input_dir),
        "feature_file": str(feature_path),
        "feature_file_sha256": base.sha256(feature_path),
        "edge_file": str(args.edge_file),
        "edge_file_sha256": base.sha256(args.edge_file),
        "generated_sounio_source": str(args.sounio_out),
        "generated_sounio_source_sha256": base.sha256(args.sounio_out),
        "fixture_kind": "graph_edge_neighborhood_lazy_walk",
        "edge_index": args.edge_index,
        "edge_stride": args.edge_stride,
        "tile_index": args.tile_index,
        "tile_stride": args.tile_stride,
        "edge_offset": edge_offset,
        "edge_source_node_id": source,
        "edge_target_node_id": target,
        "edge_source_node_name": source_name,
        "edge_target_node_name": target_name,
        "edge_weight": edge_weight,
        "graph_stats": graph_stats,
        "support_stats": support_stats,
        "selected_node_ids": selected,
        "selected_node_names": names,
        "epsilon": epsilon,
        "measure_mu": "strictly-positive lazy random walk around edge_source on the selected 16-node support",
        "measure_nu": "strictly-positive lazy random walk around edge_target on the selected 16-node support",
        "measure_role": "one-step lazy random-walk fixture measures on the selected support; not stationary distributions and not validated ORC neighborhood measures",
        "measure_mu_stats": mu_stats,
        "measure_nu_stats": nu_stats,
        "cost": "zero diagonal; off-diagonal normalized Euclidean distance over poincare_x,poincare_y,0.5*valence,0.25*c_ent mapped to [0.02,0.50]",
        "domain_bridge": "generated Sounio source runs natural-log Sinkhorn; K-AXI pack stores the algebraically equivalent base-2 log-domain values la/log2, lb/log2, K/log(2)",
        "sounio_runtime_numeric_role": "self-contained approximate-math graph-edge runtime sanity gate; Python pack oracle is the K-AXI numeric reference",
        "edge_transport_probe": {
            "transport_cost": base.plan_transport_cost(oracle_kaxi["plan"], cost),
            "edge_distance_convention": "semantic feature cost between the selected edge endpoints, not graph shortest-path distance",
            "one_minus_transport_over_endpoint_cost_probe": 1.0 - base.plan_transport_cost(oracle_kaxi["plan"], cost) / cost[1],
            "probe_role": "engineering transport-ratio fixture scalar only; not exact Wasserstein-1, not exact Ollivier-Ricci curvature, not an ORC estimator, not a biomarker, and not validated for interpretation",
        },
        "claim_limits": [
            "graph_edge_fixture_only",
            "entropic_regularized_transport_only",
            "lazy_walk_measures_are_bounded_fixture_choices",
            "clinical_claims_not_enforced",
            "biomarker_claims_not_enforced",
            "statistical_inference_claims_not_enforced",
            "no_convergence_theorem",
            "not_exact_ollivier_ricci_curvature",
            "not_exact_wasserstein_distance",
            "no_sinkhorn_convergence_or_error_bound_certificate",
            "kaxi_pack_is_launch_input_not_gpu_result",
        ],
        "sounio_oracle": {
            "iterations": oracle_nat["iterations"],
            "mass": oracle_nat["mass"],
            "max_row_err": oracle_nat["max_row_err"],
            "max_col_err": oracle_nat["max_col_err"],
            "max_fixed_err": oracle_nat["max_fixed_err"],
            "max_fixed_self_consistency_residual": oracle_nat["max_fixed_self_consistency_residual"],
            "transport_cost": base.plan_transport_cost(oracle_nat["plan"], cost),
        },
    }
    base.write_json(args.manifest, manifest)
    base.write_json(
        args.kaxi_pack,
        {
            "schema": "sounio.semantic_orc.swow16_kaxi_sinkhorn16_pack.v1",
            "status": "pass",
            "fixture_kind": "graph_edge_neighborhood_lazy_walk",
            "edge_index": args.edge_index,
            "edge_stride": args.edge_stride,
            "tile_index": args.tile_index,
            "tile_stride": args.tile_stride,
            "edge_offset": edge_offset,
            "edge_source_node_id": source,
            "edge_target_node_id": target,
            "edge_source_node_name": source_name,
            "edge_target_node_name": target_name,
            "selected_node_ids": selected,
            "selected_node_names": names,
            "layout": "f32 init_mem[320] = la_log2[16], lb_log2[16], K_log2[256], u_v_scratch[32]",
            "serialization_precision": "all init_mem numeric segments rounded to IEEE-754 binary32 before CSV serialization",
            "kaxi_pattern": "sinkhorn16",
            "log_domain": "base2",
            "epsilon": epsilon,
            "la_log2": la_log2,
            "lb_log2": lb_log2,
            "k_log2": k_base2,
            "cost": cost,
            "init_mem_count": len(init_mem),
            "init_mem_csv": ",".join(base.fmt(x) for x in init_mem),
            "oracle": {
                "iterations": oracle_kaxi["iterations"],
                "mass": oracle_kaxi["mass"],
                "max_row_err": oracle_kaxi["max_row_err"],
                "max_col_err": oracle_kaxi["max_col_err"],
                "max_fixed_err": oracle_kaxi["max_fixed_err"],
                "max_fixed_self_consistency_residual": oracle_kaxi["max_fixed_self_consistency_residual"],
                "transport_cost": base.plan_transport_cost(oracle_kaxi["plan"], cost),
                "oracle_role": "16-iteration f32-input graph-edge fixture reference, not a convergence ground truth",
                "u": oracle_kaxi["u"],
                "v": oracle_kaxi["v"],
            },
            "boundaries": [
                "pack_only_no_gpu_launch",
                "runtime_acceptance_requires_kaxi_ptx_runner_or_slurm_gate",
                "graph_edge_fixture_only",
                "entropic_regularized_transport_only",
                "clinical_claims_not_enforced",
                "biomarker_claims_not_enforced",
                "statistical_inference_claims_not_enforced",
                "not_exact_ollivier_ricci_curvature",
                "not_exact_wasserstein_distance",
                "no_sinkhorn_convergence_or_error_bound_certificate",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
