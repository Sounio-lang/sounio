#!/usr/bin/env python3
"""Generate a Sounio reducer fixture for a graph-edge tile matrix manifest.

The generated Sounio source is intentionally a reducer/sanity gate over a
previously produced matrix manifest. It does not recompute transport, estimate
ORC, launch GPU work, or perform statistical inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import semantic_orc_swow16_fixture as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--sounio-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def bool_lit(value: bool) -> str:
    return "true" if value else "false"


def emit_reducer_source(matrix: dict[str, object]) -> str:
    edge_count = int(matrix["edge_count"])
    tile_count = int(matrix["tile_count_per_edge"])
    total_fixtures = int(matrix["total_edge_tile_fixtures"])
    unique_nodes = int(matrix["matrix_unique_endpoint_free_node_count"])
    tile_positions = int(matrix["matrix_endpoint_free_tile_positions"])
    diagnostics_only = bool(matrix["numerical_values_are_engineering_diagnostics_only"])
    records = matrix.get("records", [])
    record_count = len(records)
    tile_record_total = sum(int(record["tile_count"]) for record in records)
    endpoint_free_per_fixture = tile_positions // total_fixtures
    offsets = [int(value) for value in matrix.get("unique_edge_offsets", [])]
    offset_csv = ", ".join(str(value) for value in offsets)
    edge_names = ", ".join(
        f"{record['edge_source_node_name']}->{record['edge_target_node_name']}"
        for record in records
    )
    return f"""//@ run-pass
//! Generated SWOW/CPC graph-edge tile-matrix reducer fixture.
//! edge_offsets: {offset_csv}
//! edge_labels: {edge_names}
//! This reducer checks fixture manifest invariants only.
//! It is not exact W1, not exact ORC, not an ORC estimator, not a biomarker,
//! not statistical inference, and not a GPU runtime claim.

fn main() -> i64 with IO {{
    let edge_count: i64 = {edge_count}
    let tile_count_per_edge: i64 = {tile_count}
    let total_edge_tile_fixtures: i64 = {total_fixtures}
    let endpoint_free_unique: i64 = {unique_nodes}
    let endpoint_free_positions: i64 = {tile_positions}
    let endpoint_free_per_fixture: i64 = {endpoint_free_per_fixture}
    let matrix_record_count: i64 = {record_count}
    let matrix_tile_record_total: i64 = {tile_record_total}
    let diagnostics_only: bool = {bool_lit(diagnostics_only)}

    let shape_ok = edge_count > 0 &&
        tile_count_per_edge > 0 &&
        total_edge_tile_fixtures == edge_count * tile_count_per_edge &&
        endpoint_free_per_fixture == 14 &&
        endpoint_free_positions == total_edge_tile_fixtures * endpoint_free_per_fixture &&
        endpoint_free_unique > 0 &&
        endpoint_free_unique <= endpoint_free_positions &&
        matrix_record_count == edge_count &&
        matrix_tile_record_total == total_edge_tile_fixtures
    let claim_boundary_ok = diagnostics_only

    if shape_ok && claim_boundary_ok {{
        println("SOUNIO_SWOW16_GRAPH_EDGE_TILE_MATRIX_REDUCER_PASS")
        println("claim_level=swow16_graph_edge_tile_matrix_reducer_fixture")
        println("fixture_source=swow_english_final_graph_edge_tile_matrix_manifest")
        println("edge_count={edge_count}")
        println("tile_count_per_edge={tile_count}")
        println("total_edge_tile_fixtures={total_fixtures}")
        println("endpoint_free_unique={unique_nodes}")
        println("endpoint_free_positions={tile_positions}")
        println("diagnostic_values_only=true")
        println("fixed_iteration_sinkhorn16_only=true")
        println("entropic_regularized_transport_only=true")
        println("not_exact_wasserstein_distance=true")
        println("not_exact_ollivier_ricci_curvature=true")
        println("not_an_orc_estimator=true")
        println("clinical_claims_not_enforced=true")
        println("biomarker_claims_not_enforced=true")
        println("statistical_inference_claims_not_enforced=true")
        println("gpu_runtime_claims_not_enforced=true")
        println("population_level_claims_not_enforced=true")
        println("no_convergence_theorem=true")
        0
    }} else {{
        println("SOUNIO_SWOW16_GRAPH_EDGE_TILE_MATRIX_REDUCER_FAIL")
        1
    }}
}}
"""


def validate_matrix_manifest(matrix: dict[str, object]) -> None:
    if matrix.get("schema") != "sounio.semantic_orc.swow16_graph_edge_tile_matrix_manifest.v1":
        raise RuntimeError("unexpected matrix manifest schema")
    if matrix.get("status") != "pass":
        raise RuntimeError("matrix manifest must be pass")
    if not matrix.get("numerical_values_are_engineering_diagnostics_only"):
        raise RuntimeError("matrix manifest must mark numerical values as diagnostics only")
    boundaries = set(matrix.get("boundaries", []))
    required_boundaries = {
        "edge_by_tile_matrix_fixture_only",
        "not_exact_wasserstein_distance",
        "no_sinkhorn_convergence_or_error_bound_certificate",
        "not_exact_ollivier_ricci_curvature",
        "not_an_orc_estimator",
        "not_a_biomarker",
        "not_a_clinical_claim",
        "not_statistical_inference",
        "not_population_level_evidence",
        "not_generalizability_evidence",
    }
    missing = sorted(required_boundaries - boundaries)
    if missing:
        raise RuntimeError(f"matrix manifest missing boundaries: {missing}")
    edge_count = int(matrix["edge_count"])
    tile_count = int(matrix["tile_count_per_edge"])
    total_fixtures = int(matrix["total_edge_tile_fixtures"])
    if edge_count <= 0 or tile_count <= 0:
        raise RuntimeError("edge_count and tile_count_per_edge must be positive")
    if total_fixtures != edge_count * tile_count:
        raise RuntimeError("total_edge_tile_fixtures must equal edge_count * tile_count_per_edge")
    if len(matrix.get("records", [])) != edge_count:
        raise RuntimeError("matrix record count must match edge_count")
    tile_record_total = sum(int(record["tile_count"]) for record in matrix.get("records", []))
    if tile_record_total != total_fixtures:
        raise RuntimeError("sum(records[].tile_count) must equal total_edge_tile_fixtures")
    endpoint_free_positions = int(matrix["matrix_endpoint_free_tile_positions"])
    if endpoint_free_positions % total_fixtures != 0:
        raise RuntimeError("endpoint-free tile positions must divide evenly over fixtures")
    endpoint_free_per_fixture = endpoint_free_positions // total_fixtures
    if endpoint_free_per_fixture != 14:
        raise RuntimeError("endpoint-free positions per fixture must equal 14 for 16-node edge tiles")


def main() -> int:
    args = parse_args()
    matrix = json.loads(args.matrix_manifest.read_text())
    validate_matrix_manifest(matrix)

    args.sounio_out.parent.mkdir(parents=True, exist_ok=True)
    args.sounio_out.write_text(emit_reducer_source(matrix))

    base.write_json(
        args.manifest,
        {
            "schema": "sounio.semantic_orc.swow16_graph_edge_tile_matrix_reducer_manifest.v1",
            "status": "pass",
            "generator": "scripts/research/semantic_orc_swow16_matrix_reducer_fixture.py",
            "generator_sha256": base.sha256(Path(__file__)),
            "source_matrix_manifest": str(args.matrix_manifest),
            "source_matrix_manifest_sha256": base.sha256(args.matrix_manifest),
            "generated_sounio_source": str(args.sounio_out),
            "generated_sounio_source_sha256": base.sha256(args.sounio_out),
            "fixture_kind": "sounio_native_reducer_over_graph_edge_tile_matrix_manifest",
            "runtime_role": "generated Sounio reducer over manifest invariants; not transport recomputation, not ORC, not inference, not GPU",
            "edge_count": int(matrix["edge_count"]),
            "tile_count_per_edge": int(matrix["tile_count_per_edge"]),
            "total_edge_tile_fixtures": int(matrix["total_edge_tile_fixtures"]),
            "matrix_unique_endpoint_free_node_count": int(matrix["matrix_unique_endpoint_free_node_count"]),
            "matrix_endpoint_free_tile_positions": int(matrix["matrix_endpoint_free_tile_positions"]),
            "endpoint_free_positions_per_fixture": int(int(matrix["matrix_endpoint_free_tile_positions"]) // int(matrix["total_edge_tile_fixtures"])),
            "matrix_record_count": len(matrix.get("records", [])),
            "matrix_tile_record_total": sum(int(record["tile_count"]) for record in matrix.get("records", [])),
            "worst_oracle_max_row_err": float(matrix["worst_oracle_max_row_err"]),
            "worst_oracle_max_col_err": float(matrix["worst_oracle_max_col_err"]),
            "numerical_values_are_engineering_diagnostics_only": True,
            "boundaries": [
                "sounio_reducer_fixture_only",
                "manifest_invariant_check_only",
                "does_not_recompute_transport",
                "no_gpu_runtime_claim",
                "not_exact_wasserstein_distance",
                "no_sinkhorn_convergence_or_error_bound_certificate",
                "not_exact_ollivier_ricci_curvature",
                "not_an_orc_estimator",
                "not_a_biomarker",
                "not_a_clinical_claim",
                "not_statistical_inference",
                "not_population_level_evidence",
                "not_generalizability_evidence",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
