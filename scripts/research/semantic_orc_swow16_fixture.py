#!/usr/bin/env python3
"""Emit a Sounio N=16 semantic entropic-transport fixture from CPC exports.

The generated Sounio source is intentionally temporary: CI scripts should place
it in a scratch directory, run `bin/souc check`, then run it as a runtime gate.
The computation itself stays in Sounio; this script only turns exported CSV rows
into fixed arrays so the current compiler does not need to own CSV parsing yet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input"
)
DEFAULT_TEMPLATE = Path("examples/semantic_orc/sinkhorn_lse_orc.sio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Sounio SWOW/CPC N=16 entropic transport gate."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--regime", default="normative")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


def load_features(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["node_index"]): row for row in csv.DictReader(handle)}


def select_support(path: Path, features: dict[int, dict[str, str]]) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in sorted(
                (key for key in row if key.startswith("step_")),
                key=lambda item: int(item.split("_", 1)[1]),
            ):
                node = int(row[key])
                if node in features and node not in seen:
                    selected.append(node)
                    seen.add(node)
                    if len(selected) == 16:
                        return selected
    raise RuntimeError(f"could not select 16 unique nodes from {path}")


def positive_weights(rows: list[dict[str, str]], columns: list[str]) -> list[float]:
    weights: list[float] = []
    for row in rows:
        value = 0.05
        for column in columns:
            value += abs(float(row[column]))
        weights.append(value)
    total = sum(weights)
    return [value / total for value in weights]


def semantic_cost(rows: list[dict[str, str]]) -> list[float]:
    vectors = [
        (
            float(row["feature_poincare_x"]),
            float(row["feature_poincare_y"]),
            0.5 * float(row["feature_valence"]),
            0.25 * float(row["feature_c_ent"]),
        )
        for row in rows
    ]
    raw: list[float] = []
    max_distance = 0.0
    for left in vectors:
        for right in vectors:
            distance = math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))
            raw.append(distance)
            max_distance = max(max_distance, distance)

    if max_distance <= 0.0:
        raise RuntimeError("degenerate semantic fixture: all feature vectors are equal")

    costs: list[float] = []
    for index, distance in enumerate(raw):
        i = index // 16
        j = index % 16
        if i == j:
            costs.append(0.0)
        else:
            costs.append(0.02 + 0.48 * distance / max_distance)
    return costs


def fmt(value: float) -> str:
    text = f"{value:.12g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def array_literal(values: list[float], width: int) -> str:
    lines: list[str] = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        lines.append("        " + ", ".join(fmt(value) for value in chunk))
    return ",\n".join(lines)


def template_prefix(path: Path) -> str:
    text = path.read_text()
    marker = "fn sinkhorn_lse16_nonuniform"
    if marker not in text:
        raise RuntimeError(f"template marker not found in {path}")
    return text.split(marker, 1)[0].rstrip() + "\n\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    feature_path: Path,
    trajectory_path: Path,
    selected: list[int],
    rows: list[dict[str, str]],
) -> None:
    manifest = {
        "generator": "scripts/research/semantic_orc_swow16_fixture.py",
        "generator_version": "0.1.0",
        "generator_sha256": sha256(Path(__file__)),
        "template": str(args.template),
        "template_sha256": sha256(args.template),
        "input_dir": str(args.input_dir),
        "feature_file": str(feature_path),
        "feature_file_sha256": sha256(feature_path),
        "trajectory_file": str(trajectory_path),
        "trajectory_file_sha256": sha256(trajectory_path),
        "regime": args.regime,
        "selected_node_count": len(selected),
        "selected_node_ids": selected,
        "selected_node_names": [row["node"] for row in rows],
        "measure_mu": "normalized abs(feature_entropy_norm)+abs(feature_kappa)+abs(feature_log_degree)+0.05",
        "measure_nu": "normalized abs(feature_c_ent)+abs(feature_valence)+abs(feature_eta_local)+0.05",
        "cost": "zero diagonal; off-diagonal normalized Euclidean distance over poincare_x,poincare_y,0.5*valence,0.25*c_ent mapped to [0.02,0.50]",
        "claim_limits": [
            "entropic_regularized_transport_only",
            "clinical_claims_not_enforced",
            "no_convergence_theorem",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def emit_source(
    prefix: str,
    selected: list[int],
    rows: list[dict[str, str]],
    mu: list[float],
    nu: list[float],
    cost: list[float],
    regime: str,
) -> str:
    node_names = ", ".join(row["node"] for row in rows)
    node_ids = ", ".join(str(node) for node in selected)
    return (
        prefix
        + f"//! Generated SWOW/CPC fixture regime: {regime}\n"
        + f"//! Generated SWOW/CPC fixture node_ids: {node_ids}\n"
        + f"//! Generated SWOW/CPC fixture node_names: {node_names}\n\n"
        + "fn sinkhorn_lse16_swow_fixture(epsilon: f64, iterations: i64) -> Sinkhorn16Result with Mut, Div, Panic {\n"
        + "    let mu: [f64; 16] = [\n"
        + array_literal(mu, 4)
        + "\n    ]\n"
        + "    let nu: [f64; 16] = [\n"
        + array_literal(nu, 4)
        + "\n    ]\n"
        + "    let cost: [f64; 256] = [\n"
        + array_literal(cost, 4)
        + "\n    ]\n\n"
        + "    var la: [f64; 16] = [0.0; 16]\n"
        + "    var lb: [f64; 16] = [0.0; 16]\n"
        + "    var k: [f64; 256] = [0.0; 256]\n"
        + "    var i: i64 = 0\n"
        + "    while i < 16 {\n"
        + "        la[i as usize] = ln_f64(mu[i as usize])\n"
        + "        lb[i as usize] = ln_f64(nu[i as usize])\n"
        + "        var j: i64 = 0\n"
        + "        while j < 16 {\n"
        + "            let idx = i * 16 + j\n"
        + "            k[idx as usize] = (0.0 - cost[idx as usize]) / epsilon\n"
        + "            j = j + 1\n"
        + "        }\n"
        + "        i = i + 1\n"
        + "    }\n\n"
        + "    var u: [f64; 16] = [0.0; 16]\n"
        + "    var v: [f64; 16] = [0.0; 16]\n"
        + "    var it: i64 = 0\n"
        + "    var converged = false\n"
        + "    while it < iterations && !converged {\n"
        + "        i = 0\n"
        + "        while i < 16 {\n"
        + "            u[i as usize] = la[i as usize] - row_lse16(k, v, i)\n"
        + "            i = i + 1\n"
        + "        }\n"
        + "        var j: i64 = 0\n"
        + "        while j < 16 {\n"
        + "            v[j as usize] = lb[j as usize] - col_lse16(k, u, j)\n"
        + "            j = j + 1\n"
        + "        }\n"
        + "        var iter_fixed_err = 0.0\n"
        + "        i = 0\n"
        + "        while i < 16 {\n"
        + "            let fue = abs_f64(u[i as usize] + row_lse16(k, v, i) - la[i as usize])\n"
        + "            let fve = abs_f64(v[i as usize] + col_lse16(k, u, i) - lb[i as usize])\n"
        + "            if fue > iter_fixed_err { iter_fixed_err = fue }\n"
        + "            if fve > iter_fixed_err { iter_fixed_err = fve }\n"
        + "            i = i + 1\n"
        + "        }\n"
        + "        it = it + 1\n"
        + "        if iter_fixed_err < 0.004 { converged = true }\n"
        + "    }\n\n"
        + "    var mass = 0.0\n"
        + "    var w1 = 0.0\n"
        + "    var row_sum: [f64; 16] = [0.0; 16]\n"
        + "    var col_sum: [f64; 16] = [0.0; 16]\n"
        + "    i = 0\n"
        + "    while i < 16 {\n"
        + "        var j: i64 = 0\n"
        + "        while j < 16 {\n"
        + "            let idx = i * 16 + j\n"
        + "            let pij = exp_f64(u[i as usize] + k[idx as usize] + v[j as usize])\n"
        + "            mass = mass + pij\n"
        + "            w1 = w1 + pij * cost[idx as usize]\n"
        + "            row_sum[i as usize] = row_sum[i as usize] + pij\n"
        + "            col_sum[j as usize] = col_sum[j as usize] + pij\n"
        + "            j = j + 1\n"
        + "        }\n"
        + "        i = i + 1\n"
        + "    }\n\n"
        + "    var max_row_err = 0.0\n"
        + "    var max_col_err = 0.0\n"
        + "    var max_fixed_err = 0.0\n"
        + "    i = 0\n"
        + "    while i < 16 {\n"
        + "        let re = abs_f64(row_sum[i as usize] - mu[i as usize])\n"
        + "        let ce = abs_f64(col_sum[i as usize] - nu[i as usize])\n"
        + "        let fue = abs_f64(u[i as usize] + row_lse16(k, v, i) - la[i as usize])\n"
        + "        let fve = abs_f64(v[i as usize] + col_lse16(k, u, i) - lb[i as usize])\n"
        + "        if re > max_row_err { max_row_err = re }\n"
        + "        if ce > max_col_err { max_col_err = ce }\n"
        + "        if fue > max_fixed_err { max_fixed_err = fue }\n"
        + "        if fve > max_fixed_err { max_fixed_err = fve }\n"
        + "        i = i + 1\n"
        + "    }\n\n"
        + "    Sinkhorn16Result {\n"
        + "        w1: w1,\n"
        + "        mass: mass,\n"
        + "        max_row_err: max_row_err,\n"
        + "        max_col_err: max_col_err,\n"
        + "        max_fixed_err: max_fixed_err,\n"
        + "        iterations_used: it,\n"
        + "    }\n"
        + "}\n\n"
        + "fn main() -> i64 with IO, Epistemic, Mut, Div, Panic {\n"
        + "    let eps_k: Knowledge<f64> = measure(0.05, uncertainty: 0.0005)\n"
        + "    let epsilon = require_confidence(eps_k, 0.98)\n"
        + "    let epsilon_ok = epsilon_in_prototype_range(epsilon)\n"
        + "    let result = sinkhorn_lse16_swow_fixture(epsilon, 160)\n"
        + "    let ok =\n"
        + "        epsilon_ok &&\n"
        + "        approx_eq(result.mass, 1.0, 0.015) &&\n"
        + "        result.max_row_err < 0.012 &&\n"
        + "        result.max_col_err < 0.012 &&\n"
        + "        result.max_fixed_err < 0.004 &&\n"
        + "        result.iterations_used <= 160 &&\n"
        + "        result.w1 > 0.0\n\n"
        + "    if ok {\n"
        + "        println(\"SOUNIO_SWOW16_ENTROPIC_TRANSPORT_PASS\")\n"
        + "        println(\"claim_level=swow16_fixture_runtime_gate\")\n"
        + "        println(\"fixture_source=cpc2026_sounio_input_node_features_parity\")\n"
        + "        println(\"swow16_fixture_nodes=16\")\n"
        + "        println(\"epistemic_epsilon_gate=true\")\n"
        + "        println(\"epsilon_range_gate=true\")\n"
        + "        println(\"swow16_marginal_gate=true\")\n"
        + "        println(\"swow16_fixed_point_gate=true\")\n"
        + "        println(\"swow16_adaptive_stop=true\")\n"
        + "        println(\"entropic_regularized_transport_only=true\")\n"
        + "        println(\"clinical_claims_not_enforced=true\")\n"
        + "        println(\"no_convergence_theorem=true\")\n"
        + "        0\n"
        + "    } else {\n"
        + "        println(\"SOUNIO_SWOW16_ENTROPIC_TRANSPORT_FAIL\")\n"
        + "        1\n"
        + "    }\n"
        + "}\n"
    )


def main() -> int:
    args = parse_args()
    feature_path = args.input_dir / "node_features.csv"
    trajectory_path = args.input_dir / f"trajectories_{args.regime}_nodes_parity.csv"
    features = load_features(feature_path)
    selected = select_support(trajectory_path, features)
    rows = [features[node] for node in selected]
    mu = positive_weights(rows, ["feature_entropy_norm", "feature_kappa", "feature_log_degree"])
    nu = positive_weights(rows, ["feature_c_ent", "feature_valence", "feature_eta_local"])
    cost = semantic_cost(rows)
    source = emit_source(
        template_prefix(args.template),
        selected,
        rows,
        mu,
        nu,
        cost,
        args.regime,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    if args.manifest is not None:
        write_manifest(
            args.manifest,
            args,
            feature_path,
            trajectory_path,
            selected,
            rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
