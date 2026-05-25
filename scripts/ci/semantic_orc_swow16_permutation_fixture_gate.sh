#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
SUPPORT_COUNT="${SOUNIO_SEMANTIC_ORC_SUPPORT_COUNT:-4}"
SUPPORT_STRIDE="${SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE:-16}"
PERMUTATION_COUNT="${SOUNIO_SEMANTIC_ORC_PERMUTATION_COUNT:-64}"
REGIMES="${SOUNIO_SEMANTIC_ORC_REGIMES:-normative anxious ruminative psychotic}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_PERMUTATION_FIXTURE_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-permutation-fixture.XXXXXX)}"
MANIFEST="$OUT_DIR/swow16_permutation_fixture_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-permutation-fixture] input=$INPUT_DIR"
echo "[semantic-orc-permutation-fixture] regimes=$REGIMES"
echo "[semantic-orc-permutation-fixture] support_count=$SUPPORT_COUNT"
echo "[semantic-orc-permutation-fixture] support_stride=$SUPPORT_STRIDE"
echo "[semantic-orc-permutation-fixture] permutation_count=$PERMUTATION_COUNT"
echo "[semantic-orc-permutation-fixture] out=$OUT_DIR"

manifest_paths=()
pack_paths=()

for regime in $REGIMES; do
  support_index=0
  while [[ "$support_index" -lt "$SUPPORT_COUNT" ]]; do
    support_dir="$OUT_DIR/${regime}/support_${support_index}"
    mkdir -p "$support_dir"
    SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
    SOUNIO_SEMANTIC_ORC_REGIME="$regime" \
    SOUNIO_SEMANTIC_ORC_SUPPORT_INDEX="$support_index" \
    SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE="$SUPPORT_STRIDE" \
    SOUNIO_SEMANTIC_ORC_KAXI_PACK_DIR="$support_dir" \
      bash scripts/ci/semantic_orc_swow16_kaxi_pack_gate.sh
    manifest_paths+=("$support_dir/swow16_fixture_manifest.json")
    pack_paths+=("$support_dir/swow16_kaxi_pack.json")
    support_index=$((support_index + 1))
  done
done

python3 - "$MANIFEST" "$SUPPORT_COUNT" "$SUPPORT_STRIDE" "$PERMUTATION_COUNT" "$REGIMES" "${manifest_paths[@]}" -- "${pack_paths[@]}" <<'PY'
import hashlib
import csv
import json
import math
import pathlib
import random
import sys
from collections import defaultdict

out_path = pathlib.Path(sys.argv[1])
support_count = int(sys.argv[2])
support_stride = int(sys.argv[3])
permutation_count = int(sys.argv[4])
expected_regimes = tuple(sys.argv[5].split())
sep = sys.argv.index("--")
manifest_paths = [pathlib.Path(value) for value in sys.argv[6:sep]]
pack_paths = [pathlib.Path(value) for value in sys.argv[sep + 1 :]]

N = 16
SINKHORN_ITERATIONS = 16
MASS_TOLERANCE = 0.02
MARGINAL_TOLERANCE = 0.015
FEATURE_CACHE = {}


def seed_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "big")


def log2sumexp(values):
    # K-AXI sinkhorn16 stores la/lb/K in base-2 log domain; keep this fixture
    # in the same numeric domain so it compares directly with the pack oracle.
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("log2sumexp received a non-finite value")
    m = max(values)
    if not math.isfinite(m):
        raise SystemExit("log2sumexp has no finite maximum")
    return m + math.log2(sum(2.0 ** (value - m) for value in values))


def sinkhorn_duals(la, lb, k_values, iterations=SINKHORN_ITERATIONS):
    u = [0.0] * N
    v = [0.0] * N
    for _ in range(iterations):
        for i in range(N):
            u[i] = la[i] - log2sumexp([k_values[i * N + j] + v[j] for j in range(N)])
        for j in range(N):
            v[j] = lb[j] - log2sumexp([k_values[i * N + j] + u[i] for i in range(N)])
        if not all(math.isfinite(value) for value in u + v):
            raise SystemExit("Sinkhorn dual update produced a non-finite value")
    return u, v


def transport_cost(mu, nu, la, lb, k_values, cost, iterations=SINKHORN_ITERATIONS):
    u, v = sinkhorn_duals(la, lb, k_values, iterations=iterations)
    row = [0.0] * N
    col = [0.0] * N
    mass = 0.0
    transport = 0.0
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            pij = 2.0 ** (u[i] + k_values[idx] + v[j])
            mass += pij
            row[i] += pij
            col[j] += pij
            transport += pij * cost[idx]
    return {
        "transport_cost": transport,
        "mass": mass,
        "max_row_err": max(abs(row[i] - mu[i]) for i in range(N)),
        "max_col_err": max(abs(col[i] - nu[i]) for i in range(N)),
    }


def load_features(path):
    key = str(path)
    if key not in FEATURE_CACHE:
        with pathlib.Path(path).open(newline="") as handle:
            FEATURE_CACHE[key] = {
                int(row["node_index"]): row for row in csv.DictReader(handle)
            }
    return FEATURE_CACHE[key]


def degree_quartile_strata(feature_file, selected_node_ids):
    features = load_features(feature_file)
    ranked = sorted(
        range(N),
        key=lambda local: (
            float(features[selected_node_ids[local]]["feature_log_degree"]),
            selected_node_ids[local],
        ),
    )
    return [sorted(ranked[start : start + 4]) for start in range(0, N, 4)]


def permutation_family_strata(family, manifest, selected_node_ids):
    if family == "global_target_permutation":
        return [list(range(N))]
    if family == "trajectory_block_target_permutation":
        return [list(range(start, start + 4)) for start in range(0, N, 4)]
    if family == "exported_log_degree_quartile_target_permutation":
        return degree_quartile_strata(manifest["feature_file"], selected_node_ids)
    raise SystemExit(f"unknown permutation fixture family: {family}")


def deterministic_permutation(regime, support_index, permutation_index, family, strata):
    permutation = list(range(N))
    for stratum_index, stratum in enumerate(strata):
        shuffled = list(stratum)
        rng = random.Random(
            seed_int(
                f"sounio-semantic-orc-permutation-fixture-v1:{family}:{regime}:{support_index}:{permutation_index}:{stratum_index}"
            )
        )
        rng.shuffle(shuffled)
        for target, source in zip(stratum, shuffled):
            permutation[target] = source
    return permutation


def mean(values):
    return sum(values) / len(values)


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) * (value - m) for value in values) / (len(values) - 1))


def evaluate_permutation_family(family, manifest, selected, regime, support_index, permutation_count, mu, nu, la, k_values, cost):
    strata = permutation_family_strata(family, manifest, selected)
    permutation_records = []
    permutation_costs = []
    for permutation_index in range(permutation_count):
        perm = deterministic_permutation(regime, support_index, permutation_index, family, strata)
        nu_perm = [nu[perm[j]] for j in range(N)]
        lb_perm = [math.log2(value) for value in nu_perm]
        permutation_result = transport_cost(mu, nu_perm, la, lb_perm, k_values, cost)
        if abs(permutation_result["mass"] - 1.0) >= MASS_TOLERANCE:
            raise SystemExit(
                f"permutation fixture mass tolerance failure family={family} regime={regime} support={support_index} permutation={permutation_index}"
            )
        if permutation_result["max_row_err"] >= MARGINAL_TOLERANCE or permutation_result["max_col_err"] >= MARGINAL_TOLERANCE:
            raise SystemExit(
                f"permutation fixture marginal tolerance failure family={family} regime={regime} support={support_index} permutation={permutation_index}"
            )
        permutation_costs.append(permutation_result["transport_cost"])
        permutation_records.append(
            {
                "permutation_index": permutation_index,
                "target_permutation": perm,
                "transport_cost": permutation_result["transport_cost"],
                "mass": permutation_result["mass"],
                "max_row_err": permutation_result["max_row_err"],
                "max_col_err": permutation_result["max_col_err"],
            }
        )
    return {
        "type": family,
        "permutation_count": permutation_count,
        "strata": strata,
        "transport_cost_mean": mean(permutation_costs),
        "transport_cost_sd": stdev(permutation_costs),
        "transport_cost_min": min(permutation_costs),
        "transport_cost_max": max(permutation_costs),
        "records": permutation_records,
    }


if not expected_regimes:
    raise SystemExit("REGIMES must not be empty")
if permutation_count <= 0:
    raise SystemExit("PERMUTATION_COUNT must be positive")
if len(manifest_paths) != len(pack_paths):
    raise SystemExit("manifest/pack path count mismatch")
expected_total = len(expected_regimes) * support_count
if len(manifest_paths) != expected_total:
    raise SystemExit(f"expected {expected_total} support packs, got {len(manifest_paths)}")

records = []
seen_keys = set()
nodes_by_regime = defaultdict(set)
regime_costs = defaultdict(list)

for manifest_path, pack_path in zip(manifest_paths, pack_paths):
    manifest = json.loads(manifest_path.read_text())
    pack = json.loads(pack_path.read_text())
    if manifest.get("schema") != "sounio.semantic_orc.swow16_fixture_manifest.v1":
        raise SystemExit(f"unexpected fixture manifest schema in {manifest_path}")
    if pack.get("schema") != "sounio.semantic_orc.swow16_kaxi_sinkhorn16_pack.v1":
        raise SystemExit(f"unexpected K-AXI pack schema in {pack_path}")
    regime = str(pack.get("regime", ""))
    support_index = int(pack.get("support_index", -1))
    key = (regime, support_index)
    if key in seen_keys:
        raise SystemExit(f"duplicate regime/support: {regime}/{support_index}")
    seen_keys.add(key)
    if regime not in expected_regimes:
        raise SystemExit(f"unexpected regime: {regime}")
    if support_stride != int(pack.get("support_stride", -1)):
        raise SystemExit(f"support_stride mismatch in {pack_path}")

    selected = [int(value) for value in pack.get("selected_node_ids", [])]
    if len(selected) != N or len(set(selected)) != N:
        raise SystemExit(f"support must contain {N} unique nodes in {pack_path}")
    overlap = nodes_by_regime[regime].intersection(selected)
    if overlap:
        raise SystemExit(f"support overlap in {regime} support {support_index}: {sorted(overlap)}")
    nodes_by_regime[regime].update(selected)

    la = [float(value) for value in pack["la_log2"]]
    lb = [float(value) for value in pack["lb_log2"]]
    k_values = [float(value) for value in pack["k_log2"]]
    cost = [float(value) for value in pack["cost"]]
    if len(la) != N or len(lb) != N or len(k_values) != N * N or len(cost) != N * N:
        raise SystemExit(f"pack shape mismatch in {pack_path}")
    mu = [2.0 ** value for value in la]
    nu = [2.0 ** value for value in lb]
    observed = transport_cost(mu, nu, la, lb, k_values, cost)
    oracle_transport = float((pack.get("oracle") or {}).get("transport_cost", observed["transport_cost"]))
    if abs(observed["transport_cost"] - oracle_transport) > 1e-9:
        raise SystemExit(f"observed transport mismatch in {pack_path}")
    if abs(observed["mass"] - 1.0) >= MASS_TOLERANCE:
        raise SystemExit(f"observed mass tolerance failure in {pack_path}")
    if observed["max_row_err"] >= MARGINAL_TOLERANCE or observed["max_col_err"] >= MARGINAL_TOLERANCE:
        raise SystemExit(f"observed marginal tolerance failure in {pack_path}")

    observed_cost = observed["transport_cost"]
    regime_costs[regime].append(observed_cost)
    permutation_models = {
        family: evaluate_permutation_family(
            family,
            manifest,
            selected,
            regime,
            support_index,
            permutation_count,
            mu,
            nu,
            la,
            k_values,
            cost,
        )
        for family in (
            "global_target_permutation",
            "trajectory_block_target_permutation",
            "exported_log_degree_quartile_target_permutation",
        )
    }
    records.append(
        {
            "regime": regime,
            "support_index": support_index,
            "support_offset": int(pack.get("support_offset", -1)),
            "selected_node_ids": selected,
            "selected_node_names": pack.get("selected_node_names", []),
            "fixture_manifest": str(manifest_path),
            "kaxi_pack": str(pack_path),
            "observed": observed,
            "permutation_models": permutation_models,
        }
    )

missing = [
    (regime, support_index)
    for regime in expected_regimes
    for support_index in range(support_count)
    if (regime, support_index) not in seen_keys
]
if missing:
    raise SystemExit(f"missing regime/support records: {missing}")

support_fixture_constraint = {
    "support_size": N,
    "constraint_role": "equal-support-size fixture constraint only; not statistical Type I error control, not sample-size calculation, not power analysis, and not confounding-control design",
    "support_count_per_regime": support_count,
    "support_stride": support_stride,
    "support_stride_role": "generation parameter and consistency check for contiguous support windows",
    "equal_support_count_across_regimes": True,
    "non_overlapping_within_regime_only": True,
    "observed_nodes_per_regime": {
        regime: len(nodes_by_regime[regime]) for regime in expected_regimes
    },
    "observed_transport_cost_range_by_regime": {
        regime: {
            "min": min(regime_costs[regime]),
            "max": max(regime_costs[regime]),
            "mean": mean(regime_costs[regime]),
        }
        for regime in expected_regimes
    },
}

records.sort(key=lambda item: (item["regime"], item["support_index"]))
out_path.write_text(
    json.dumps(
            {
                "schema": "sounio.semantic_orc.swow16_permutation_fixture_manifest.v1",
                "status": "pass",
                "clinical_warning": "This is not a biomarker, diagnostic, clinical utility, treatment-response, or patient-level prediction artifact.",
                "fixture_count": len(records),
                "permutation_count_per_fixture": permutation_count,
                "permutation_fixture_families": [
                    "global_target_permutation",
                    "trajectory_block_target_permutation",
                    "exported_log_degree_quartile_target_permutation",
                ],
                "sinkhorn_iterations": SINKHORN_ITERATIONS,
                "mass_tolerance": MASS_TOLERANCE,
                "marginal_tolerance": MARGINAL_TOLERANCE,
                "permutation_count_role": "smoke-scale deterministic permutation fixture draws only; not sized for inference, sample-size calculation, or power analysis",
                "numeric_domain": "base-2 log-domain Sinkhorn to match the K-AXI sinkhorn16 pack contract",
                "unit_convention": "k_log2 stores the base-2 log kernel; transport_cost is the expected original semantic cost under the induced plan",
                "support_fixture_constraint": support_fixture_constraint,
            "records": records,
            "boundaries": [
                "cpu_pack_oracle_fixture_only",
                "no_gpu_runtime_claim",
                "deterministic_permutation_fixture_families_descriptive_only",
                "not_a_significance_test",
                "no_multiple_testing_claim",
                "not_sample_size_calculation",
                "not_power_analysis",
                "not_confounding_control_design",
                "entropic_regularized_transport_only",
                "clinical_claims_not_enforced",
                "biomarker_claims_not_enforced",
                "no_convergence_theorem",
                "not_a_clinical_biomarker_validation",
            ],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"semantic_orc_swow16_permutation_fixture_gate: PASS artifact={out_path}")
PY
