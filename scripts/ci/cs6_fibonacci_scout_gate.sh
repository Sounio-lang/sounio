#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scout_source="$repo_root/scripts/research/cs6_fibonacci_scout.cpp"
scout_receipt="$repo_root/scripts/research/cs6_fibonacci_scout_receipt_v1.json"
capd_source="$repo_root/scripts/research/cs6_capd_fibonacci_covering.cpp"
aggregate="$repo_root/scripts/research/cs6_capd_fibonacci_covering_aggregate.py"
run_driver="$repo_root/scripts/research/cs6_capd_fibonacci_covering_run.sh"
certificate="$repo_root/scripts/research/cs6_capd_fibonacci_covering_certificate_v1.txt"
note="$repo_root/docs/research/cs6_fibonacci_scout_2026-07-29.md"

# Filled after the mandatory math review. These hashes make receipt drift fail
# loudly instead of silently changing a scientific claim boundary.
expected_scout_source_sha="32e1865c4943b9410dcac70c64df196b92b47e10e745e81b4cd2c6a17a28bc76"
expected_scout_receipt_sha="c259ed750c60ab38fc0b4daeef0f5875d85a84fa78914b94a9ec4d2b7c3fd7ea"
expected_capd_source_sha="8e44540bd122a97adb6a6bf1fb6e9eab16b57c9453ff092dbc18bcb151c50a0b"
expected_aggregate_sha="e1c1dea5374c0f60aff444789fe14eaa2cf04483bc634e993546a65d7a5ce91d"
expected_run_driver_sha="809430236f37bd8b198f2fec81e79cdd40ad6e3e40914a6ac233ab110e331fef"
expected_certificate_sha="a4f7aede0fb017d37167136ae7bda62a8343cbd65e4727d71c4cc604cb07b6aa"
expected_note_sha="c4ca1e4c45eb354f561b3fdeb8cca1f2dbf7a8ea22892c368ab53d9b5b7a0551"

for artifact in \
  "$scout_source" \
  "$scout_receipt" \
  "$capd_source" \
  "$aggregate" \
  "$run_driver" \
  "$certificate" \
  "$note"; do
  test -s "$artifact"
done

test "$(sha256sum "$scout_source" | awk '{print $1}')" = "$expected_scout_source_sha"
test "$(sha256sum "$scout_receipt" | awk '{print $1}')" = "$expected_scout_receipt_sha"
test "$(sha256sum "$capd_source" | awk '{print $1}')" = "$expected_capd_source_sha"
test "$(sha256sum "$aggregate" | awk '{print $1}')" = "$expected_aggregate_sha"
test "$(sha256sum "$run_driver" | awk '{print $1}')" = "$expected_run_driver_sha"
test "$(sha256sum "$certificate" | awk '{print $1}')" = "$expected_certificate_sha"
test "$(sha256sum "$note" | awk '{print $1}')" = "$expected_note_sha"

python3 -m py_compile "$aggregate"
python3 - "$scout_receipt" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
assert receipt["schema"] == "sounio.cs6.fibonacci-scout.v1"
assert receipt["map"]["name"] == "P^6"
assert receipt["map"]["returns_per_map"] == 6
assert receipt["support_u_separation"] > 0
assert receipt["sample_grid"] == [81, 81]
assert len(receipt["sampled_edges"]) == 3
assert all(edge["candidate_pass"] for edge in receipt["sampled_edges"])
assert all(edge["entry_margin"] > 0 for edge in receipt["sampled_edges"])
assert receipt["rigorous_replay_plan"]["ledger_records_expected"] == 42825
assert receipt["rigorous_replay_plan"]["raw_interval_maps_expected"] == 25425
claims = receipt["claims"]
assert claims["numerical_candidate_found"] is True
assert claims["sampled_covering_inequalities_pass"] is True
for key in (
    "rigorous_coverings_proved",
    "positive_entropy_proved",
    "uniform_hyperbolicity_proved",
    "chaotic_attractor_proved",
    "flow_entropy_bound_proved",
):
    assert claims[key] is False, key
assert receipt["u250_in_trusted_computing_base"] is False
print("CS6_FIBONACCI_SCOUT_RECEIPT PASS")
PY

grep -Fxq 'CERTIFICATE_KIND=CAPD_RIGOROUS_COVERING_PREPROMOTION_V1' "$certificate"
grep -Fxq 'STATUS=NOT_RUN' "$certificate"
grep -Fxq 'EXPECTED_RAW_INTERVAL_MAPS=25425' "$certificate"
grep -Fxq 'EXPECTED_LEDGER_RECORDS=42825' "$certificate"
grep -Fxq 'BOUNDED_SAMPLE_EXCEPTIONS=0' "$certificate"
grep -Fxq 'REMOTE_ATTESTATION_PRESENT=false' "$certificate"
grep -Fxq 'INDEPENDENT_REPLAY_REQUIRED=true' "$certificate"
grep -Fxq 'FIBONACCI_COVERINGS_PROVED=false' "$certificate"
grep -Fxq 'POSITIVE_ENTROPY_PROVED=false' "$certificate"
if grep -Eq '^(FIBONACCI_COVERINGS_PROVED|POSITIVE_ENTROPY_PROVED)=true$' "$certificate"; then
  echo "prepromotion receipt contains a promoted claim" >&2
  exit 1
fi

grep -Fq 'fibonacci_coverings_proved = false' "$note"
grep -Fq 'positive_entropy_proved = false' "$note"
grep -Fq 'BLK-20260728-cs6-cluster-ops-auth-bridge' "$note"
grep -Fq 'default Sounio interval path used = false' "$note"
grep -Fq 'rebuilt current-source CAPD path used = true' "$note"
grep -Fq 'legacy numerical reconnaissance kept = true' "$note"

# Exercise the promotion boundary with all 42,825 canonical keys. This checks
# ledger geometry and aggregation mechanics, not the ODE inequalities.
aggregate_test="$(mktemp -d)"
trap 'rm -rf "$aggregate_test"' EXIT
python3 - "$aggregate_test" "$capd_source" <<'PY'
import hashlib
import sys
from decimal import Decimal
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
snapshot = root / "proof-source.cpp"
snapshot.write_bytes(source.read_bytes())
binary = root / "proof-binary"
binary.write_bytes(b"synthetic aggregator fixture; not CAPD evidence\n")
retained = {
    "CAPD_CONFIG_SHA256": (root / "capd-config-retained", b"synthetic config\n"),
    "CAPD_CFLAGS_SHA256": (root / "capd-cflags.txt", b"synthetic cflags"),
    "CAPD_LIBS_SHA256": (root / "capd-libs.txt", b"synthetic libs"),
    "CXX_DRIVER_SHA256": (root / "compiler-driver-retained", b"synthetic compiler\n"),
    "CXX_VERSION_SHA256": (root / "compiler-version.txt", b"synthetic fixture"),
}
retained_hashes = {}
for key, (path, content) in retained.items():
    path.write_bytes(content)
    retained_hashes[key] = hashlib.sha256(content).hexdigest()
source_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
(root / "run-manifest.txt").write_text(
    "\n".join(
        (
            "MANIFEST_KIND=CS6_CAPD_FIBONACCI_RUN_V1",
            "RUN_COMPLETE=true",
            f"SOURCE_SHA256={source_sha}",
            f"EXECUTABLE_SHA256={binary_sha}",
            f"CAPD_CONFIG_SHA256={retained_hashes['CAPD_CONFIG_SHA256']}",
            f"CAPD_CFLAGS_SHA256={retained_hashes['CAPD_CFLAGS_SHA256']}",
            f"CAPD_LIBS_SHA256={retained_hashes['CAPD_LIBS_SHA256']}",
            f"CXX_DRIVER_SHA256={retained_hashes['CXX_DRIVER_SHA256']}",
            f"CXX_VERSION_SHA256={retained_hashes['CXX_VERSION_SHA256']}",
            "CAPD_CONFIG_PATH=/synthetic/capd-config",
            "CXX_PATH=/synthetic/c++",
            "CXX_VERSION=synthetic fixture",
            "SLURM_JOB_ID=synthetic-fixture",
            "EXECUTION_TRUST_MODEL=AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION",
            "REMOTE_ATTESTATION_PRESENT=false",
            "INDEPENDENT_REPLAY_REQUIRED=true",
            "GRID=N0_U:200,N1_U:75,SUPPORT_S:75,EXIT_S:1200",
            "ORDER=8",
            "SHARDS=2",
        )
    )
    + "\n",
    encoding="ascii",
)
edges = (("N0->N0", -1), ("N0->N1", -1), ("N1->N0", 1))
roles = ("support", "left_exit", "right_exit")
preamble = [
    "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0",
    "INTERVAL_BACKEND_DECLARED=FILIB",
    "MAP=P^6",
    "SECTION_ORIENTATION=MinusPlus",
    "ORDER=8",
    "ZSEC=[22.327463739099997, 22.327463739100004]",
    "ORIGIN={[15.186446520640784, 15.186446520640787],[10.908543194765464, 10.908543194765468]}",
    "UNSTABLE={[-0.6743031621419977, -0.67430316214199748],[-0.73845463335624284, -0.73845463335624262]}",
    "STABLE={[-0.94170446778164529, -0.94170446778164507],[0.33644122125579118, 0.33644122125579129]}",
    "FRAME_DETERMINANT=[-0.92226940685332637, -0.9222694068533257]",
    "N0_LOCAL={[0, 0],[0, 0],[0.0039999999999999992, 0.004000000000000001],[0.29999999999999993, 0.30000000000000004]}",
    "N1_LOCAL={[0.019771776972779202, 0.019771776972779209],[0, 0],[0.0014999999999999998, 0.0015000000000000002],[0.29999999999999993, 0.30000000000000004]}",
    "HSETS_DISJOINT=true",
    "FRAME_RIGOROUSLY_INVERTIBLE=true",
    "GRID=N0_U:200,N1_U:75,SUPPORT_S:75,EXIT_S:1200",
]

ledgers = {1: [], 2: []}
counts = {ordinal: {(edge, role): 0 for edge, _ in edges for role in roles} for ordinal in (1, 2)}

def dec(value: Decimal) -> str:
    return format(value, "f")

def add(edge, degree, role, u_index, s_index, ordinal, source_u, source_s):
    if role == "support":
        image_u, image_s = (Decimal(0), Decimal(0)), (Decimal(0), Decimal(0))
    else:
        positive = (degree == 1 and role == "right_exit") or (degree == -1 and role == "left_exit")
        value = Decimal(2) if positive else Decimal(-2)
        image_u, image_s = (value, value), (Decimal(0), Decimal(0))
    ledgers[ordinal].append(
        f"EDGE={edge} ROLE={role} U_INDEX={u_index} S_INDEX={s_index} "
        f"SOURCE_U=[{dec(source_u[0])}, {dec(source_u[1])}] "
        f"SOURCE_S=[{dec(source_s[0])}, {dec(source_s[1])}] "
        f"IMAGE_U=[{dec(image_u[0])}, {dec(image_u[1])}] "
        f"IMAGE_S=[{dec(image_s[0])}, {dec(image_s[1])}] "
        "INITIAL_NORMAL_VELOCITY=[3, 4] NORMAL_VELOCITY=[3, 4] "
        "RETURN_TIME=[1, 2] PHYSICAL_DIAMETER=0.1 MARGIN=1 PASS=true\n"
    )
    counts[ordinal][(edge, role)] += 1

geometry = {
    "N0": (Decimal(0), Decimal("0.004")),
    "N1": (Decimal("0.019771776972779206"), Decimal("0.0015")),
}
for edge, degree in edges:
    source = edge.split("->", 1)[0]
    center, radius = geometry[source]
    u_tiles = 200 if source == "N0" else 75
    u_step = 2 * radius / u_tiles
    s_step = Decimal("0.6") / 75
    for u_index in range(u_tiles):
        u0 = center - radius + u_index * u_step
        for s_index in range(75):
            s0 = Decimal("-0.3") + s_index * s_step
            linear = u_index * 75 + s_index
            add(edge, degree, "support", u_index, s_index, linear % 2 + 1,
                (u0, u0 + u_step), (s0, s0 + s_step))
    face_step = Decimal("0.6") / 1200
    for role, source_u in (("left_exit", center - radius), ("right_exit", center + radius)):
        for s_index in range(1200):
            s0 = Decimal("-0.3") + s_index * face_step
            add(edge, degree, role, 0, s_index, s_index % 2 + 1,
                (source_u, source_u), (s0, s0 + face_step))

for ordinal in (1, 2):
    lines = preamble + [f"SHARD={ordinal}/2", "LEDGER_ENABLED=true"]
    for edge, degree in edges:
        for role in roles:
            count = counts[ordinal][(edge, role)]
            lines.append(
                f"EDGE={edge} DEGREE={degree} ROLE={role} EXPECTED={count} "
                f"PROCESSED={count} PASS={count} MIN_MARGIN=1 "
                "RETURN_TIME=[1,2] MIN_INITIAL_NORMAL_VELOCITY=3 "
                "MIN_NORMAL_VELOCITY=3 MAX_PHYSICAL_DIAMETER=0.1"
            )
    lines.extend((
        "SHARD_PASS=true",
        f"LEDGER_RECORDS={len(ledgers[ordinal])}",
        "FIBONACCI_COVERINGS_PROVED=false",
        "POSITIVE_ENTROPY_PROVED=false",
        "UNIFORM_HYPERBOLICITY_PROVED=false",
        "CHAOTIC_ATTRACTOR_PROVED=false",
        "FLOW_ENTROPY_BOUND_PROVED=false",
    ))
    (root / f"shard-{ordinal}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (root / f"ledger-{ordinal}.txt").write_text("".join(ledgers[ordinal]), encoding="ascii")
PY
python3 "$aggregate" \
  --run-dir "$aggregate_test" \
  --shards 2 \
  --source "$capd_source" \
  --ledger-output "$aggregate_test/ledger-canonical.txt" \
  --certificate-output "$aggregate_test/certificate.txt"
grep -Fxq 'LEDGER_RECORDS=42825' "$aggregate_test/certificate.txt"
grep -Fxq 'FIBONACCI_COVERINGS_PROVED=true' "$aggregate_test/certificate.txt"
grep -Fxq 'POSITIVE_ENTROPY_PROVED=true' "$aggregate_test/certificate.txt"
grep -Fxq 'REMOTE_ATTESTATION_PRESENT=false' "$aggregate_test/certificate.txt"
grep -Fxq 'INDEPENDENT_REPLAY_REQUIRED=true' "$aggregate_test/certificate.txt"
if python3 "$aggregate" \
  --run-dir "$aggregate_test" \
  --shards 2 \
  --source "$capd_source" \
  --ledger-output "$aggregate_test/ledger-canonical.txt" \
  --certificate-output "$aggregate_test/certificate.txt" \
  >"$aggregate_test/existing.out" 2>"$aggregate_test/existing.err"; then
  echo "aggregator overwrote an existing promoted output" >&2
  exit 1
fi
grep -Fq 'refusing existing output' "$aggregate_test/existing.err"
python3 - "$aggregate_test/ledger-1.txt" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="ascii")
needle = "EDGE=N0->N0 ROLE=support U_INDEX=0 S_INDEX=0 "
lines = text.splitlines(keepends=True)
matches = [index for index, line in enumerate(lines) if line.startswith(needle)]
assert matches == [0], matches
lines[0], substitutions = re.subn(
    r"SOURCE_U=\[[^]]+\]", "SOURCE_U=[0, 0]", lines[0], count=1
)
assert substitutions == 1
path.write_text("".join(lines), encoding="ascii")
PY
if python3 "$aggregate" \
  --run-dir "$aggregate_test" \
  --shards 2 \
  --source "$capd_source" \
  --ledger-output "$aggregate_test/should-fail-ledger.txt" \
  --certificate-output "$aggregate_test/should-fail.txt" \
  >"$aggregate_test/negative.out" 2>"$aggregate_test/negative.err"; then
  echo "aggregator accepted a relabelled source box" >&2
  exit 1
fi
grep -Fq 'SOURCE_U does not enclose its canonical tile' \
  "$aggregate_test/negative.err"
echo "CS6_FIBONACCI_AGGREGATOR_BOUNDARY PASS"

if [[ "${CS6_CAPD_SAMPLE_REPLAY:-0}" == "1" ]]; then
  capd_config="${CS6_CAPD_CONFIG:-capd-config}"
  if ! command -v "$capd_config" >/dev/null 2>&1; then
    echo "CS6_CAPD_SAMPLE_REPLAY REFUSED: capd-config is unavailable" >&2
    exit 3
  fi
  # capd-config intentionally emits the compiler and linker arguments.
  # shellcheck disable=SC2046
  "${CXX:-c++}" -std=c++17 -O2 "$capd_source" \
    $("$capd_config" --cflags --libs) \
    -o "$aggregate_test/cs6_capd"
  # shellcheck disable=SC2046
  "${CXX:-c++}" -std=c++17 -O3 "$scout_source" \
    $("$capd_config" --cflags --libs) \
    -o "$aggregate_test/cs6_scout"
  "$aggregate_test/cs6_scout" candidate 6 81 0.004 0.3 0.0015 0.3 \
    > "$aggregate_test/scout.txt"
  test "$(grep -c 'CANDIDATE_PASS=true' "$aggregate_test/scout.txt")" -eq 3
  "$aggregate_test/cs6_capd" 200 75 75 1200 8 1 1000 \
    "$aggregate_test/sample-ledger.txt" > "$aggregate_test/sample-shard.txt"
  grep -Fxq 'SHARD_PASS=true' "$aggregate_test/sample-shard.txt"
  grep -Fxq 'FIBONACCI_COVERINGS_PROVED=false' "$aggregate_test/sample-shard.txt"
  grep -Fxq 'LEDGER_RECORDS=48' "$aggregate_test/sample-shard.txt"
  echo "CS6_CAPD_SAMPLE_REPLAY PASS"
fi

# The inherited UPO certificate is a prerequisite for any promotion output.
bash "$repo_root/scripts/ci/cs6_proof_machine_gate.sh"

if [[ -n "${CS6_CAPD_AGGREGATE_RUN_DIR:-}" ]]; then
  test -n "${CS6_CAPD_SHARDS:-}"
  test -n "${CS6_CAPD_AGGREGATE_OUTPUT_DIR:-}"
  if [[ -e "$CS6_CAPD_AGGREGATE_OUTPUT_DIR" ]]; then
    echo "refusing existing aggregate output directory" >&2
    exit 2
  fi
  mkdir "$CS6_CAPD_AGGREGATE_OUTPUT_DIR"
  python3 "$aggregate" \
    --run-dir "$CS6_CAPD_AGGREGATE_RUN_DIR" \
    --shards "$CS6_CAPD_SHARDS" \
    --source "$capd_source" \
    --ledger-output "$CS6_CAPD_AGGREGATE_OUTPUT_DIR/ledger-canonical.txt" \
    --certificate-output "$CS6_CAPD_AGGREGATE_OUTPUT_DIR/certificate.txt"
  grep -Fxq 'FIBONACCI_COVERINGS_PROVED=true' "$CS6_CAPD_AGGREGATE_OUTPUT_DIR/certificate.txt"
  echo "CS6_CAPD_FULL_AGGREGATE PASS artifact=$CS6_CAPD_AGGREGATE_OUTPUT_DIR/certificate.txt"
fi

echo "CS6_FIBONACCI_SCOUT_GATE PASS"
