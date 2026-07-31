#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_capd_c1_cone.cpp"
runner="$repo_root/scripts/research/cs6_capd_c1_cone_run.sh"
aggregate="$repo_root/scripts/research/cs6_capd_c1_cone_aggregate.py"
certificate="$repo_root/scripts/research/cs6_capd_c1_cone_certificate_v1.txt"

for path in "$source_file" "$runner" "$aggregate" "$certificate"; do
  test -f "$path"
done

grep -Fxq 'STATUS=NOT_RUN' "$certificate"
grep -Fxq 'C0_GRID_VALID_AS_C1_DEFAULT=false' "$certificate"
grep -Fxq 'C0_RAW_EVIDENCE_REAGGREGATION_REQUIRED=true' "$certificate"
grep -Fxq 'C0_EXECUTION_TRUST_MODEL_TREATED_AS_DECLARED_METADATA=true' "$certificate"
grep -Fxq 'C0_EXECUTION_PROVENANCE_VERIFIED=false' "$certificate"
grep -Fxq 'C0_C1_SEMANTIC_CONTRACT_BOUND=true' "$certificate"
grep -Fxq 'C1_EXPLICIT_CAPD_ARTIFACT_SETS_BOUND=true' "$certificate"
grep -Fxq 'C1_LIOUVILLE_FINAL_RETURN_OVERLAP_REQUIRED=true' "$certificate"
grep -Fxq 'LIOUVILLE_EXPONENTIAL_OPERAND_EMITTED=true' "$certificate"
grep -Fxq 'LIOUVILLE_EXPONENTIATION_RECOMPUTED_BY_AGGREGATOR=false' "$certificate"
grep -Fxq 'C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_REQUIRED=true' "$certificate"
grep -Fxq 'LIVE_WORKER_AGGREGATOR_LEDGER_ROUNDTRIP=true' "$certificate"
grep -Fxq 'CONE_DETERMINANT_FORM=EXPANDED_EXACT_CANCELLATION_BEFORE_INTERVAL_EVALUATION' "$certificate"
grep -Fxq 'BOUNDED_FINE_PROBE_EDGE=N0->N0' "$certificate"
grep -Fxq 'BOUNDED_FINE_PROBE_DET_M_EXPANDED_LOWER=5.2100640574428869' "$certificate"
grep -Fxq 'FULL_SOURCE_C1_DERIVATIVE_ENCLOSURE_PROVED=false' "$certificate"
grep -Fxq 'GLOBAL_FULL_SOURCE_HULL_TESTED=false' "$certificate"
grep -Fxq 'PAIRWISE_CHORD_CONE_CONDITION_PROVED=false' "$certificate"
grep -Fxq 'LIOUVILLE_INVERTIBILITY_PROVED=false' "$certificate"
grep -Fxq 'COMBINED_C0_C1_MATHEMATICAL_EVIDENCE_COMPLETE=false' "$certificate"
grep -Fxq 'COMBINED_C0_C1_EXECUTION_PROVENANCE_ATTESTED=false' "$certificate"
grep -Fxq 'UNIFORM_HYPERBOLICITY_PROVED=false' "$certificate"
grep -Fxq 'CHAOTIC_ATTRACTOR_PROVED=false' "$certificate"
python3 -m py_compile "$aggregate"
bash -n "$runner"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fixture="$work/fixture"
mkdir -p "$fixture"
python3 - "$source_file" "$fixture" <<'PY'
import hashlib
import math
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

source = Path(sys.argv[1])
root = Path(sys.argv[2])
c0_source = source.with_name("cs6_capd_fibonacci_covering.cpp")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def outward_endpoint(value: Fraction, direction: float) -> str:
    result = float(value)
    exact = Fraction.from_float(result)
    if direction < 0:
        while exact > value:
            result = math.nextafter(result, -math.inf)
            exact = Fraction.from_float(result)
    else:
        while exact < value:
            result = math.nextafter(result, math.inf)
            exact = Fraction.from_float(result)
    return math.nextafter(result, direction).hex()

def interval(lower, upper=None) -> str:
    lower = Fraction(lower)
    upper = lower if upper is None else Fraction(upper)
    return f"[{outward_endpoint(lower, -math.inf)},{outward_endpoint(upper, math.inf)}]"

snapshot = root / "proof-source.cpp"
snapshot.write_bytes(source.read_bytes())
(root / "proof-binary").write_bytes(b"synthetic-c1-proof-binary\n")
artifact_manifest = (
    f"{sha(root / 'proof-binary')}  {root / 'proof-binary'}\n"
).encode("ascii")
include_root = root / "synthetic-include"
include_root.mkdir()
(include_root / "synthetic.hpp").write_bytes(b"// synthetic CAPD header\n")
header_manifest = (
    f"{sha(include_root / 'synthetic.hpp')}  {include_root / 'synthetic.hpp'}\n"
).encode("ascii")
retained = {
    "capd-config-retained": b"synthetic-capd-config\n",
    "capd-cflags.txt": f"-I{include_root} -D__USE_FILIB__ -frounding-math\n".encode("ascii"),
    "capd-libs.txt": f"{root / 'proof-binary'}\n".encode("ascii"),
    "compiler-driver-retained": b"synthetic-cxx\n",
    "compiler-version.txt": b"synthetic-cxx-version\n",
    "capd-version.txt": b"5.3.0\n",
    "capd.pc": b"Name: capd\nVersion: 5.3.0\n",
    "capd-libraries.sha256": artifact_manifest,
    "capd-headers.sha256": header_manifest,
    "runtime-linkage.txt": f"{root / 'proof-binary'} (0x0)\n".encode("ascii"),
    "runtime-libraries.sha256": artifact_manifest,
    "slurm-job.txt": b"JobId=synthetic-1 UserId=synthetic(1000) JobState=RUNNING NodeList=synthetic-node\n",
    "slurm-version.txt": b"slurm synthetic\n",
    "slurm-hostnames.txt": b"synthetic-node\n",
}
for name, content in retained.items():
    (root / name).write_bytes(content)

c0_aggregator = source.with_name("cs6_capd_fibonacci_covering_aggregate.py")
(root / "c0-aggregator.py").write_bytes(c0_aggregator.read_bytes())
c0_run = root / "c0-run"
c0_run.mkdir()
c0_snapshot = c0_run / "proof-source.cpp"
c0_snapshot.write_bytes(c0_source.read_bytes())
(c0_run / "proof-binary").write_bytes(b"synthetic C0 boundary fixture\n")
c0_retained = {
    "CAPD_CONFIG_SHA256": (c0_run / "capd-config-retained", b"synthetic config\n"),
    "CAPD_CFLAGS_SHA256": (c0_run / "capd-cflags.txt", b"synthetic cflags"),
    "CAPD_LIBS_SHA256": (c0_run / "capd-libs.txt", b"synthetic libs"),
    "CXX_DRIVER_SHA256": (c0_run / "compiler-driver-retained", b"synthetic compiler\n"),
    "CXX_VERSION_SHA256": (c0_run / "compiler-version.txt", b"synthetic fixture"),
}
c0_retained_hashes = {}
for key, (path, content) in c0_retained.items():
    path.write_bytes(content)
    c0_retained_hashes[key] = hashlib.sha256(content).hexdigest()
(c0_run / "run-manifest.txt").write_text(
    "\n".join(
        (
            "MANIFEST_KIND=CS6_CAPD_FIBONACCI_RUN_V1",
            "RUN_COMPLETE=true",
            f"SOURCE_SHA256={sha(c0_snapshot)}",
            f"EXECUTABLE_SHA256={sha(c0_run / 'proof-binary')}",
            f"CAPD_CONFIG_SHA256={c0_retained_hashes['CAPD_CONFIG_SHA256']}",
            f"CAPD_CFLAGS_SHA256={c0_retained_hashes['CAPD_CFLAGS_SHA256']}",
            f"CAPD_LIBS_SHA256={c0_retained_hashes['CAPD_LIBS_SHA256']}",
            f"CXX_DRIVER_SHA256={c0_retained_hashes['CXX_DRIVER_SHA256']}",
            f"CXX_VERSION_SHA256={c0_retained_hashes['CXX_VERSION_SHA256']}",
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
    ) + "\n",
    encoding="ascii",
)
c0_edges = (("N0->N0", -1), ("N0->N1", -1), ("N1->N0", 1))
c0_roles = ("support", "left_exit", "right_exit")
c0_preamble = [
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
c0_ledgers = {1: [], 2: []}
c0_counts = {
    ordinal: {
        (edge, role): 0
        for edge, _ in c0_edges
        for role in c0_roles
    }
    for ordinal in (1, 2)
}

def decimal_text(value: Decimal) -> str:
    return format(value, "f")

def add_c0(edge, degree, role, u_index, s_index, ordinal, source_u, source_s):
    if role == "support":
        image_u, image_s = (Decimal(0), Decimal(0)), (Decimal(0), Decimal(0))
    else:
        positive = ((degree == 1 and role == "right_exit") or
                    (degree == -1 and role == "left_exit"))
        value = Decimal(2) if positive else Decimal(-2)
        image_u, image_s = (value, value), (Decimal(0), Decimal(0))
    c0_ledgers[ordinal].append(
        f"EDGE={edge} ROLE={role} U_INDEX={u_index} S_INDEX={s_index} "
        f"SOURCE_U=[{decimal_text(source_u[0])}, {decimal_text(source_u[1])}] "
        f"SOURCE_S=[{decimal_text(source_s[0])}, {decimal_text(source_s[1])}] "
        f"IMAGE_U=[{decimal_text(image_u[0])}, {decimal_text(image_u[1])}] "
        f"IMAGE_S=[{decimal_text(image_s[0])}, {decimal_text(image_s[1])}] "
        "INITIAL_NORMAL_VELOCITY=[3, 4] NORMAL_VELOCITY=[3, 4] "
        "RETURN_TIME=[1, 2] PHYSICAL_DIAMETER=0.1 MARGIN=1 PASS=true\n"
    )
    c0_counts[ordinal][(edge, role)] += 1

c0_geometry = {
    "N0": (Decimal(0), Decimal("0.004")),
    "N1": (Decimal("0.019771776972779206"), Decimal("0.0015")),
}
for edge, degree in c0_edges:
    source_name = edge.split("->", 1)[0]
    center, radius = c0_geometry[source_name]
    u_tiles = 200 if source_name == "N0" else 75
    u_step = 2 * radius / u_tiles
    s_step = Decimal("0.6") / 75
    for u_index in range(u_tiles):
        u0 = center - radius + u_index * u_step
        for s_index in range(75):
            s0 = Decimal("-0.3") + s_index * s_step
            linear = u_index * 75 + s_index
            add_c0(
                edge, degree, "support", u_index, s_index, linear % 2 + 1,
                (u0, u0 + u_step), (s0, s0 + s_step),
            )
    face_step = Decimal("0.6") / 1200
    for role, source_u in (
        ("left_exit", center - radius), ("right_exit", center + radius)
    ):
        for s_index in range(1200):
            s0 = Decimal("-0.3") + s_index * face_step
            add_c0(
                edge, degree, role, 0, s_index, s_index % 2 + 1,
                (source_u, source_u), (s0, s0 + face_step),
            )

for ordinal in (1, 2):
    lines = c0_preamble + [f"SHARD={ordinal}/2", "LEDGER_ENABLED=true"]
    for edge, degree in c0_edges:
        for role in c0_roles:
            count = c0_counts[ordinal][(edge, role)]
            lines.append(
                f"EDGE={edge} DEGREE={degree} ROLE={role} EXPECTED={count} "
                f"PROCESSED={count} PASS={count} MIN_MARGIN=1 "
                "RETURN_TIME=[1,2] MIN_INITIAL_NORMAL_VELOCITY=3 "
                "MIN_NORMAL_VELOCITY=3 MAX_PHYSICAL_DIAMETER=0.1"
            )
    lines.extend((
        "SHARD_PASS=true",
        f"LEDGER_RECORDS={len(c0_ledgers[ordinal])}",
        "FIBONACCI_COVERINGS_PROVED=false",
        "POSITIVE_ENTROPY_PROVED=false",
        "UNIFORM_HYPERBOLICITY_PROVED=false",
        "CHAOTIC_ATTRACTOR_PROVED=false",
        "FLOW_ENTROPY_BOUND_PROVED=false",
    ))
    (c0_run / f"shard-{ordinal}.txt").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )
    (c0_run / f"ledger-{ordinal}.txt").write_text(
        "".join(c0_ledgers[ordinal]), encoding="ascii"
    )

c0_output = root / "c0-aggregate"
c0_output.mkdir()
subprocess.run(
    (
        sys.executable, str(c0_aggregator),
        "--run-dir", str(c0_run), "--shards", "2",
        "--source", str(c0_source),
        "--ledger-output", str(c0_output / "ledger.txt"),
        "--certificate-output", str(c0_output / "certificate.txt"),
    ),
    check=True,
)
c0 = root / "c0-certificate.txt"
c0.write_bytes((c0_output / "certificate.txt").read_bytes())

manifest = (
    "MANIFEST_KIND=CS6_CAPD_C1_CONE_RUN_V1\n"
    "RUN_COMPLETE=true\n"
    f"SOURCE_SHA256={sha(snapshot)}\n"
    f"EXECUTABLE_SHA256={sha(root / 'proof-binary')}\n"
    f"C0_CERTIFICATE_SHA256={sha(c0)}\n"
    f"CAPD_CONFIG_SHA256={sha(root / 'capd-config-retained')}\n"
    f"CAPD_CFLAGS_SHA256={sha(root / 'capd-cflags.txt')}\n"
    f"CAPD_LIBS_SHA256={sha(root / 'capd-libs.txt')}\n"
    f"CXX_DRIVER_SHA256={sha(root / 'compiler-driver-retained')}\n"
    f"CXX_VERSION_SHA256={sha(root / 'compiler-version.txt')}\n"
    f"CAPD_VERSION_SHA256={sha(root / 'capd-version.txt')}\n"
    f"CAPD_PC_SHA256={sha(root / 'capd.pc')}\n"
    f"CAPD_LIBRARY_MANIFEST_SHA256={sha(root / 'capd-libraries.sha256')}\n"
    f"CAPD_HEADER_MANIFEST_SHA256={sha(root / 'capd-headers.sha256')}\n"
    f"RUNTIME_LINKAGE_SHA256={sha(root / 'runtime-linkage.txt')}\n"
    f"RUNTIME_LIBRARY_MANIFEST_SHA256={sha(root / 'runtime-libraries.sha256')}\n"
    f"SLURM_JOB_RECORD_SHA256={sha(root / 'slurm-job.txt')}\n"
    f"SLURM_VERSION_SHA256={sha(root / 'slurm-version.txt')}\n"
    f"SLURM_HOSTNAMES_SHA256={sha(root / 'slurm-hostnames.txt')}\n"
    f"C0_AGGREGATOR_SHA256={sha(root / 'c0-aggregator.py')}\n"
    "CAPD_CONFIG_PATH=/synthetic/capd-config\n"
    "CXX_PATH=/synthetic/c++\n"
    "CXX_VERSION=synthetic-cxx-version\n"
    "SLURM_JOB_ID=synthetic-1\n"
    "SLURM_NODELIST=synthetic-node\n"
    "EXECUTION_NODE=synthetic-node\n"
    "EXECUTION_UID=1000\n"
    "EXECUTION_TRUST_MODEL=SAME_UID_ACTIVE_SLURM_ALLOCATION_INCLUDES_EXECUTION_NODE_NO_REMOTE_ATTESTATION\n"
    "REMOTE_ATTESTATION_PRESENT=false\n"
    "INDEPENDENT_REPLAY_REQUIRED=true\n"
    "GRID=N0_U:2,N1_U:1,S:2\n"
    "ORDER=8\n"
    "C1_SET=C1Rect2Set\n"
    "C1_INITIAL_DERIVATIVE=B*R_SOURCE_TANGENT_ZERO_NORMAL\n"
    "RAW_TILES=6\n"
    "EDGE_RECORDS=10\n"
    "SHARDS=2\n"
)
(root / "run-manifest.txt").write_text(manifest, encoding="ascii")

geometry = {
    "N0": (Fraction("0"), Fraction("0.004"), Fraction("0.3"), 2),
    "N1": (Fraction("0.019771776972779206"), Fraction("0.0015"), Fraction("0.3"), 1),
}
edges = {
    "N0": (("N0", Fraction(2)), ("N1", Fraction(16, 3))),
    "N1": (("N0", Fraction(1)),),
}

for ordinal in (1, 2):
    records = []
    raw_count = 0
    for source, (center, ru, rs, u_tiles) in geometry.items():
        for u in range(u_tiles):
            for s in range(2):
                linear = u * 2 + s
                if linear % 2 != ordinal - 1:
                    continue
                raw_count += 1
                u0 = center - ru + Fraction(2) * ru * u / u_tiles
                u1 = center - ru + Fraction(2) * ru * (u + 1) / u_tiles
                s0 = -rs + Fraction(2) * rs * s / 2
                s1 = -rs + Fraction(2) * rs * (s + 1) / 2
                for target, p in edges[source]:
                    values = {
                        "SOURCE": source,
                        "TARGET": target,
                        "EDGE": f"{source}->{target}",
                        "U_INDEX": str(u),
                        "S_INDEX": str(s),
                        "SOURCE_U": interval(u0, u1),
                        "SOURCE_S": interval(s0, s1),
                        "A00": interval(p),
                        "A01": interval(0),
                        "A10": interval(0),
                        "A11": interval(Fraction(1, 2)),
                        "TILE_M00": interval(1, 2),
                        "TILE_DET_M_NAIVE": interval(1, 2),
                        "TILE_DET_M_EXPANDED": interval(1, 2),
                        "C1_RETURN_TIME": interval(6, 7),
                        "INTEGRAL_DIVERGENCE": interval(0),
                        "EXP_INTEGRAL_DIVERGENCE": interval(1),
                        "DET_LIOUVILLE": interval(
                            1 if source == "N0" else Fraction(4, 3)
                        ),
                    }
                    for crossing in range(7):
                        normal = (
                            1 if source == "N0" else
                            (4 if crossing == 0 else 3)
                        )
                        values[f"NU{crossing}"] = interval(normal)
                    for crossing in range(1, 7):
                        values[f"T{crossing}"] = interval(
                            crossing, Fraction(100 * crossing + 1, 100)
                        )
                    values["TILE_CONE_DIAGNOSTIC"] = "true"
                    values["LIOUVILLE_INVERTIBLE"] = "true"
                    records.append(" ".join(f"{key}={value}" for key, value in values.items()) + "\n")
    (root / f"ledger-{ordinal}.txt").write_text("".join(records), encoding="ascii")
    output = (
        "SCHEMA=sounio.cs6.capd-c1-cone.v1\n"
        "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
        "INTERVAL_BACKEND_DECLARED=FILIB\n"
        "C1_SET=C1Rect2Set\n"
        "C1_INITIAL_DERIVATIVE=B*R_SOURCE_TANGENT_ZERO_NORMAL\n"
        "MAP=P^6\nRETURNS_PER_MAP=6\n"
        "SECTION_ORIENTATION=MinusPlus\nORDER=8\n"
        "Q_DECIMAL_INTERPRETATION=exact-decimal-input-outward-interval\n"
        "LEDGER_ENDPOINT_ENCODING=outward-one-ulp-exact-hexadecimal-binary64\n"
        "Q_N0=1,-2.3023784599059653\n"
        "Q_N1=0.06526711140171336,-2.3023784599059653\n"
        "CONE_DETERMINANT_FORM=expanded-exact-cancellation-before-interval-evaluation\n"
        "C1_LIOUVILLE_FINAL_RETURN_OVERLAP_REQUIRED=true\n"
        "LIOUVILLE_EXPONENTIAL_OPERAND_EMITTED=true\n"
        "C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_REQUIRED=true\n"
        "VECTOR_FIELD_CAPD=par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;\n"
        "LIOUVILLE_FIELD_CAPD=par:zs;var:x,y,w,ell;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs,x-y-(w+zs)/2-1;\n"
        "ZSEC=[22.327463739099997, 22.327463739100004]\n"
        "ORIGIN={[15.186446520640784, 15.186446520640787],[10.908543194765464, 10.908543194765468]}\n"
        "UNSTABLE={[-0.6743031621419977, -0.67430316214199748],[-0.73845463335624284, -0.73845463335624262]}\n"
        "STABLE={[-0.94170446778164529, -0.94170446778164507],[0.33644122125579118, 0.33644122125579129]}\n"
        "FRAME_DETERMINANT=[-0.92226940685332637, -0.9222694068533257]\n"
        "N0_LOCAL={[0, 0],[0, 0],[0.0039999999999999992, 0.004000000000000001],[0.29999999999999993, 0.30000000000000004]}\n"
        "N1_LOCAL={[0.019771776972779202, 0.019771776972779209],[0, 0],[0.0014999999999999998, 0.0015000000000000002],[0.29999999999999993, 0.30000000000000004]}\n"
        "HSETS_DISJOINT=true\n"
        "FRAME_RIGOROUSLY_INVERTIBLE=true\n"
        "GRID=N0_U:2,N1_U:1,S:2\n"
        f"SHARD={ordinal}/2\n"
        "LEDGER_ENABLED=true\n"
        f"RAW_TILES_EXPECTED={raw_count}\n"
        f"RAW_TILES_PROCESSED={raw_count}\n"
        f"RAW_TILES_VALID={raw_count}\n"
        f"EDGE_RECORDS_EXPECTED={len(records)}\n"
        f"EDGE_RECORDS_WRITTEN={len(records)}\n"
        f"TILE_CONE_DIAGNOSTIC_PASSES={len(records)}\n"
        "SHARD_PASS=true\n"
        "FULL_SOURCE_GLOBAL_HULL_TESTED=false\n"
        "PAIRWISE_CHORD_CONE_CONDITION_PROVED=false\n"
        "TANGENT_CONE_CONDITION_PROVED=false\n"
        "LIOUVILLE_INVERTIBILITY_PROVED=false\n"
        "UNIFORM_HYPERBOLICITY_PROVED=false\n"
        "CHAOTIC_ATTRACTOR_PROVED=false\n"
    )
    (root / f"shard-{ordinal}.txt").write_text(output, encoding="ascii")
PY

mkdir "$work/aggregate"
python3 "$aggregate" \
  --run-dir "$fixture" \
  --shards 2 \
  --source "$source_file" \
  --ledger-output "$work/aggregate/ledger.txt" \
  --certificate-output "$work/aggregate/certificate.txt"
grep -Fxq 'FULL_SOURCE_C1_DERIVATIVE_ENCLOSURE_PROVED=true' "$work/aggregate/certificate.txt"
grep -Fxq 'C0_RAW_EVIDENCE_REAGGREGATED=true' "$work/aggregate/certificate.txt"
grep -Fxq 'C0_C1_SEMANTIC_CONTRACT_BOUND=true' "$work/aggregate/certificate.txt"
grep -Fxq 'C1_DYNAMICAL_SYSTEM_PREAMBLE_BOUND=true' "$work/aggregate/certificate.txt"
grep -Fxq 'C1_EXPLICIT_CAPD_ARTIFACT_SETS_BOUND=true' "$work/aggregate/certificate.txt"
grep -Fxq 'GLOBAL_FULL_SOURCE_HULL_TESTED=true' "$work/aggregate/certificate.txt"
grep -Fxq 'PAIRWISE_CHORD_CONE_CONDITION_PROVED=true' "$work/aggregate/certificate.txt"
grep -Fxq 'LIOUVILLE_INVERTIBILITY_PROVED=true' "$work/aggregate/certificate.txt"
grep -Fxq 'C1_LIOUVILLE_FINAL_RETURN_BOUND=true' "$work/aggregate/certificate.txt"
grep -Fxq 'LIOUVILLE_DETERMINANT_OPERAND_CONSISTENCY_BOUND=true' \
  "$work/aggregate/certificate.txt"
grep -Fxq 'LIOUVILLE_EXPONENTIATION_RECOMPUTED_BY_AGGREGATOR=false' \
  "$work/aggregate/certificate.txt"
grep -Fxq 'C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_BOUND=true' \
  "$work/aggregate/certificate.txt"
grep -Fxq 'C0_EXECUTION_PROVENANCE_VERIFIED=false' "$work/aggregate/certificate.txt"
grep -Fxq 'C0_EXECUTION_TRUST_MODEL_DECLARED=AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION' \
  "$work/aggregate/certificate.txt"
grep -Fxq 'COMBINED_C0_C1_MATHEMATICAL_EVIDENCE_COMPLETE=true' "$work/aggregate/certificate.txt"
grep -Fxq 'COMBINED_C0_C1_EXECUTION_PROVENANCE_ATTESTED=false' "$work/aggregate/certificate.txt"
grep -Fxq 'UNIFORM_HYPERBOLICITY_PROVED=false' "$work/aggregate/certificate.txt"
grep -Fxq 'CHAOTIC_ATTRACTOR_PROVED=false' "$work/aggregate/certificate.txt"

if python3 "$aggregate" \
  --run-dir "$fixture" --shards 2 --source "$source_file" \
  --ledger-output "$work/aggregate/ledger.txt" \
  --certificate-output "$work/aggregate/certificate.txt" \
  >"$work/existing.out" 2>"$work/existing.err"; then
  echo "aggregator overwrote existing outputs" >&2
  exit 1
fi
grep -Fq 'refusing existing output' "$work/existing.err"

negative="$work/negative-hull"
cp -a "$fixture" "$negative"
python3 - "$negative" <<'PY'
import re
import sys
from pathlib import Path
for path in Path(sys.argv[1]).glob("ledger-*.txt"):
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    lines = [
        re.sub(
            r"A00=\[[^]]+\]",
            "A00=[0x1.fffffffffffffp-1,0x1.5555555555556p+2]",
            line,
            count=1,
        ) if "EDGE=N0->N1" in line else line
        for line in lines
    ]
    path.write_text("".join(lines), encoding="ascii")
PY
mkdir "$work/negative-hull-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-hull-output/ledger.txt" \
  --certificate-output "$work/negative-hull-output/certificate.txt" \
  >"$work/hull.out" 2>"$work/hull.err"; then
  echo "aggregator accepted a failing global derivative hull" >&2
  exit 1
fi
grep -Fq 'global full-source Sylvester predicate failed' "$work/hull.err"

negative="$work/negative-source"
cp -a "$fixture" "$negative"
sed -i '0,/SOURCE_U=\[[^]]*\]/{s/SOURCE_U=\[[^]]*\]/SOURCE_U=[0x0p+0,0x0p+0]/}' "$negative/ledger-1.txt"
mkdir "$work/negative-source-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-source-output/ledger.txt" \
  --certificate-output "$work/negative-source-output/certificate.txt" \
  >"$work/source.out" 2>"$work/source.err"; then
  echo "aggregator accepted a relabelled source tile" >&2
  exit 1
fi
grep -Fq 'SOURCE_U does not enclose its canonical tile' "$work/source.err"

negative="$work/negative-liouville"
cp -a "$fixture" "$negative"
sed -i '0,/DET_LIOUVILLE=\[[^]]*\]/{s/DET_LIOUVILLE=\[[^]]*\]/DET_LIOUVILLE=[0x0p+0,0x0p+0]/}' "$negative/ledger-1.txt"
mkdir "$work/negative-liouville-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-liouville-output/ledger.txt" \
  --certificate-output "$work/negative-liouville-output/certificate.txt" \
  >"$work/liouville.out" 2>"$work/liouville.err"; then
  echo "aggregator accepted a zero Liouville determinant" >&2
  exit 1
fi
grep -Fq 'Liouville invertibility failed' "$work/liouville.err"

negative="$work/negative-return-binding"
cp -a "$fixture" "$negative"
sed -i '0,/C1_RETURN_TIME=\[[^]]*\]/{s/C1_RETURN_TIME=\[[^]]*\]/C1_RETURN_TIME=[0x1p+4,0x1.1p+4]/}' \
  "$negative/ledger-1.txt"
mkdir "$work/negative-return-binding-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-return-binding-output/ledger.txt" \
  --certificate-output "$work/negative-return-binding-output/certificate.txt" \
  >"$work/return-binding.out" 2>"$work/return-binding.err"; then
  echo "aggregator accepted disjoint C1 and Liouville sixth returns" >&2
  exit 1
fi
grep -Fq 'C1/Liouville sixth-return time mismatch' "$work/return-binding.err"

negative="$work/negative-liouville-formula"
cp -a "$fixture" "$negative"
sed -i '0,/EXP_INTEGRAL_DIVERGENCE=\[[^]]*\]/{s/EXP_INTEGRAL_DIVERGENCE=\[[^]]*\]/EXP_INTEGRAL_DIVERGENCE=[0x1.fffffffffffffp+0,0x1.0000000000001p+1]/}' \
  "$negative/ledger-1.txt"
mkdir "$work/negative-liouville-formula-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-liouville-formula-output/ledger.txt" \
  --certificate-output "$work/negative-liouville-formula-output/certificate.txt" \
  >"$work/liouville-formula.out" 2>"$work/liouville-formula.err"; then
  echo "aggregator accepted a determinant inconsistent with Liouville operands" >&2
  exit 1
fi
grep -Fq 'Liouville determinant formula mismatch' "$work/liouville-formula.err"

negative="$work/negative-normalized-determinant"
cp -a "$fixture" "$negative"
sed -i '0,/A11=\[[^]]*\]/{s/A11=\[[^]]*\]/A11=[0x1.fffffffffffffp+0,0x1.0000000000001p+1]/}' \
  "$negative/ledger-1.txt"
mkdir "$work/negative-normalized-determinant-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-normalized-determinant-output/ledger.txt" \
  --certificate-output "$work/negative-normalized-determinant-output/certificate.txt" \
  >"$work/normalized-determinant.out" 2>"$work/normalized-determinant.err"; then
  echo "aggregator accepted inconsistent C1 and Liouville determinants" >&2
  exit 1
fi
grep -Fq 'C1/Liouville normalized determinant mismatch' \
  "$work/normalized-determinant.err"

negative="$work/negative-hex-underflow"
cp -a "$fixture" "$negative"
sed -i '0,/A00=\[[^]]*\]/{s/A00=\[[^]]*\]/A00=[0x1p-999999,0x1p-999999]/}' \
  "$negative/ledger-1.txt"
mkdir "$work/negative-hex-underflow-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-hex-underflow-output/ledger.txt" \
  --certificate-output "$work/negative-hex-underflow-output/certificate.txt" \
  >"$work/hex-underflow.out" 2>"$work/hex-underflow.err"; then
  echo "aggregator accepted a hex-shaped non-binary64 endpoint" >&2
  exit 1
fi
grep -Fq 'invalid A00 interval' "$work/hex-underflow.err"

negative="$work/negative-shard-count"
cp -a "$fixture" "$negative"
sed -i 's/RAW_TILES_PROCESSED=3/RAW_TILES_PROCESSED=2/' "$negative/shard-1.txt"
mkdir "$work/negative-shard-count-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-shard-count-output/ledger.txt" \
  --certificate-output "$work/negative-shard-count-output/certificate.txt" \
  >"$work/shard-count.out" 2>"$work/shard-count.err"; then
  echo "aggregator accepted a false per-shard count" >&2
  exit 1
fi
grep -Fq 'RAW_TILES_PROCESSED mismatch' "$work/shard-count.err"

negative="$work/negative-shard-owner"
cp -a "$fixture" "$negative"
mv "$negative/ledger-1.txt" "$negative/ledger-swap.txt"
mv "$negative/ledger-2.txt" "$negative/ledger-1.txt"
mv "$negative/ledger-swap.txt" "$negative/ledger-2.txt"
mkdir "$work/negative-shard-owner-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-shard-owner-output/ledger.txt" \
  --certificate-output "$work/negative-shard-owner-output/certificate.txt" \
  >"$work/shard-owner.out" 2>"$work/shard-owner.err"; then
  echo "aggregator accepted records assigned to the wrong shard" >&2
  exit 1
fi
grep -Fq 'ledger record assigned to wrong shard' "$work/shard-owner.err"

negative="$work/negative-tile-diagnostic-count"
cp -a "$fixture" "$negative"
sed -i 's/TILE_CONE_DIAGNOSTIC_PASSES=5/TILE_CONE_DIAGNOSTIC_PASSES=4/' \
  "$negative/shard-1.txt"
mkdir "$work/negative-tile-diagnostic-count-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-tile-diagnostic-count-output/ledger.txt" \
  --certificate-output "$work/negative-tile-diagnostic-count-output/certificate.txt" \
  >"$work/tile-diagnostic-count.out" 2>"$work/tile-diagnostic-count.err"; then
  echo "aggregator accepted a false tile diagnostic count" >&2
  exit 1
fi
grep -Fq 'tile cone diagnostic count mismatch' "$work/tile-diagnostic-count.err"

negative="$work/negative-c0"
cp -a "$fixture" "$negative"
sed -i 's/FIBONACCI_COVERINGS_PROVED=true/FIBONACCI_COVERINGS_PROVED=false/' "$negative/c0-certificate.txt"
python3 - "$negative" <<'PY'
import hashlib
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
digest = hashlib.sha256((root / "c0-certificate.txt").read_bytes()).hexdigest()
manifest = (root / "run-manifest.txt")
text = re.sub(r"C0_CERTIFICATE_SHA256=[0-9a-f]{64}", f"C0_CERTIFICATE_SHA256={digest}", manifest.read_text(encoding="ascii"))
manifest.write_text(text, encoding="ascii")
PY
mkdir "$work/negative-c0-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-c0-output/ledger.txt" \
  --certificate-output "$work/negative-c0-output/certificate.txt" \
  >"$work/c0.out" 2>"$work/c0.err"; then
  echo "aggregator accepted a false C0 certificate" >&2
  exit 1
fi
grep -Fq 'C0 certificate FIBONACCI_COVERINGS_PROVED mismatch' "$work/c0.err"

negative="$work/negative-c0-map"
cp -a "$fixture" "$negative"
sed -i 's/MAP=P\^6/MAP=P^5/' "$negative/c0-certificate.txt"
python3 - "$negative" <<'PY'
import hashlib
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
digest = hashlib.sha256((root / "c0-certificate.txt").read_bytes()).hexdigest()
manifest = root / "run-manifest.txt"
text = re.sub(
    r"C0_CERTIFICATE_SHA256=[0-9a-f]{64}",
    f"C0_CERTIFICATE_SHA256={digest}",
    manifest.read_text(encoding="ascii"),
)
manifest.write_text(text, encoding="ascii")
PY
mkdir "$work/negative-c0-map-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-c0-map-output/ledger.txt" \
  --certificate-output "$work/negative-c0-map-output/certificate.txt" \
  >"$work/c0-map.out" 2>"$work/c0-map.err"; then
  echo "aggregator combined a C0 certificate for a different map" >&2
  exit 1
fi
grep -Fq 'C0 certificate MAP mismatch' "$work/c0-map.err"

negative="$work/negative-c0-raw"
cp -a "$fixture" "$negative"
sed -i '0,/ PASS=true/{s/ PASS=true/ PASS=false/}' \
  "$negative/c0-run/ledger-1.txt"
mkdir "$work/negative-c0-raw-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-c0-raw-output/ledger.txt" \
  --certificate-output "$work/negative-c0-raw-output/certificate.txt" \
  >"$work/c0-raw.out" 2>"$work/c0-raw.err"; then
  echo "aggregator accepted C0 raw evidence inconsistent with its certificate" >&2
  exit 1
fi
grep -Fq 'retained C0 raw evidence failed reaggregation' "$work/c0-raw.err"

negative="$work/negative-c0-symlink"
cp -a "$fixture" "$negative"
rm "$negative/c0-run/ledger-1.txt"
ln -s "$fixture/c0-run/ledger-1.txt" "$negative/c0-run/ledger-1.txt"
mkdir "$work/negative-c0-symlink-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-c0-symlink-output/ledger.txt" \
  --certificate-output "$work/negative-c0-symlink-output/certificate.txt" \
  >"$work/c0-symlink.out" 2>"$work/c0-symlink.err"; then
  echo "aggregator accepted a C0 raw bundle backed by an external symlink" >&2
  exit 1
fi
grep -Fq 'retained C0 raw evidence bundle contains symlink' \
  "$work/c0-symlink.err"

negative="$work/negative-c1-contract"
cp -a "$fixture" "$negative"
sed -i '0,/ZSEC=\[[^]]*\]/{s/ZSEC=\[[^]]*\]/ZSEC=[1, 1]/}' \
  "$negative/shard-1.txt"
mkdir "$work/negative-c1-contract-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-c1-contract-output/ledger.txt" \
  --certificate-output "$work/negative-c1-contract-output/certificate.txt" \
  >"$work/c1-contract.out" 2>"$work/c1-contract.err"; then
  echo "aggregator combined a C1 shard for different dynamical parameters" >&2
  exit 1
fi
grep -Fq 'shard 1: ZSEC mismatch' "$work/c1-contract.err"

negative="$work/negative-library-binding"
cp -a "$fixture" "$negative"
printf '%s\n' "$negative/synthetic-include/synthetic.hpp" \
  > "$negative/capd-libs.txt"
python3 - "$negative" <<'PY'
import hashlib
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
digest = hashlib.sha256((root / "capd-libs.txt").read_bytes()).hexdigest()
manifest = root / "run-manifest.txt"
text = re.sub(
    r"CAPD_LIBS_SHA256=[0-9a-f]{64}",
    f"CAPD_LIBS_SHA256={digest}",
    manifest.read_text(encoding="ascii"),
)
manifest.write_text(text, encoding="ascii")
PY
mkdir "$work/negative-library-binding-output"
if python3 "$aggregate" --run-dir "$negative" --shards 2 --source "$source_file" \
  --ledger-output "$work/negative-library-binding-output/ledger.txt" \
  --certificate-output "$work/negative-library-binding-output/certificate.txt" \
  >"$work/library-binding.out" 2>"$work/library-binding.err"; then
  echo "aggregator accepted an unrelated CAPD library manifest" >&2
  exit 1
fi
grep -Fq 'CAPD library manifest does not match linker arguments' \
  "$work/library-binding.err"

if env -u SLURM_JOB_ID bash "$runner" \
  --run-dir "$work/refused-run" --c0-certificate "$fixture/c0-certificate.txt" \
  --c0-run-dir "$fixture/c0-run" \
  --n0-u 2 --n1-u 1 --s-tiles 2 --order 8 \
  >"$work/refused.out" 2>"$work/refused.err"; then
  echo "runner accepted an exhaustive run outside Slurm" >&2
  exit 1
fi
grep -Fq 'refusing exhaustive C1 run outside a Slurm allocation' "$work/refused.err"

mkdir "$work/mock-bin"
python3 - "$work/mock-bin/scontrol" <<'PY'
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.write_text("#!/usr/bin/env sh\nexit 1\n", encoding="ascii")
path.chmod(0o755)
PY
if PATH="$work/mock-bin:$PATH" SLURM_JOB_ID=definitely-not-a-job bash "$runner" \
  --run-dir "$work/fake-slurm-run" --c0-certificate "$fixture/c0-certificate.txt" \
  --c0-run-dir "$fixture/c0-run" \
  --n0-u 2 --n1-u 1 --s-tiles 2 --order 8 \
  >"$work/fake-slurm.out" 2>"$work/fake-slurm.err"; then
  echo "runner accepted an unverified SLURM_JOB_ID" >&2
  exit 1
fi
grep -Fq 'cannot verify SLURM_JOB_ID against the Slurm control plane' \
  "$work/fake-slurm.err"

if [[ "${CS6_CAPD_C1_SAMPLE_REPLAY:-0}" == "1" ]]; then
  capd_config="${CS6_CAPD_CONFIG:-capd-config}"
  command -v "$capd_config" >/dev/null 2>&1 || {
    echo "CS6_CAPD_C1_SAMPLE_REPLAY REFUSED: capd-config unavailable" >&2
    exit 3
  }
  # capd-config intentionally emits compiler and linker arguments.
  # shellcheck disable=SC2046
  "${CXX:-c++}" -std=c++17 -O2 "$source_file" \
    $("$capd_config" --cflags --libs) -o "$work/cs6_capd_c1_cone"
  "$work/cs6_capd_c1_cone" selftest > "$work/selftest.txt"
  grep -Fxq 'SELFTEST_PASS=true' "$work/selftest.txt"
  grep -Fxq 'EXACT_HEX_ENDPOINT_ENCODING=true' "$work/selftest.txt"
  grep -Fxq 'OUTWARD_ONE_ULP_ENDPOINT_ENCODING=true' "$work/selftest.txt"
  "$work/cs6_capd_c1_cone" probe-ledger \
    N0 N0 19999 14999 40000 30000 8 "$work/live-ledger.txt" \
    > "$work/fine-probe.txt"
  grep -Fq 'PROBE_EDGE=N0->N0' "$work/fine-probe.txt"
  grep -Fxq 'ZSEC=[22.327463739099997, 22.327463739100004]' \
    "$work/fine-probe.txt"
  grep -Fxq 'VECTOR_FIELD_CAPD=par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;' \
    "$work/fine-probe.txt"
  grep -Fxq 'C1_LIOUVILLE_FINAL_RETURN_OVERLAP_REQUIRED=true' \
    "$work/fine-probe.txt"
  grep -Fxq 'LIOUVILLE_EXPONENTIAL_OPERAND_EMITTED=true' \
    "$work/fine-probe.txt"
  grep -Fxq 'C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_REQUIRED=true' \
    "$work/fine-probe.txt"
  grep -Fxq 'TILE_CONE_DIAGNOSTIC=true' "$work/fine-probe.txt"
  grep -Fxq 'LIOUVILLE_INVERTIBLE=true' "$work/fine-probe.txt"
  grep -Fxq 'PROBE_PASS=true' "$work/fine-probe.txt"
  grep -Fxq 'LEDGER_ENABLED=true' "$work/fine-probe.txt"
  grep -Fxq 'EDGE_RECORDS_WRITTEN=1' "$work/fine-probe.txt"
  grep -Fxq 'PAIRWISE_CHORD_CONE_CONDITION_PROVED=false' "$work/fine-probe.txt"
  python3 - "$aggregate" "$work/live-ledger.txt" <<'PY'
import runpy
import sys
from pathlib import Path

module = runpy.run_path(sys.argv[1])
raw_lines = Path(sys.argv[2]).read_bytes().splitlines(keepends=True)
assert len(raw_lines) == 1 and raw_lines[0].endswith(b"\n")
fields = module["parse_ledger_line"](raw_lines[0])
assert set(fields) == module["LEDGER_REQUIRED_FIELDS"]
values = {
    name: module["parse_interval"](fields[name], name, require_hex=True)
    for name in module["LEDGER_INTERVAL_FIELDS"]
}
ideal_u, ideal_s = module["ideal_tile"]("N0", 19999, 14999, 40000, 30000)
module["check_tight_enclosure"](values["SOURCE_U"], ideal_u, "live SOURCE_U")
module["check_tight_enclosure"](values["SOURCE_S"], ideal_s, "live SOURCE_S")
assert fields["EDGE"] == "N0->N0"
assert fields["TILE_CONE_DIAGNOSTIC"] == "true"
assert fields["LIOUVILLE_INVERTIBLE"] == "true"
assert module["overlaps"](values["C1_RETURN_TIME"], values["T6"])
formula = module["interval_div_positive"](
    module["interval_mul"](
        values["EXP_INTEGRAL_DIVERGENCE"], values["NU0"]
    ),
    values["NU6"],
)
assert module["contains"](formula, values["DET_LIOUVILLE"])
c1_det = module["interval_sub"](
    module["interval_mul"](values["A00"], values["A11"]),
    module["interval_mul"](values["A01"], values["A10"]),
)
assert module["overlaps"](c1_det, values["DET_LIOUVILLE"])
PY
  "$work/cs6_capd_c1_cone" probe N0 N1 19999 14999 40000 30000 8 \
    > "$work/fine-probe-n0-n1.txt"
  grep -Fq 'PROBE_EDGE=N0->N1' "$work/fine-probe-n0-n1.txt"
  grep -Fxq 'TILE_CONE_DIAGNOSTIC=true' "$work/fine-probe-n0-n1.txt"
  grep -Fxq 'PROBE_PASS=true' "$work/fine-probe-n0-n1.txt"
  "$work/cs6_capd_c1_cone" probe N1 N0 14999 29999 30000 60000 8 \
    > "$work/fine-probe-n1-n0.txt"
  grep -Fq 'PROBE_EDGE=N1->N0' "$work/fine-probe-n1-n0.txt"
  grep -Fxq 'TILE_CONE_DIAGNOSTIC=true' "$work/fine-probe-n1-n0.txt"
  grep -Fxq 'PROBE_PASS=true' "$work/fine-probe-n1-n0.txt"
  if "$work/cs6_capd_c1_cone" probe N0 N0 39999 29999 40000 30000 8 \
    > "$work/failing-cone.out" 2> "$work/failing-cone.err"; then
    echo "failing cone probe returned success" >&2
    exit 1
  fi
  grep -Fxq 'TILE_CONE_DIAGNOSTIC=false' "$work/failing-cone.out"
  grep -Fxq 'PROBE_PASS=false' "$work/failing-cone.out"
  if "$work/cs6_capd_c1_cone" probe N0 N0 99 37 200 75 8 \
    > "$work/c0-scale.out" 2> "$work/c0-scale.err"; then
    echo "C0-scale C1 probe unexpectedly passed; recalibrate the frozen boundary" >&2
    exit 1
  fi
  grep -Fq 'possible nontransversal return to the section' "$work/c0-scale.err"
  echo "CS6_CAPD_C1_SAMPLE_REPLAY PASS"
fi

bash "$repo_root/scripts/ci/cs6_cone_scout_gate.sh"
bash "$repo_root/scripts/ci/cs6_fibonacci_scout_gate.sh"
echo "CS6_CAPD_C1_CONE_GATE PASS"
