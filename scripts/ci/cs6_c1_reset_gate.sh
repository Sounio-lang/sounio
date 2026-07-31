#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_c1_reset_probe.cpp"
verifier="$repo_root/scripts/research/cs6_c1_reset_verify.py"
runner="$repo_root/scripts/research/cs6_c1_reset_probe_run.sh"
receipt="$repo_root/scripts/research/cs6_c1_reset_receipt_v1.txt"
receipt_n0_n1="$repo_root/scripts/research/cs6_c1_reset_n0_n1_receipt_v1.txt"
receipt_n1_n0="$repo_root/scripts/research/cs6_c1_reset_n1_n0_receipt_v1.txt"
coarse_receipt="$repo_root/scripts/research/cs6_c1_reset_coarse_failure_v1.txt"
provenance="$repo_root/scripts/research/cs6_c1_reset_provenance_v1.txt"
document="$repo_root/docs/research/cs6_c1_reset_2026-07-31.md"

for path in "$source_file" "$verifier" "$runner" "$receipt" "$receipt_n0_n1" \
  "$receipt_n1_n0" "$coarse_receipt" "$provenance" "$document"; do
  test -f "$path"
done

bash -n "$runner"
python3 -m py_compile "$verifier"
verify_n0_n0="$(python3 "$verifier" "$receipt" --expect-rebox-worse)"
verify_n0_n1="$(python3 "$verifier" "$receipt_n0_n1" --expect-rebox-worse)"
verify_n1_n0="$(python3 "$verifier" "$receipt_n1_n0")"
physical_n0_n0="$(sed -n 's/^PHYSICAL_CHAIN_SHA256=//p' <<< "$verify_n0_n0")"
physical_n0_n1="$(sed -n 's/^PHYSICAL_CHAIN_SHA256=//p' <<< "$verify_n0_n1")"
[[ -n "$physical_n0_n0" && "$physical_n0_n0" == "$physical_n0_n1" ]] || {
  printf 'target-only normalization changed the N0 physical chain\n' >&2
  exit 1
}
grep -Fxq 'CANONICAL_TO_DIRECT_RATIO=9.6536938226073339' <<< "$verify_n1_n0"
grep -Fxq 'DYADIC_TO_DIRECT_RATIO=9.8634927686827218' <<< "$verify_n1_n0"
if python3 "$verifier" "$coarse_receipt" > /dev/null 2>&1; then
  printf 'ordinary success verifier accepted the coarse failure receipt\n' >&2
  exit 1
fi
verify_coarse="$(python3 "$verifier" "$coarse_receipt" \
  --expect-c0-nontransversal-failure)"
grep -Fxq 'RESULT_CLASS=BOUNDED_EXPECTED_C0_NONTRANSVERSAL_FAILURE' \
  <<< "$verify_coarse"
grep -Fxq 'LIOUVILLE_PREFIX_COUNT=5' <<< "$verify_coarse"

python3 - "$repo_root" "$provenance" "$source_file" "$verifier" "$runner" <<'PY'
import hashlib
import re
import sys
import tarfile
from pathlib import Path

repo = Path(sys.argv[1])
provenance = Path(sys.argv[2])
source = Path(sys.argv[3])
verifier = Path(sys.argv[4])
runner = Path(sys.argv[5])
sha_re = re.compile(r"^[0-9a-f]{64}$")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

header = {}
runs = []
for line_number, line in enumerate(provenance.read_text(encoding="ascii").splitlines(), 1):
    if not line:
        raise SystemExit(f"blank provenance line {line_number}")
    tokens = line.split()
    if tokens[0] == "RUN":
        record = {}
        for token in tokens[1:]:
            key, separator, value = token.partition("=")
            if not separator or not key or not value or key in record:
                raise SystemExit(f"bad provenance run token on line {line_number}")
            record[key] = value
        runs.append(record)
        continue
    if len(tokens) != 1:
        raise SystemExit(f"bad provenance header line {line_number}")
    key, separator, value = tokens[0].partition("=")
    if not separator or not key or not value or key in header:
        raise SystemExit(f"bad provenance header token on line {line_number}")
    header[key] = value

expected_header = {
    "SCHEMA", "SOURCE_SHA256", "VERIFIER_SHA256", "RUNNER_SHA256",
    "EXECUTABLE_SHA256", "CAPD_CONFIG_SHA256", "CAPD_CFLAGS_SHA256",
    "CAPD_LIBS_SHA256", "CAPD_VERSION_SHA256", "CAPD_LIBRARY_MANIFEST_SHA256",
    "CAPD_HEADER_MANIFEST_SHA256", "RUNTIME_LIBRARY_MANIFEST_SHA256",
    "CAPD_PREPROCESSOR_MACROS_SHA256", "CXX_EFFECTIVE_OPTIONS_SHA256",
    "CXX_EFFECTIVE_OPTIONS_STDERR_SHA256",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE", "EXECUTION_TRUST_MODEL",
    "REMOTE_ATTESTATION_PRESENT", "INDEPENDENT_REPLAY_REQUIRED",
    "PROMOTION_ELIGIBLE",
}
if set(header) != expected_header:
    raise SystemExit("provenance header grammar mismatch")
fixed = {
    "SCHEMA": "sounio.cs6.c1-reset-bounded-provenance.v1",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE": "false",
    "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
    "REMOTE_ATTESTATION_PRESENT": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "PROMOTION_ELIGIBLE": "false",
}
for key, value in fixed.items():
    if header[key] != value:
        raise SystemExit(f"provenance {key} changed")
for key, value in header.items():
    if key.endswith("SHA256") and sha_re.fullmatch(value) is None:
        raise SystemExit(f"malformed provenance hash {key}")
for key, path in (("SOURCE_SHA256", source), ("VERIFIER_SHA256", verifier),
                  ("RUNNER_SHA256", runner)):
    if header[key] != digest(path):
        raise SystemExit(f"current {path.name} does not match executed provenance")

toolchain_receipts = {
    "CAPD_CFLAGS_SHA256": "capd-cflags.txt",
    "CAPD_LIBS_SHA256": "capd-libs.txt",
    "CAPD_VERSION_SHA256": "capd-version.txt",
    "CAPD_PREPROCESSOR_MACROS_SHA256": "capd-preprocessor-macros.txt",
    "CXX_EFFECTIVE_OPTIONS_SHA256": "cxx-effective-options.txt",
    "CXX_EFFECTIVE_OPTIONS_STDERR_SHA256": "cxx-effective-options-stderr.txt",
}
toolchain_archive = repo / "scripts/research/receipts/cs6_c1_reset_effective_toolchain_v1.tar.gz"
with tarfile.open(toolchain_archive, mode="r:gz") as archive:
    members = {member.name: member for member in archive.getmembers()}
    if set(members) != set(toolchain_receipts.values()):
        raise SystemExit("retained toolchain archive grammar mismatch")
    for key, member_name in toolchain_receipts.items():
        member = members[member_name]
        if not member.isfile():
            raise SystemExit(f"retained toolchain member is not a file: {member_name}")
        stream = archive.extractfile(member)
        if stream is None or header[key] != hashlib.sha256(stream.read()).hexdigest():
            raise SystemExit(f"retained toolchain evidence mismatch for {key}")

expected_runs = {
    "N0_N0": {
        "RECEIPT": "scripts/research/cs6_c1_reset_receipt_v1.txt",
        "MANIFEST_PATH": "scripts/research/receipts/cs6_c1_reset_n0_n0_run_manifest_v1.txt",
        "VERIFICATION_PATH": "scripts/research/receipts/cs6_c1_reset_n0_n0_verification_v1.txt",
        "SOURCE": "N0", "TARGET": "N0", "U_INDEX": "20000", "S_INDEX": "15000",
        "U_TILES": "40000", "S_TILES": "30000", "PROBE_EXIT": "0",
        "EXPECTED_OUTCOME": "PASS",
    },
    "N0_N1": {
        "RECEIPT": "scripts/research/cs6_c1_reset_n0_n1_receipt_v1.txt",
        "MANIFEST_PATH": "scripts/research/receipts/cs6_c1_reset_n0_n1_run_manifest_v1.txt",
        "VERIFICATION_PATH": "scripts/research/receipts/cs6_c1_reset_n0_n1_verification_v1.txt",
        "SOURCE": "N0", "TARGET": "N1", "U_INDEX": "20000", "S_INDEX": "15000",
        "U_TILES": "40000", "S_TILES": "30000", "PROBE_EXIT": "0",
        "EXPECTED_OUTCOME": "PASS",
    },
    "N1_N0": {
        "RECEIPT": "scripts/research/cs6_c1_reset_n1_n0_receipt_v1.txt",
        "MANIFEST_PATH": "scripts/research/receipts/cs6_c1_reset_n1_n0_run_manifest_v1.txt",
        "VERIFICATION_PATH": "scripts/research/receipts/cs6_c1_reset_n1_n0_verification_v1.txt",
        "SOURCE": "N1", "TARGET": "N0", "U_INDEX": "15000", "S_INDEX": "30000",
        "U_TILES": "30000", "S_TILES": "60000", "PROBE_EXIT": "0",
        "EXPECTED_OUTCOME": "PASS",
    },
    "COARSE_N0_N0": {
        "RECEIPT": "scripts/research/cs6_c1_reset_coarse_failure_v1.txt",
        "MANIFEST_PATH": "scripts/research/receipts/cs6_c1_reset_coarse_run_manifest_v1.txt",
        "VERIFICATION_PATH": "scripts/research/receipts/cs6_c1_reset_coarse_verification_v1.txt",
        "SOURCE": "N0", "TARGET": "N0", "U_INDEX": "99", "S_INDEX": "37",
        "U_TILES": "200", "S_TILES": "75", "PROBE_EXIT": "2",
        "EXPECTED_OUTCOME": "C0_NONTRANSVERSAL_FAILURE",
    },
}
run_keys = {
    "ID", "RECEIPT", "LEDGER_SHA256", "MANIFEST_SHA256", "VERIFICATION_SHA256",
    "MANIFEST_PATH", "VERIFICATION_PATH",
    "SOURCE", "TARGET", "U_INDEX", "S_INDEX", "U_TILES", "S_TILES", "ORDER",
    "PROBE_EXIT", "EXPECTED_OUTCOME",
}
if len(runs) != len(expected_runs) or {record.get("ID") for record in runs} != set(expected_runs):
    raise SystemExit("provenance run set mismatch")
for record in runs:
    if set(record) != run_keys:
        raise SystemExit(f"provenance run grammar mismatch for {record.get('ID')}")
    expected = expected_runs[record["ID"]]
    for key, value in expected.items():
        if record[key] != value:
            raise SystemExit(f"provenance {key} mismatch for {record['ID']}")
    if record["ORDER"] != "8":
        raise SystemExit(f"provenance order mismatch for {record['ID']}")
    for key in ("LEDGER_SHA256", "MANIFEST_SHA256", "VERIFICATION_SHA256"):
        if sha_re.fullmatch(record[key]) is None:
            raise SystemExit(f"malformed {key} for {record['ID']}")
    receipt = repo / record["RECEIPT"]
    manifest_path = repo / record["MANIFEST_PATH"]
    verification_path = repo / record["VERIFICATION_PATH"]
    if digest(receipt) != record["LEDGER_SHA256"]:
        raise SystemExit(f"receipt hash mismatch for {record['ID']}")
    if digest(manifest_path) != record["MANIFEST_SHA256"]:
        raise SystemExit(f"retained manifest hash mismatch for {record['ID']}")
    if digest(verification_path) != record["VERIFICATION_SHA256"]:
        raise SystemExit(f"retained verification hash mismatch for {record['ID']}")
    ledger_header = {}
    for line in receipt.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if stripped.split(" ", 1)[0] in {
            "RESULT", "PREFIX", "RESET", "LIOUVILLE_STATUS", "LIOUVILLE", "SUMMARY"
        }:
            break
        for token in stripped.split():
            key, separator, value = token.partition("=")
            if not separator or key in ledger_header:
                raise SystemExit(f"bad receipt header for {record['ID']}")
            ledger_header[key] = value
    for key in ("SOURCE", "TARGET", "U_INDEX", "S_INDEX", "U_TILES", "S_TILES"):
        if ledger_header.get(key) != record[key]:
            raise SystemExit(f"receipt {key} mismatch for {record['ID']}")
    if ledger_header.get("ORDER") != "8":
        raise SystemExit(f"receipt order mismatch for {record['ID']}")
    source_binding = f"WORKER_SOURCE_SHA256={header['SOURCE_SHA256']}"
    if receipt.read_text(encoding="ascii").splitlines().count(source_binding) != 1:
        raise SystemExit(f"receipt/source binding mismatch for {record['ID']}")

    manifest = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in manifest:
            raise SystemExit(f"bad retained manifest for {record['ID']}")
        manifest[key] = value
    manifest_keys = {
        "MANIFEST_KIND", "RUN_COMPLETE", "SOURCE_SHA256", "VERIFIER_SHA256",
        "RUNNER_SHA256", "EXECUTABLE_SHA256", "LEDGER_SHA256",
        "VERIFICATION_SHA256", "CAPD_CONFIG_SHA256", "CAPD_CFLAGS_SHA256",
        "CAPD_LIBS_SHA256", "CAPD_VERSION_SHA256",
        "CAPD_PREPROCESSOR_MACROS_SHA256", "CXX_EFFECTIVE_OPTIONS_SHA256",
        "CXX_EFFECTIVE_OPTIONS_STDERR_SHA256", "CXX_DRIVER_SHA256",
        "CXX_VERSION_SHA256", "RUNTIME_LINKAGE_SHA256",
        "CAPD_LIBRARY_MANIFEST_SHA256", "CAPD_HEADER_MANIFEST_SHA256",
        "RUNTIME_LIBRARY_MANIFEST_SHA256", "COMPILE_STDERR_SHA256",
        "PROBE_STDERR_SHA256", "PYTHON_DRIVER_SHA256", "PYTHON_VERSION_SHA256",
        "GIT_HEAD_SHA256", "GIT_STATUS_SHA256", "CAPD_CONFIG_PATH", "CXX_PATH",
        "PYTHON_PATH", "SOURCE", "TARGET", "U_INDEX", "S_INDEX", "U_TILES",
        "S_TILES", "ORDER", "PROBE_EXIT", "EXPECTED_OUTCOME",
        "EXPECT_REBOX_WORSE", "DEPENDENCY_CONTENT_HASHES_COMPLETE",
        "EXECUTION_TRUST_MODEL", "REMOTE_ATTESTATION_PRESENT",
        "INDEPENDENT_REPLAY_REQUIRED", "PROMOTION_ELIGIBLE",
    }
    if set(manifest) != manifest_keys:
        raise SystemExit(f"retained manifest grammar mismatch for {record['ID']}")
    manifest_expected = {
        "MANIFEST_KIND": "CS6_C1_RESET_BOUNDED_RUN_V1", "RUN_COMPLETE": "true",
        "SOURCE_SHA256": header["SOURCE_SHA256"],
        "VERIFIER_SHA256": header["VERIFIER_SHA256"],
        "RUNNER_SHA256": header["RUNNER_SHA256"],
        "EXECUTABLE_SHA256": header["EXECUTABLE_SHA256"],
        "LEDGER_SHA256": record["LEDGER_SHA256"],
        "SOURCE": record["SOURCE"], "TARGET": record["TARGET"],
        "U_INDEX": record["U_INDEX"], "S_INDEX": record["S_INDEX"],
        "U_TILES": record["U_TILES"], "S_TILES": record["S_TILES"],
        "ORDER": "8", "PROBE_EXIT": record["PROBE_EXIT"],
        "EXPECTED_OUTCOME": record["EXPECTED_OUTCOME"],
        "EXPECT_REBOX_WORSE": "true" if record["ID"] in {"N0_N0", "N0_N1"} else "false",
        "DEPENDENCY_CONTENT_HASHES_COMPLETE": "false",
        "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
        "REMOTE_ATTESTATION_PRESENT": "false",
        "INDEPENDENT_REPLAY_REQUIRED": "true", "PROMOTION_ELIGIBLE": "false",
        "CAPD_CONFIG_SHA256": header["CAPD_CONFIG_SHA256"],
        "CAPD_CFLAGS_SHA256": header["CAPD_CFLAGS_SHA256"],
        "CAPD_LIBS_SHA256": header["CAPD_LIBS_SHA256"],
        "CAPD_VERSION_SHA256": header["CAPD_VERSION_SHA256"],
        "CAPD_LIBRARY_MANIFEST_SHA256": header["CAPD_LIBRARY_MANIFEST_SHA256"],
        "CAPD_HEADER_MANIFEST_SHA256": header["CAPD_HEADER_MANIFEST_SHA256"],
        "RUNTIME_LIBRARY_MANIFEST_SHA256": header["RUNTIME_LIBRARY_MANIFEST_SHA256"],
        "CAPD_PREPROCESSOR_MACROS_SHA256": header["CAPD_PREPROCESSOR_MACROS_SHA256"],
        "CXX_EFFECTIVE_OPTIONS_SHA256": header["CXX_EFFECTIVE_OPTIONS_SHA256"],
        "CXX_EFFECTIVE_OPTIONS_STDERR_SHA256": header["CXX_EFFECTIVE_OPTIONS_STDERR_SHA256"],
    }
    for key, value in manifest_expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"retained manifest {key} mismatch for {record['ID']}")
    if manifest.get("VERIFICATION_SHA256") != record["VERIFICATION_SHA256"]:
        raise SystemExit(f"manifest verification hash mismatch for {record['ID']}")

    verification_lines = verification_path.read_text(encoding="ascii").splitlines()
    tile = (
        f"{record['U_INDEX']},{record['S_INDEX']}/"
        f"{record['U_TILES']},{record['S_TILES']}"
    )
    required_verification = {
        "VERIFY_PASS=true", f"LEDGER_SHA256={record['LEDGER_SHA256']}",
        f"SOURCE={record['SOURCE']}", f"TARGET={record['TARGET']}",
        f"TILE={tile}", "PROMOTION_ELIGIBLE=false",
    }
    if not required_verification.issubset(set(verification_lines)):
        raise SystemExit(f"retained verification content mismatch for {record['ID']}")
    if record["PROBE_EXIT"] == "2":
        required_failure = {
            "RESULT_CLASS=BOUNDED_EXPECTED_C0_NONTRANSVERSAL_FAILURE",
            "C1_STRATEGIES_COMPLETE=true", "LIOUVILLE_PREFIX_COUNT=5", "WORKER_EXIT=2",
        }
        if not required_failure.issubset(set(verification_lines)):
            raise SystemExit("coarse retained verification is incomplete")
    elif "RESULT_CLASS=BOUNDED_NEGATIVE_EFFICIENCY_RESULT" not in verification_lines:
        raise SystemExit(f"success result class mismatch for {record['ID']}")
print("PROVENANCE_BINDINGS_VERIFIED=4")
PY

grep -Fxq 'RESET_SEMANTICS=REPRESENTATION_PRESERVING_CUMULATIVE_JRAW_REBOX' "$receipt"
grep -Fxq 'CALL_PATTERN=6xP1_SAME_MUTABLE_SET' "$receipt"
grep -Fxq 'LOCAL_FACTOR_CHAIN=false' "$receipt"
grep -Fxq 'PREFIX_DP_PRODUCT_FORBIDDEN=true' "$receipt"
grep -Fxq 'FINAL_DP=PREFIX_6_ONLY' "$receipt"
grep -Fxq 'REBOX_COUNT=5' "$receipt"
grep -Fxq 'LAST_MATRIX_POLICY=PRESERVE_EXACTLY_NO_REPARAMETERIZATION' "$receipt"
grep -Fxq 'INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX' "$receipt"
grep -Fxq 'LIOUVILLE_REJECT_ONLY=true' "$receipt"
grep -Fxq 'C1_CLIPPED_BY_LIOUVILLE=false' "$receipt"
grep -Fxq 'EXECUTION_PROVENANCE_ATTESTED=false' "$receipt"
grep -Fxq 'PROMOTION_ELIGIBLE=false' "$receipt"
grep -Fxq 'C1_REBOX_SCALING_BLOCKER_RESOLVED=false' "$receipt"
grep -Fxq 'FULL_SOURCE_C1_DERIVATIVE_ENCLOSURE_PROVED=false' "$receipt"
grep -Fxq 'UNIFORM_HYPERBOLICITY_PROVED=false' "$receipt"
grep -Fxq 'CHAOTIC_ATTRACTOR_PROVED=false' "$receipt"

python3 - "$source_file" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="ascii")
match = re.search(
    r"ResetAudit rebox\(.*?\n  }\n};",
    source,
    flags=re.DOTALL,
)
if match is None:
    raise SystemExit("cannot isolate ResettableC1Rect2Set::rebox")
rebox = match.group(0)
required = (
    "audit.pre_internal = static_cast<IMatrix>(*this);",
    "const IMatrix last_before = getLastMatrixEnclosure();",
    "C1BaseSet reset(candidate);",
    "static_cast<C1BaseSet&>(*this) = reset;",
    "m_invBjac = IMatrix::Identity(kDimension);",
    "m_currentMatrix = candidate;",
    "equal(last_before, getLastMatrixEnclosure())",
    "subset(physical_before, physical_after)",
    "zero_third_column(candidate)",
)
for needle in required:
    if needle not in rebox:
        raise SystemExit(f"missing atomic rebox contract: {needle}")
for forbidden in (
    "computeDP",
    "flow_derivative",
    "flowDerivative",
    "C1BaseSet(candidate, true)",
    "m_lastMatrixEnclosure =",
):
    if forbidden in rebox:
        raise SystemExit(f"forbidden rebox operation: {forbidden}")
PY

python3 - "$verifier" "$receipt" "$coarse_receipt" <<'PY'
import re
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from fractions import Fraction

verifier = Path(sys.argv[1])
baseline = Path(sys.argv[2]).read_text(encoding="ascii")
coarse = Path(sys.argv[3]).read_text(encoding="ascii")

spec = importlib.util.spec_from_file_location("cs6_reset_verifier", verifier)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
wide = module.Interval(Fraction(0), Fraction(10))
left = module.Interval(Fraction(0), Fraction(1))
right = module.Interval(Fraction(9), Fraction(10))
if not wide.overlaps(left) or not wide.overlaps(right) or module.joint_interval((wide, left, right)):
    raise SystemExit("joint-intersection adversarial unit failed")

def replace_once(text, old, new):
    if text.count(old) < 1:
        raise RuntimeError(f"mutation anchor missing: {old}")
    return text.replace(old, new, 1)

def mutate_first_record(text, marker, transform):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(marker + " "):
            lines[index] = transform(line)
            return "\n".join(lines) + "\n"
    raise RuntimeError(f"record marker missing: {marker}")

def copy_first_time_to_second_sequential(text):
    lines = text.splitlines()
    indices = [index for index, line in enumerate(lines)
               if line.startswith("PREFIX STRATEGY=sequential ")]
    if len(indices) != 6:
        raise RuntimeError("sequential prefix anchors missing")
    first = re.search(r"TIME=(\[[^\]]+\])", lines[indices[0]]).group(1)
    lines[indices[1]] = re.sub(r"TIME=\[[^\]]+\]", f"TIME={first}", lines[indices[1]], count=1)
    return "\n".join(lines) + "\n"

def make_first_prefix_x2_positive(text):
    def transform(line):
        positive = re.search(r"X0=(\[[^\]]+\])", line).group(1)
        return re.sub(r"X2=\[[^\]]+\]", f"X2={positive}", line, count=1)
    return mutate_first_record(text, "PREFIX", transform)

def widen_rebox_a(text):
    lines = text.splitlines()
    changed = 0
    for index, line in enumerate(lines):
        if line.startswith(("RESULT STRATEGY=canonical-rebox ",
                            "RESULT STRATEGY=dyadic-right-rebox ")):
            lines[index] = re.sub(
                r"A00=\[[^\]]+\]", "A00=[-0x1p+20,0x1p+20]", line, count=1
            )
            changed += 1
    if changed != 2:
        raise RuntimeError("rebox result anchors missing")
    return "\n".join(lines) + "\n"

def forge_direct_determinant(text):
    forged = {
        "DP00": "[0x1.47ae147ae147ap-6,0x1.47ae147ae147cp-6]",
        "DP01": "[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]",
        "DP10": "[0x1.47ae147ae147ap-7,0x1.47ae147ae147cp-7]",
        "DP11": "[0x1.0624dd2f1a9fbp-11,0x1.0624dd2f1a9fdp-11]",
        "A00": "[-0x1.181b5559f831dp+2,-0x1.181b5559f8317p+2]",
        "A01": "[-0x1.05650bafc1a6ep-3,-0x1.05650bafc1a68p-3]",
        "A10": "[-0x1.db45f18003faap-6,-0x1.db45f18003f9fp-6]",
        "A11": "[0x1.3f70080c05734p-10,0x1.3f70080c0573ap-10]",
        "A_MAX_WIDTH": "3.5527136788005009e-15",
    }
    def transform(line):
        for key, value in forged.items():
            line, count = re.subn(rf"{key}=[^ ]+", f"{key}={value}", line, count=1)
            if count != 1:
                raise RuntimeError(f"direct determinant anchor missing: {key}")
        return line
    return mutate_first_record(text, "RESULT STRATEGY=direct", transform)

mutations = {
    "local-factor-lie": lambda t: replace_once(t, "LOCAL_FACTOR_CHAIN=false", "LOCAL_FACTOR_CHAIN=true"),
    "prefix-product": lambda t: replace_once(t, "PREFIX_DP_PRODUCT_FORBIDDEN=true", "PREFIX_DP_PRODUCT_FORBIDDEN=false"),
    "wrong-final": lambda t: replace_once(t, "FINAL_DP=PREFIX_6_ONLY", "FINAL_DP=PRODUCT_OF_PREFIX_DP"),
    "c0-drift": lambda t: replace_once(t, "C0_UNCHANGED=true", "C0_UNCHANGED=false"),
    "last-matrix-drift": lambda t: replace_once(t, "LAST_MATRIX_UNCHANGED=true", "LAST_MATRIX_UNCHANGED=false"),
    "stale-inverse": lambda t: replace_once(t, "INVERSE_BASIS_IDENTITY=true", "INVERSE_BASIS_IDENTITY=false"),
    "third-column-loss": lambda t: replace_once(t, "THIRD_COLUMN_ZERO=true", "THIRD_COLUMN_ZERO=false"),
    "event-dp-seed": lambda t: replace_once(t, "CANDIDATE_SOURCE=POSTSECTION_CURRENT_MATRIX", "CANDIDATE_SOURCE=EVENT_COMPUTE_DP"),
    "wrong-reset-count": lambda t: replace_once(t, "REBOX_COUNT=5", "REBOX_COUNT=4"),
    "bad-chart": lambda t: mutate_first_record(t, "RESET STRATEGY=canonical-rebox", lambda line: replace_once(line, "S0=0x1p+0", "S0=0x1.8p+0")),
    "narrowed-candidate": lambda t: mutate_first_record(t, "RESET STRATEGY=canonical-rebox", lambda line: re.sub(r"POST00=\[[^\]]+\]", "POST00=[0x0p+0,0x0p+0]", line, count=1)),
    "missing-reset": lambda t: "\n".join(line for line in t.splitlines() if not line.startswith("RESET STRATEGY=canonical-rebox RETURN=3 ")) + "\n",
    "missing-prefix": lambda t: "\n".join(line for line in t.splitlines() if not line.startswith("PREFIX STRATEGY=sequential RETURN=4 ")) + "\n",
    "liouville-intermediate-disjoint": lambda t: mutate_first_record(t, "LIOUVILLE", lambda line: re.sub(r"X0=\[[^\]]+\]", "X0=[0x1p+100,0x1p+101]", line, count=1)),
    "liouville-clipping": lambda t: replace_once(t, "C1_CLIPPED_BY_LIOUVILLE=false", "C1_CLIPPED_BY_LIOUVILLE=true"),
    "promotion-lie": lambda t: replace_once(t, "PROMOTION_ELIGIBLE=false", "PROMOTION_ELIGIBLE=true"),
    "summary-lie": lambda t: replace_once(t, "PROBE_PASS=true", "PROBE_PASS=false"),
    "nonfinite-endpoint": lambda t: re.sub(r"SOURCE_U=\[[^\]]+\]", "SOURCE_U=[-0x1p+0,inf]", t, count=1),
    "inverted-interval": lambda t: re.sub(r"SOURCE_U=\[[^\]]+\]", "SOURCE_U=[0x1p+1,0x1p+0]", t, count=1),
    "chart-underflow-zero": lambda t: mutate_first_record(t, "RESET STRATEGY=canonical-rebox", lambda line: replace_once(line, "NEW_E0=0x1p+0", "NEW_E0=0x0p+0")),
    "normal-chart-scaled": lambda t: mutate_first_record(t, "RESET STRATEGY=canonical-rebox", lambda line: replace_once(line, "S2=0x1p+0", "S2=0x1p+1")),
    "scale-audit-false": lambda t: replace_once(t, "SCALE_CHAIN_VALID=true", "SCALE_CHAIN_VALID=false"),
    "nu-crosses-zero": lambda t: mutate_first_record(t, "PREFIX", lambda line: re.sub(r"NU=\[[^\]]+\]", "NU=[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]", line, count=1)),
    "section-coordinate-positive": make_first_prefix_x2_positive,
    "time-not-increasing": copy_first_time_to_second_sequential,
    "zero-exp-ell": lambda t: mutate_first_record(t, "LIOUVILLE", lambda line: re.sub(r"EXP_ELL=\[[^\]]+\]", "EXP_ELL=[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]", line, count=1)),
    "infinite-liouville-det": lambda t: mutate_first_record(t, "LIOUVILLE", lambda line: re.sub(r"DET_SOURCE_FRAME=\[[^\]]+\]", "DET_SOURCE_FRAME=[-0x1p+0,inf]", line, count=1)),
    "unknown-field": lambda t: t + "UNKNOWN_FIELD=true\n",
    "free-form-after-eof": lambda t: t + "this is not a ledger record\n",
    "duplicate-prefix": lambda t: t + next(line for line in t.splitlines() if line.startswith("PREFIX ")) + "\n",
    "tile-index-relabel": lambda t: replace_once(t, "U_INDEX=20000", "U_INDEX=20001"),
    "tile-count-relabel": lambda t: replace_once(t, "U_TILES=40000", "U_TILES=40001"),
    "order-relabel": lambda t: replace_once(t, "ORDER=8", "ORDER=9"),
    "order-int32-overflow": lambda t: replace_once(t, "ORDER=8", "ORDER=2147483648"),
    "worker-source-unbound": lambda t: re.sub(
        r"WORKER_SOURCE_SHA256=[0-9a-f]{64}", "WORKER_SOURCE_SHA256=UNBOUND", t, count=1
    ),
    "dp-width-zero": lambda t: re.sub(r"DP_MAX_WIDTH=[^ ]+", "DP_MAX_WIDTH=0", t, count=1),
    "a-width-zero": lambda t: re.sub(r"A_MAX_WIDTH=[^\n ]+", "A_MAX_WIDTH=0", t, count=1),
    "reported-a-widening": widen_rebox_a,
    "reported-nu-widening": lambda t: mutate_first_record(
        t, "PREFIX", lambda line: re.sub(r"NU=\[[^\]]+\]", "NU=[0x1p-100,0x1p+100]", line, count=1)
    ),
    "reported-det-widening": lambda t: mutate_first_record(
        t, "PREFIX", lambda line: re.sub(r"DET=\[[^\]]+\]", "DET=[-0x1p+100,0x1p+100]", line, count=1)
    ),
    "direct-liouville-determinant-disjoint": forge_direct_determinant,
    "worker-scale-range": lambda t: mutate_first_record(
        t, "RESET STRATEGY=canonical-rebox", lambda line: replace_once(line, "S0=0x1p+0", "S0=0x1p+501")
    ),
}

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name, mutation in mutations.items():
        candidate = root / f"{name}.txt"
        candidate.write_text(mutation(baseline), encoding="ascii")
        run = subprocess.run(
            (sys.executable, str(verifier), str(candidate)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if run.returncode == 0:
            raise SystemExit(f"negative mutation accepted: {name}")
print(f"NEGATIVE_MUTATIONS_REJECTED={len(mutations)}")

baseline_run = subprocess.run(
    (sys.executable, str(verifier), str(Path(sys.argv[2]))),
    capture_output=True, text=True, check=True,
)
baseline_digest = re.search(
    r"^PHYSICAL_CHAIN_SHA256=([0-9a-f]{64})$", baseline_run.stdout, re.MULTILINE
).group(1)
one_ulp = replace_once(
    baseline,
    "DP00=[0x1.fe5bd68811fffp-13,0x1.d0402106252c1p-6]",
    "DP00=[0x1.fe5bd68811ffep-13,0x1.d0402106252c1p-6]",
)
with tempfile.TemporaryDirectory() as directory:
    candidate = Path(directory) / "physical-digest-one-ulp.txt"
    candidate.write_text(one_ulp, encoding="ascii")
    changed_run = subprocess.run(
        (sys.executable, str(verifier), str(candidate)),
        capture_output=True, text=True, check=True,
    )
changed_digest = re.search(
    r"^PHYSICAL_CHAIN_SHA256=([0-9a-f]{64})$", changed_run.stdout, re.MULTILINE
).group(1)
if changed_digest == baseline_digest:
    raise SystemExit("physical digest ignored one-ULP direct DP mutation")
print("PHYSICAL_DIGEST_ONE_ULP_SENSITIVE=true")

coarse_mutations = {
    "invalid-dp": lambda t: re.sub(r"DP00=\[[^\]]+\]", "DP00=[totally_invalid]", t, count=1),
    "missing-prefix": lambda t: "\n".join(
        line for line in t.splitlines()
        if not line.startswith("PREFIX STRATEGY=canonical-rebox RETURN=4 ")
    ) + "\n",
    "wrong-failure-class": lambda t: replace_once(
        t, "possible_nontransversal_return_to_the_section", "unclassified_failure"
    ),
    "negated-failure-class": lambda t: replace_once(
        t, "possible_nontransversal_return_to_the_section",
        "not_possible_nontransversal_return_to_the_section",
    ),
    "promotion-lie": lambda t: replace_once(t, "PROMOTION_ELIGIBLE=false", "PROMOTION_ELIGIBLE=true"),
    "sixth-liouville-copy": lambda t: t.replace(
        next(line for line in t.splitlines() if line.startswith("SUMMARY ")),
        next(line for line in t.splitlines() if line.startswith("LIOUVILLE RETURN=5 ")).replace(
            "RETURN=5", "RETURN=6", 1
        ) + "\n" + next(line for line in t.splitlines() if line.startswith("SUMMARY ")),
        1,
    ),
}
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name, mutation in coarse_mutations.items():
        candidate = root / f"coarse-{name}.txt"
        candidate.write_text(mutation(coarse), encoding="ascii")
        run = subprocess.run(
            (
                sys.executable, str(verifier), str(candidate),
                "--expect-c0-nontransversal-failure",
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if run.returncode == 0:
            raise SystemExit(f"negative coarse mutation accepted: {name}")
print(f"COARSE_NEGATIVE_MUTATIONS_REJECTED={len(coarse_mutations)}")
PY

grep -Fq 'bounded negative efficiency result' "$document"
grep -Fq 'section-resident C0' "$document"
grep -Fq 'BLK-20260731-cs6-section-resident-c0-reset' "$document"

bash "$repo_root/scripts/ci/cs6_capd_c1_cone_gate.sh" > /dev/null

if [[ "${CS6_C1_RESET_SAMPLE_REPLAY:-0}" == 1 ]]; then
  [[ -n "${CAPD_CONFIG:-}" ]] || { printf 'CAPD_CONFIG is required for replay\n' >&2; exit 64; }
  replay_root="$(mktemp -d)"
  trap 'rm -rf "$replay_root"' EXIT
  bash "$runner" --capd-config "$CAPD_CONFIG" \
    --run-dir "$replay_root/run" --expect-rebox-worse > /dev/null
  bash "$runner" --capd-config "$CAPD_CONFIG" \
    --run-dir "$replay_root/coarse" --source N0 --target N0 \
    --u-index 99 --s-index 37 --u-tiles 200 --s-tiles 75 --order 8 \
    --expect-c0-nontransversal-failure > /dev/null
fi

printf 'CS6_C1_RESET_GATE_PASS=true\n'
