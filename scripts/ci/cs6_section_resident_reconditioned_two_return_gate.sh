#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_probe.cpp"
verifier="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_verify.py"
runner="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_run.sh"
receipt="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_receipt_v1.txt"
provenance="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_provenance_v1.txt"
document="$repo_root/docs/research/cs6_section_resident_reconditioned_two_return_2026-07-31.md"
flattened_receipt="$repo_root/scripts/research/cs6_section_resident_two_return_receipt_v1.txt"
flattened_provenance="$repo_root/scripts/research/cs6_section_resident_two_return_provenance_v1.txt"
flattened_verifier="$repo_root/scripts/research/cs6_section_resident_two_return_verify.py"

for path in "$source_file" "$verifier" "$runner" "$receipt" \
  "$provenance" "$document" "$flattened_receipt" "$flattened_provenance" \
  "$flattened_verifier"; do
  [[ -f "$path" ]] || { printf 'missing gate input: %s\n' "$path" >&2; exit 66; }
done

bash -n "$runner"
python3 - "$verifier" <<'PY'
import sys
from pathlib import Path

compile(Path(sys.argv[1]).read_bytes(), sys.argv[1], "exec")
PY

python3 - "$repo_root" "$source_file" "$verifier" "$runner" \
  "$receipt" "$provenance" "$document" "$flattened_receipt" \
  "$flattened_provenance" "$flattened_verifier" <<'PY'
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(sys.argv[1])
source = Path(sys.argv[2])
verifier = Path(sys.argv[3])
runner = Path(sys.argv[4])
receipt = Path(sys.argv[5])
provenance_path = Path(sys.argv[6])
document = Path(sys.argv[7])
flattened_receipt = Path(sys.argv[8])
flattened_provenance_path = Path(sys.argv[9])
flattened_verifier = Path(sys.argv[10])
sha_re = re.compile(r"^[0-9a-f]{64}$")
expected_flattened_receipt = (
    "14315dd35ada83d13bddaa1c653e0dea86a9da91379559e7f64d69b314077dba"
)
expected_flattened_physical = (
    "536dea89d9f841e0afedaaeb9ef116f5237fb7dd96f7774340850833b5f4b0b1"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kv(path: Path, expected_keys: set[str]) -> dict[str, str]:
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII key/value artifact: {path}") from error
    if not text.endswith("\n") or "\r" in text:
        raise SystemExit(f"noncanonical line endings: {path}")
    result: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.count("=") != 1:
            raise SystemExit(f"bad key/value line {number}: {path}")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise SystemExit(f"bad or duplicate key on line {number}: {path}")
        result[key] = value
    if set(result) != expected_keys:
        missing = sorted(expected_keys - set(result))
        extra = sorted(set(result) - expected_keys)
        raise SystemExit(f"key grammar mismatch for {path}: missing={missing} extra={extra}")
    return result


provenance_keys = {
    "SCHEMA", "SOURCE_SHA256", "VERIFIER_SHA256", "RUNNER_SHA256",
    "DOCUMENT_PATH", "DOCUMENT_SHA256",
    "RECEIPT_PATH", "RECEIPT_SHA256", "INPUT_PATH", "INPUT_SHA256",
    "MANIFEST_PATH", "MANIFEST_SHA256", "VERIFICATION_PATH",
    "VERIFICATION_SHA256", "DEPENDENCIES_PATH", "DEPENDENCIES_SHA256",
    "LINK_INPUTS_PATH", "LINK_INPUTS_SHA256", "RUNTIME_LIBRARIES_PATH",
    "RUNTIME_LIBRARIES_SHA256", "COMPILE_COMMAND_PATH",
    "COMPILE_COMMAND_SHA256", "CAPD_CFLAGS_PATH", "CAPD_CFLAGS_SHA256",
    "CAPD_LIBS_PATH", "CAPD_LIBS_SHA256", "CAPD_VERSION_PATH",
    "CAPD_VERSION_SHA256", "PREPROCESSOR_MACROS_PATH",
    "PREPROCESSOR_MACROS_SHA256", "EFFECTIVE_OPTIONS_PATH",
    "EFFECTIVE_OPTIONS_SHA256", "RUN_CHALLENGE", "PHYSICAL_CHAIN_SHA256",
    "COMPARISON_CHAIN_SHA256",
    "FLATTENED_BASELINE_RECEIPT_PATH",
    "FLATTENED_BASELINE_RECEIPT_SHA256",
    "FLATTENED_BASELINE_PROVENANCE_PATH",
    "FLATTENED_BASELINE_PROVENANCE_SHA256",
    "FLATTENED_BASELINE_VERIFIER_PATH",
    "FLATTENED_BASELINE_VERIFIER_SHA256",
    "FLATTENED_BASELINE_PHYSICAL_SHA256",
    "CAPD_VERSION", "INTERVAL_BACKEND", "DEPENDENCY_CONTENT_HASHES_COMPLETE",
    "EXECUTION_TRUST_MODEL", "REMOTE_ATTESTATION_PRESENT",
    "INDEPENDENT_REPLAY_REQUIRED", "PROMOTION_ELIGIBLE",
}
provenance = read_kv(provenance_path, provenance_keys)
fixed_provenance = {
    "SCHEMA": "sounio.cs6.section-resident-reconditioned-two-return-provenance.v1",
    "DOCUMENT_PATH":
        "docs/research/cs6_section_resident_reconditioned_two_return_2026-07-31.md",
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE": "false",
    "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
    "REMOTE_ATTESTATION_PRESENT": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "PROMOTION_ELIGIBLE": "false",
}
for key, expected in fixed_provenance.items():
    if provenance[key] != expected:
        raise SystemExit(f"provenance {key} mismatch")
for key, value in provenance.items():
    if key.endswith("SHA256") and sha_re.fullmatch(value) is None:
        raise SystemExit(f"malformed provenance hash: {key}")
for key, path in (
    ("SOURCE_SHA256", source),
    ("VERIFIER_SHA256", verifier),
    ("RUNNER_SHA256", runner),
    ("DOCUMENT_SHA256", document),
    ("RECEIPT_SHA256", receipt),
):
    if provenance[key] != digest(path):
        raise SystemExit(f"current artifact does not match provenance: {path.name}")

flattened_artifacts = {
    "FLATTENED_BASELINE_RECEIPT": (
        flattened_receipt,
        "scripts/research/cs6_section_resident_two_return_receipt_v1.txt",
    ),
    "FLATTENED_BASELINE_PROVENANCE": (
        flattened_provenance_path,
        "scripts/research/cs6_section_resident_two_return_provenance_v1.txt",
    ),
    "FLATTENED_BASELINE_VERIFIER": (
        flattened_verifier,
        "scripts/research/cs6_section_resident_two_return_verify.py",
    ),
}
for prefix, (path, expected_path) in flattened_artifacts.items():
    if provenance[f"{prefix}_PATH"] != expected_path:
        raise SystemExit(f"unexpected flattened artifact path: {prefix}")
    if not path.is_file() or provenance[f"{prefix}_SHA256"] != digest(path):
        raise SystemExit(f"flattened artifact hash mismatch: {prefix}")
if provenance["FLATTENED_BASELINE_RECEIPT_SHA256"] != expected_flattened_receipt:
    raise SystemExit("flattened baseline receipt constant mismatch")
if provenance["FLATTENED_BASELINE_PHYSICAL_SHA256"] != expected_flattened_physical:
    raise SystemExit("flattened baseline physical constant mismatch")

artifact_pairs = (
    ("RECEIPT", receipt),
    ("INPUT", None),
    ("MANIFEST", None),
    ("VERIFICATION", None),
    ("DEPENDENCIES", None),
    ("LINK_INPUTS", None),
    ("RUNTIME_LIBRARIES", None),
    ("COMPILE_COMMAND", None),
    ("CAPD_CFLAGS", None),
    ("CAPD_LIBS", None),
    ("CAPD_VERSION", None),
    ("PREPROCESSOR_MACROS", None),
    ("EFFECTIVE_OPTIONS", None),
)
expected_paths = {
    "RECEIPT": "scripts/research/cs6_section_resident_reconditioned_two_return_receipt_v1.txt",
    "INPUT": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_input_v1.txt",
    "MANIFEST": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_run_manifest_v1.txt",
    "VERIFICATION": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_verification_v1.txt",
    "DEPENDENCIES": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_dependencies_v1.sha256",
    "LINK_INPUTS": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_link_inputs_v1.sha256",
    "RUNTIME_LIBRARIES": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_runtime_libraries_v1.sha256",
    "COMPILE_COMMAND": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_compile_command_v1.txt",
    "CAPD_CFLAGS": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_capd_cflags_v1.txt",
    "CAPD_LIBS": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_capd_libs_v1.txt",
    "CAPD_VERSION": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_capd_version_v1.txt",
    "PREPROCESSOR_MACROS": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_preprocessor_macros_v1.txt",
    "EFFECTIVE_OPTIONS": "scripts/research/receipts/cs6_section_resident_reconditioned_two_return_effective_options_v1.txt",
}
resolved: dict[str, Path] = {}
for prefix, fixed_path in artifact_pairs:
    if provenance[f"{prefix}_PATH"] != expected_paths[prefix]:
        raise SystemExit(f"unexpected retained artifact path: {prefix}")
    path = fixed_path if fixed_path is not None else repo / provenance[f"{prefix}_PATH"]
    try:
        path.resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise SystemExit(f"provenance path escapes repository: {path}") from error
    if not path.is_file() or digest(path) != provenance[f"{prefix}_SHA256"]:
        raise SystemExit(f"retained artifact hash mismatch: {prefix}")
    resolved[prefix] = path

manifest_keys = {
    "MANIFEST_KIND", "RUN_COMPLETE", "WORKER_EXIT", "SOURCE_SHA256",
    "VERIFIER_SHA256", "RUNNER_SHA256", "INPUT_SHA256", "RUN_CHALLENGE",
    "EXECUTABLE_SHA256", "RECEIPT_SHA256", "VERIFICATION_SHA256",
    "PHYSICAL_CHAIN_SHA256", "COMPARISON_CHAIN_SHA256",
    "FLATTENED_BASELINE_RECEIPT_SHA256",
    "FLATTENED_BASELINE_PROVENANCE_SHA256",
    "FLATTENED_BASELINE_VERIFIER_SHA256",
    "FLATTENED_BASELINE_PHYSICAL_SHA256",
    "CAPD_CONFIG_SHA256", "CAPD_CFLAGS_SHA256", "CAPD_LIBS_SHA256",
    "CAPD_VERSION_SHA256", "PREPROCESSOR_MACROS_SHA256",
    "EFFECTIVE_OPTIONS_SHA256", "EFFECTIVE_OPTIONS_STDERR_SHA256",
    "DEPENDENCY_PATHS_SHA256", "DEPENDENCIES_SHA256", "LINK_INPUTS_SHA256",
    "RUNTIME_LIBRARIES_SHA256", "COMPILER_SHA256", "COMPILER_VERSION_SHA256",
    "PYTHON_SHA256", "PYTHON_VERSION_SHA256", "RUNTIME_LINKAGE_SHA256",
    "COMPILE_COMMAND_SHA256", "COMPILE_STDERR_SHA256", "PROBE_STDERR_SHA256",
    "GIT_HEAD", "GIT_STATUS_CLEAN", "CAPD_VERSION", "INTERVAL_BACKEND",
    "OPTIMIZATION_LEVEL",
    "ROUNDING_MATH_EFFECTIVE", "DEPENDENCIES_STABLE_DURING_RUN",
    "FLATTENED_BASELINE_STABLE_DURING_RUN",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE", "EXECUTION_PROVENANCE_ATTESTED",
    "INDEPENDENT_REPLAY_REQUIRED", "PROMOTION_ELIGIBLE",
}
manifest = read_kv(resolved["MANIFEST"], manifest_keys)
for key, value in manifest.items():
    if key.endswith("SHA256") and sha_re.fullmatch(value) is None:
        raise SystemExit(f"malformed retained manifest hash: {key}")
fixed_manifest = {
    "MANIFEST_KIND": "CS6_SECTION_RESIDENT_RECONDITIONED_TWO_RETURN_V1",
    "RUN_COMPLETE": "true", "WORKER_EXIT": "0", "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB", "OPTIMIZATION_LEVEL": "O0",
    "ROUNDING_MATH_EFFECTIVE": "true",
    "DEPENDENCIES_STABLE_DURING_RUN": "true",
    "FLATTENED_BASELINE_STABLE_DURING_RUN": "true",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE": "false",
    "EXECUTION_PROVENANCE_ATTESTED": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true", "PROMOTION_ELIGIBLE": "false",
}
for key, expected in fixed_manifest.items():
    if manifest[key] != expected:
        raise SystemExit(f"retained manifest {key} mismatch")
manifest_bindings = {
    "SOURCE_SHA256": provenance["SOURCE_SHA256"],
    "VERIFIER_SHA256": provenance["VERIFIER_SHA256"],
    "RUNNER_SHA256": provenance["RUNNER_SHA256"],
    "INPUT_SHA256": provenance["INPUT_SHA256"],
    "RUN_CHALLENGE": provenance["RUN_CHALLENGE"],
    "RECEIPT_SHA256": provenance["RECEIPT_SHA256"],
    "VERIFICATION_SHA256": provenance["VERIFICATION_SHA256"],
    "PHYSICAL_CHAIN_SHA256": provenance["PHYSICAL_CHAIN_SHA256"],
    "COMPARISON_CHAIN_SHA256": provenance["COMPARISON_CHAIN_SHA256"],
    "FLATTENED_BASELINE_RECEIPT_SHA256":
        provenance["FLATTENED_BASELINE_RECEIPT_SHA256"],
    "FLATTENED_BASELINE_PROVENANCE_SHA256":
        provenance["FLATTENED_BASELINE_PROVENANCE_SHA256"],
    "FLATTENED_BASELINE_VERIFIER_SHA256":
        provenance["FLATTENED_BASELINE_VERIFIER_SHA256"],
    "FLATTENED_BASELINE_PHYSICAL_SHA256":
        provenance["FLATTENED_BASELINE_PHYSICAL_SHA256"],
    "CAPD_CFLAGS_SHA256": provenance["CAPD_CFLAGS_SHA256"],
    "CAPD_LIBS_SHA256": provenance["CAPD_LIBS_SHA256"],
    "CAPD_VERSION_SHA256": provenance["CAPD_VERSION_SHA256"],
    "PREPROCESSOR_MACROS_SHA256": provenance["PREPROCESSOR_MACROS_SHA256"],
    "EFFECTIVE_OPTIONS_SHA256": provenance["EFFECTIVE_OPTIONS_SHA256"],
    "DEPENDENCIES_SHA256": provenance["DEPENDENCIES_SHA256"],
    "LINK_INPUTS_SHA256": provenance["LINK_INPUTS_SHA256"],
    "RUNTIME_LIBRARIES_SHA256": provenance["RUNTIME_LIBRARIES_SHA256"],
    "COMPILE_COMMAND_SHA256": provenance["COMPILE_COMMAND_SHA256"],
}
for key, expected in manifest_bindings.items():
    if manifest[key] != expected:
        raise SystemExit(f"manifest/provenance binding mismatch: {key}")

expected_input = (
    "INPUT_SCHEMA=sounio.cs6.section-resident-reconditioned-two-return-input.v1\n"
    "SOURCE=N0\n"
    "U_INDEX=20000\n"
    "S_INDEX=15000\n"
    "U_TILES=40000\n"
    "S_TILES=30000\n"
    "ORDER=8\n"
    "RETURN_COUNT=2\n"
    "SECTION=COORDINATE_W_EQUALS_ZERO\n"
    "CROSSING_DIRECTION=MINUS_PLUS\n"
    "VECTOR_FIELD=CS6_FROZEN_22.3274637391\n"
    "TANGENT_GAUGES=IDENTITY,MIDPOINT_M,ORIENTED_QR\n"
).encode("ascii")
if resolved["INPUT"].read_bytes() != expected_input:
    raise SystemExit("retained input grammar or value mismatch")


def verify_content_manifest(path: Path, *, allow_bundle_source: bool) -> int:
    data = path.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        raise SystemExit(f"noncanonical content manifest: {path}")
    seen: set[str] = set()
    count = 0
    for number, line in enumerate(data.decode("ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SystemExit(f"malformed content manifest line {number}: {path}")
        expected, name = match.groups()
        if name in seen:
            raise SystemExit(f"duplicate content-manifest path: {name}")
        seen.add(name)
        if name == "BUNDLE/probe-source.cpp" and allow_bundle_source:
            target = source
        elif name.startswith("BUNDLE/"):
            raise SystemExit(f"unsupported bundle-relative dependency: {name}")
        else:
            target = Path(name)
        if not target.is_file() or digest(target) != expected:
            raise SystemExit(f"dependency content mismatch: {name}")
        count += 1
    if count == 0:
        raise SystemExit(f"empty content manifest: {path}")
    if allow_bundle_source and "BUNDLE/probe-source.cpp" not in seen:
        raise SystemExit("dependency manifest omits the compiled source snapshot")
    return count


dependency_entries = verify_content_manifest(
    resolved["DEPENDENCIES"], allow_bundle_source=True
)
link_entries = verify_content_manifest(resolved["LINK_INPUTS"], allow_bundle_source=False)
runtime_entries = verify_content_manifest(
    resolved["RUNTIME_LIBRARIES"], allow_bundle_source=False
)

verification_lines = tuple(
    resolved["VERIFICATION"].read_text(encoding="ascii").splitlines()
)
required_verification = (
    "VERIFY_SCHEMA=sounio.cs6.section-resident-reconditioned-two-return-verification.v1",
    "VERIFY_PASS=true", "SOURCE=N0", "TILE=20000,15000/40000,30000",
    "RETURN_COUNT=2", "RAW_C0_RECONSTRUCTED=true", "RAW_C1_RECONSTRUCTED=true",
    "POINCARE_DP_RECOMPUTED=true", "POINCARE_DP_CONTAINS_RECOMPUTATION=true",
    "COMPOSITION_EXACT_RECOMPUTED=true",
    "REVERSED_ORDER_EXACT_RECOMPUTED=true", "EXP_ELL_RECOMPUTED=true",
    f"PHYSICAL_CHAIN_SHA256={provenance['PHYSICAL_CHAIN_SHA256']}",
    f"FLATTENED_PHYSICAL_CHAIN_SHA256={expected_flattened_physical}",
    f"COMPARISON_CHAIN_SHA256={provenance['COMPARISON_CHAIN_SHA256']}",
    "MEAN_VALUE_C0_RECOMPUTED=true",
    "GAUGE_TRANSITIONS_RECOMPUTED=true",
    "FIXED_FRAME_COMPOSITIONS_RECOMPUTED=true",
    "CORRELATED_STATE_COMPONENTWISE_NARROWER=true",
    "IDENTITY_DETERMINANT_CROSSES_ZERO=true",
    "MIDPOINT_M_DETERMINANT_CROSSES_ZERO=true",
    "ORIENTED_QR_DETERMINANT_CROSSES_ZERO=true",
    "ANY_GAUGE_SIGN_DEFINITE=false",
    "LIOUVILLE_DETERMINANT_NEGATIVE=true",
    "PROMOTION_ELIGIBLE=false",
)
if verification_lines != required_verification:
    raise SystemExit("retained verification grammar or content mismatch")

baseline = receipt.read_bytes()
expected_args = {
    "source": provenance["SOURCE_SHA256"],
    "input": provenance["INPUT_SHA256"],
    "challenge": provenance["RUN_CHALLENGE"],
}


def invoke(data: bytes, *, source_hash: str | None = None,
           input_hash: str | None = None, challenge: str | None = None,
           receipt_hash: str | None = None,
           baseline_receipt_hash: str | None = None,
           flattened_physical_hash: str | None = None,
           verifier_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as candidate:
        candidate.write(data)
        candidate.flush()
        return subprocess.run(
            (
                sys.executable, str(verifier_path or verifier), candidate.name,
                "--expected-source-sha256", source_hash or expected_args["source"],
                "--expected-input-sha256", input_hash or expected_args["input"],
                "--expected-run-challenge", challenge or expected_args["challenge"],
                "--expected-receipt-sha256",
                receipt_hash or hashlib.sha256(data).hexdigest(),
                "--expected-baseline-receipt-sha256",
                baseline_receipt_hash or expected_flattened_receipt,
                "--expected-flattened-physical-sha256",
                flattened_physical_hash or expected_flattened_physical,
            ), capture_output=True, text=True, check=False,
        )


def rejected(name: str, data: bytes, **kwargs: str) -> None:
    result = invoke(data, **kwargs)
    if result.returncode != 2 or not result.stderr.startswith("VERIFY_ERROR="):
        raise SystemExit(
            f"negative mutation did not fail closed: {name}, rc={result.returncode}"
        )


flattened_provenance_keys = {
    "SCHEMA", "SOURCE_SHA256", "VERIFIER_SHA256", "RUNNER_SHA256",
    "RECEIPT_PATH", "RECEIPT_SHA256", "INPUT_PATH", "INPUT_SHA256",
    "MANIFEST_PATH", "MANIFEST_SHA256", "VERIFICATION_PATH",
    "VERIFICATION_SHA256", "DEPENDENCIES_PATH", "DEPENDENCIES_SHA256",
    "LINK_INPUTS_PATH", "LINK_INPUTS_SHA256", "RUNTIME_LIBRARIES_PATH",
    "RUNTIME_LIBRARIES_SHA256", "COMPILE_COMMAND_PATH",
    "COMPILE_COMMAND_SHA256", "CAPD_CFLAGS_PATH", "CAPD_CFLAGS_SHA256",
    "CAPD_LIBS_PATH", "CAPD_LIBS_SHA256", "CAPD_VERSION_PATH",
    "CAPD_VERSION_SHA256", "PREPROCESSOR_MACROS_PATH",
    "PREPROCESSOR_MACROS_SHA256", "EFFECTIVE_OPTIONS_PATH",
    "EFFECTIVE_OPTIONS_SHA256", "RUN_CHALLENGE", "PHYSICAL_CHAIN_SHA256",
    "CAPD_VERSION", "INTERVAL_BACKEND", "DEPENDENCY_CONTENT_HASHES_COMPLETE",
    "EXECUTION_TRUST_MODEL", "REMOTE_ATTESTATION_PRESENT",
    "INDEPENDENT_REPLAY_REQUIRED", "PROMOTION_ELIGIBLE",
}
flattened_provenance = read_kv(
    flattened_provenance_path, flattened_provenance_keys
)
flattened_fixed = {
    "SCHEMA": "sounio.cs6.section-resident-two-return-provenance.v1",
    "RECEIPT_PATH":
        "scripts/research/cs6_section_resident_two_return_receipt_v1.txt",
    "RECEIPT_SHA256": expected_flattened_receipt,
    "VERIFIER_SHA256": digest(flattened_verifier),
    "PHYSICAL_CHAIN_SHA256": expected_flattened_physical,
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE": "false",
    "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
    "REMOTE_ATTESTATION_PRESENT": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "PROMOTION_ELIGIBLE": "false",
}
for key, expected in flattened_fixed.items():
    if flattened_provenance[key] != expected:
        raise SystemExit(f"flattened provenance {key} mismatch")
if digest(flattened_receipt) != expected_flattened_receipt:
    raise SystemExit("retained flattened receipt hash mismatch")
flattened_result = subprocess.run(
    (
        sys.executable, str(flattened_verifier), str(flattened_receipt),
        "--expected-source-sha256", flattened_provenance["SOURCE_SHA256"],
        "--expected-input-sha256", flattened_provenance["INPUT_SHA256"],
        "--expected-run-challenge", flattened_provenance["RUN_CHALLENGE"],
        "--expected-receipt-sha256", expected_flattened_receipt,
    ), capture_output=True, text=True, check=False,
)
if flattened_result.returncode != 0:
    raise SystemExit("retained flattened receipt no longer verifies")
if flattened_result.stdout.count(
    f"PHYSICAL_CHAIN_SHA256={expected_flattened_physical}\n"
) != 1:
    raise SystemExit("retained flattened verifier physical digest mismatch")


baseline_result = invoke(baseline)
if baseline_result.returncode != 0:
    raise SystemExit("retained receipt no longer verifies")
if tuple(baseline_result.stdout.splitlines()) != required_verification:
    raise SystemExit("retained receipt verification output is not canonical")


def text_mutation(transform) -> bytes:
    text = baseline.decode("ascii")
    changed = transform(text)
    if changed == text:
        raise RuntimeError("mutation made no change")
    return changed.encode("ascii")


def replace_once(old: str, new: str):
    def transform(text: str) -> str:
        if text.count(old) != 1:
            raise RuntimeError(f"non-unique mutation anchor: {old}")
        return text.replace(old, new, 1)
    return transform


def mutate_record(marker: str, key: str, value: str):
    def transform(text: str) -> str:
        lines = text.splitlines()
        matches = [index for index, line in enumerate(lines)
                   if line.startswith(marker + " ")]
        if len(matches) != 1:
            raise RuntimeError(f"record anchor mismatch: {marker}")
        index = matches[0]
        pattern = re.compile(rf"(?<![^ ]){re.escape(key)}=[^ ]+")
        lines[index], count = pattern.subn(f"{key}={value}", lines[index], count=1)
        if count != 1:
            raise RuntimeError(f"token anchor mismatch: {marker}.{key}")
        return "\n".join(lines) + "\n"
    return transform


zero = "[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]"
one = "[0x1.fffffffffffffp-1,0x1.0000000000001p+0]"


def mutate_records(edits: list[tuple[str, str, str]]):
    def transform(text: str) -> str:
        lines = text.splitlines()
        for marker, key, value in edits:
            matches = [index for index, line in enumerate(lines)
                       if line.startswith(marker + " ")]
            if len(matches) != 1:
                raise RuntimeError(f"record anchor mismatch: {marker}")
            index = matches[0]
            pattern = re.compile(rf"(?<![^ ]){re.escape(key)}=[^ ]+")
            lines[index], count = pattern.subn(
                f"{key}={value}", lines[index], count=1
            )
            if count != 1:
                raise RuntimeError(f"token anchor mismatch: {marker}.{key}")
        return "\n".join(lines) + "\n"
    return transform


def shift_binary_exponent(value: str, amount: int) -> str:
    if value == zero:
        return value

    def shift_endpoint(match: re.Match[str]) -> str:
        exponent = int(match.group(1))
        return f"p{exponent + amount:+d}"

    shifted, count = re.subn(r"p([+-][0-9]+)", shift_endpoint, value)
    if count != 2:
        raise RuntimeError(f"cannot dyadically scale interval: {value}")
    return shifted


def mutate_scaled_records(
    edits: list[tuple[str, str, int]]
):
    def transform(text: str) -> str:
        lines = text.splitlines()
        for marker, key, exponent_shift in edits:
            matches = [index for index, line in enumerate(lines)
                       if line.startswith(marker + " ")]
            if len(matches) != 1:
                raise RuntimeError(f"record anchor mismatch: {marker}")
            index = matches[0]
            pattern = re.compile(
                rf"(?<![^ ]){re.escape(key)}=(\[[^ ]+\])"
            )

            def replacement(match: re.Match[str]) -> str:
                return (
                    f"{key}="
                    f"{shift_binary_exponent(match.group(1), exponent_shift)}"
                )

            lines[index], count = pattern.subn(
                replacement, lines[index], count=1
            )
            if count != 1:
                raise RuntimeError(f"token anchor mismatch: {marker}.{key}")
        return "\n".join(lines) + "\n"
    return transform


lines = baseline.decode("ascii").splitlines()
record_tokens = {
    line.split(" ", 1)[0]: {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in line.split(" ")[1:]
    }
    for line in lines if " " in line
}
p1_time = record_tokens["LOCAL_P1"]["TIME"]
p2_time = record_tokens["LOCAL_P2"]["TIME"]
reverse_product = [
    ("COMPOSED_P2", f"DP{row}{column}",
     record_tokens["COMPOSED_P2"][f"REVERSED_DP{row}{column}"])
    for row in range(3) for column in range(3)
]
fabricated_direct = [
    ("COMPOSED_P2", f"DP{row}{column}",
     record_tokens["DIRECT_P2"][f"SECTION_DP{row}{column}"])
    for row in range(3) for column in range(3)
]
swap_headers = lines.copy()
swap_headers[7], swap_headers[8] = swap_headers[8], swap_headers[7]
swap_records = lines.copy()
source_record_index = next(
    index for index, line in enumerate(lines) if line.startswith("SOURCE_TILE ")
)
swap_records[source_record_index], swap_records[source_record_index + 1] = (
    swap_records[source_record_index + 1], swap_records[source_record_index]
)

mutations = {
    "header-order": ("\n".join(swap_headers) + "\n").encode("ascii"),
    "record-order": ("\n".join(swap_records) + "\n").encode("ascii"),
    "missing-line": ("\n".join(lines[:-1]) + "\n").encode("ascii"),
    "blank-line": baseline.replace(b"SOURCE=N0\n", b"SOURCE=N0\n\n", 1),
    "crlf": baseline.replace(b"\n", b"\r\n"),
    "non-ascii": baseline.replace(b"SOURCE=N0", "SOURCE=N\u2080".encode("utf-8"), 1),
    "extra-line": baseline + b"UNKNOWN=true\n",
    "source-binding": text_mutation(replace_once(
        f"WORKER_SOURCE_SHA256={expected_args['source']}",
        "WORKER_SOURCE_SHA256=" + "0" * 64,
    )),
    "input-binding": text_mutation(replace_once(
        f"INPUT_SHA256={expected_args['input']}", "INPUT_SHA256=" + "0" * 64,
    )),
    "challenge-binding": text_mutation(replace_once(
        f"RUN_CHALLENGE={expected_args['challenge']}", "RUN_CHALLENGE=" + "0" * 64,
    )),
    "flattened-receipt-header-binding": text_mutation(replace_once(
        f"FLATTENED_BASELINE_RECEIPT_SHA256={expected_flattened_receipt}",
        "FLATTENED_BASELINE_RECEIPT_SHA256=" + "0" * 64,
    )),
    "flattened-physical-header-binding": text_mutation(replace_once(
        "FLATTENED_BASELINE_PHYSICAL_CHAIN_SHA256="
        f"{expected_flattened_physical}",
        "FLATTENED_BASELINE_PHYSICAL_CHAIN_SHA256=" + "0" * 64,
    )),
    "tangent-gauge-header": text_mutation(replace_once(
        "TANGENT_GAUGES=IDENTITY,MIDPOINT_M,ORIENTED_QR",
        "TANGENT_GAUGES=IDENTITY,ORIENTED_QR",
    )),
    "scientific-class-lie": text_mutation(replace_once(
        "SCIENTIFIC_RESULT_CLASS=CORRELATION_PRESERVED_ORIENTATION_UNRESOLVED",
        "SCIENTIFIC_RESULT_CLASS=ORIENTATION_CERTIFIED",
    )),
    "promotion-lie": text_mutation(replace_once(
        "PROMOTION_ELIGIBLE=false", "PROMOTION_ELIGIBLE=true",
    )),
    "full-source-lie": text_mutation(replace_once(
        "FULL_SOURCE_CARRIER_PROVED=false", "FULL_SOURCE_CARRIER_PROVED=true",
    )),
    "nonautonomous-lie": text_mutation(replace_once(
        "NONAUTONOMOUS_GENERALIZATION_PROVED=false",
        "NONAUTONOMOUS_GENERALIZATION_PROVED=true",
    )),
    "summary-false": text_mutation(replace_once(
        "PROBE_PASS=true", "PROBE_PASS=false",
    )),
    "source-tile": text_mutation(mutate_record("SOURCE_TILE", "SOURCE_U", one)),
    "source-q0-normal": text_mutation(mutate_record("SOURCE_TILE", "Q022", one)),
    "direct1-flow": text_mutation(mutate_record("DIRECT_P1", "FLOW_TANGENT00", one)),
    "direct1-dp": text_mutation(mutate_record("DIRECT_P1", "DP00", one)),
    "direct2-dp": text_mutation(mutate_record("DIRECT_P2", "DP00", one)),
    "direct2-section-dp": text_mutation(mutate_record(
        "DIRECT_P2", "SECTION_DP00", one,
    )),
    "local1-dp": text_mutation(mutate_record("LOCAL_P1", "DP00", one)),
    "local2-flow": text_mutation(mutate_record("LOCAL_P2", "FLOW_TANGENT00", one)),
    "local2-dp": text_mutation(mutate_record("LOCAL_P2", "DP00", one)),
    "local2-section-dp": text_mutation(mutate_record(
        "LOCAL_P2", "SECTION_DP00", one,
    )),
    "reverse-product": text_mutation(mutate_records(reverse_product)),
    "fabricated-direct-product": text_mutation(mutate_records(fabricated_direct)),
    "j1-after-compose": text_mutation(mutate_record("COMPOSED_P2", "J100", one)),
    "j2-after-compose": text_mutation(mutate_record(
        "COMPOSED_P2", "J2_LOCAL00", one,
    )),
    "composed-det": text_mutation(mutate_record("COMPOSED_P2", "DET", one)),
    "event1-c0-raw": text_mutation(mutate_record(
        "EVENT1_CARRIER", "C0_X0", one,
    )),
    "event1-c0-normal": text_mutation(mutate_record(
        "EVENT1_CARRIER", "C0_HULL2", one,
    )),
    "event1-c1-raw": text_mutation(mutate_record(
        "EVENT1_CARRIER", "C1_D00", one,
    )),
    "continuation1-seed": text_mutation(mutate_record(
        "CONTINUATION1_CARRIER", "C1_D00", zero,
    )),
    "continuation1-normal": text_mutation(mutate_record(
        "CONTINUATION1_CARRIER", "C1_D22", one,
    )),
    "incoming-j1": text_mutation(mutate_record(
        "CONTINUATION1_CARRIER", "INCOMING_J100", one,
    )),
    "event2-c0-raw": text_mutation(mutate_record(
        "EVENT2_CARRIER", "C0_X0", one,
    )),
    "event2-c1-local": text_mutation(mutate_record(
        "EVENT2_CARRIER", "C1_D00", one,
    )),
    "continuation2-seed": text_mutation(mutate_record(
        "CONTINUATION2_CARRIER", "C1_D00", zero,
    )),
    "incoming-j2-local": text_mutation(mutate_record(
        "CONTINUATION2_CARRIER", "INCOMING_J2_LOCAL00", one,
    )),
    "incoming-composed": text_mutation(mutate_record(
        "CONTINUATION2_CARRIER", "INCOMING_COMPOSED_P200", one,
    )),
    "local2-time-reuse": text_mutation(mutate_record(
        "LOCAL_P2", "TIME", p1_time,
    )),
    "direct2-time-reuse": text_mutation(mutate_record(
        "DIRECT_P2", "TIME", p1_time,
    )),
    "event2-time-reuse": text_mutation(mutate_record(
        "EVENT2_CARRIER", "TIME", p1_time,
    )),
    "liouville2-time-reuse": text_mutation(mutate_record(
        "LIOUVILLE_P2", "TIME", p1_time,
    )),
    "duration-zero": text_mutation(mutate_record("LOCAL_P2", "DURATION", zero)),
    "post2-time": text_mutation(mutate_record("POSTSECTION2", "TIME", p2_time)),
    "post2-x2": text_mutation(mutate_record("POSTSECTION2", "X2", zero)),
    "post2-sign": text_mutation(mutate_record("POSTSECTION2", "SECTION_SIGN", zero)),
    "direct2-nu": text_mutation(mutate_record("DIRECT_P2", "NU", zero)),
    "ell-x3-coordinated-zero": text_mutation(mutate_records([
        ("LIOUVILLE_P2", "X3", zero), ("LIOUVILLE_P2", "ELL", zero),
    ])),
    "exp-only": text_mutation(mutate_record("LIOUVILLE_P2", "EXP_ELL", one)),
    "liouville-nu0": text_mutation(mutate_record("LIOUVILLE_P2", "NU0", one)),
    "liouville-nu2": text_mutation(mutate_record("LIOUVILLE_P2", "NU2", one)),
    "liouville-det": text_mutation(mutate_record("LIOUVILLE_P2", "DET", one)),
    "unknown-marker": text_mutation(replace_once("SUMMARY ", "SUMMERY ")),
}


def different_value(marker: str, key: str) -> str:
    return one if record_tokens[marker][key] == zero else zero


critical_fields = {
    "MIDPOINT_P1": ("TIME", "SECTION_X0"),
    "MEAN_VALUE_C0": (
        "CENTER0", "NORMALIZED_DELTA0", "NORMALIZED_DELTA1",
        "M00", "M01", "M10", "M11", "RESIDUAL_BASIS00",
        "CENTER_ERROR0", "LINEARIZATION_ERROR0", "RESIDUAL0",
    ),
    "RECONDITIONED_EVENT1_CARRIER": (
        "C0_X0", "C0_C00", "C0_R00", "C0_B00", "C0_R0",
        "C1_D00", "C0_HULL0", "C1_HULL00",
    ),
    "GAUGE_IDENTITY": (
        "BASIS00", "INVERSE_BASIS00", "TRANSITION00",
        "BASIS_TIMES_INVERSE00", "INVERSE_TIMES_BASIS00",
        "BASIS_TIMES_TRANSITION00",
    ),
    "GAUGE_MIDPOINT_M": (
        "BASIS00", "INVERSE_BASIS00", "TRANSITION00",
        "BASIS_TIMES_INVERSE00", "INVERSE_TIMES_BASIS00",
        "BASIS_TIMES_TRANSITION00",
    ),
    "GAUGE_ORIENTED_QR": (
        "BASIS00", "INVERSE_BASIS00", "TRANSITION00",
        "BASIS_TIMES_INVERSE00", "INVERSE_TIMES_BASIS00",
        "BASIS_TIMES_TRANSITION00",
    ),
    "GAUGE_IDENTITY_CONTINUATION1": (
        "C0_C00", "C0_R00", "C1_D00", "C0_HULL0",
        "C1_HULL00", "INCOMING_J100",
    ),
    "GAUGE_MIDPOINT_M_CONTINUATION1": (
        "C0_C00", "C0_R00", "C1_D00", "C0_HULL0",
        "C1_HULL00", "INCOMING_J100",
    ),
    "GAUGE_ORIENTED_QR_CONTINUATION1": (
        "C0_C00", "C0_R00", "C1_D00", "C0_HULL0",
        "C1_HULL00", "INCOMING_J100",
    ),
    "GAUGE_IDENTITY_LOCAL_P2": (
        "TIME", "DURATION", "FLOW_TANGENT00", "DP00",
        "SECTION_X0", "SECTION_DP00", "NU", "DET_IN_BASIS",
    ),
    "GAUGE_MIDPOINT_M_LOCAL_P2": (
        "TIME", "DURATION", "FLOW_TANGENT00", "DP00",
        "SECTION_X0", "SECTION_DP00", "NU", "DET_IN_BASIS",
    ),
    "GAUGE_ORIENTED_QR_LOCAL_P2": (
        "TIME", "DURATION", "FLOW_TANGENT00", "DP00",
        "SECTION_X0", "SECTION_DP00", "NU", "DET_IN_BASIS",
    ),
    "GAUGE_IDENTITY_COMPOSED_P2": (
        "J2_BASIS00", "TRANSITION00", "DP_FIXED_Q000",
        "DET_FIXED_Q0",
    ),
    "GAUGE_MIDPOINT_M_COMPOSED_P2": (
        "J2_BASIS00", "TRANSITION00", "DP_FIXED_Q000",
        "DET_FIXED_Q0",
    ),
    "GAUGE_ORIENTED_QR_COMPOSED_P2": (
        "J2_BASIS00", "TRANSITION00", "DP_FIXED_Q000",
        "DET_FIXED_Q0",
    ),
    "GAUGE_IDENTITY_POSTSECTION2": ("TIME", "X2", "SECTION_SIGN"),
    "GAUGE_MIDPOINT_M_POSTSECTION2": ("TIME", "X2", "SECTION_SIGN"),
    "GAUGE_ORIENTED_QR_POSTSECTION2": ("TIME", "X2", "SECTION_SIGN"),
}
for marker, keys in critical_fields.items():
    for key in keys:
        name = f"{marker.lower()}-{key.lower()}"
        if name in mutations:
            raise RuntimeError(f"duplicate mutation name: {name}")
        mutations[name] = text_mutation(
            mutate_record(marker, key, different_value(marker, key))
        )

correlated_carriers = (
    "RECONDITIONED_EVENT1_CARRIER",
    "GAUGE_IDENTITY_CONTINUATION1",
    "GAUGE_MIDPOINT_M_CONTINUATION1",
    "GAUGE_ORIENTED_QR_CONTINUATION1",
)
mean_value_scale_edits: list[tuple[str, str, int]] = [
    ("MEAN_VALUE_C0", f"M{row}{column}", 1)
    for row in range(2) for column in range(2)
]
mean_value_scale_edits.extend(
    ("MEAN_VALUE_C0", f"NORMALIZED_DELTA{row}", -1)
    for row in range(2)
)
for marker in correlated_carriers:
    mean_value_scale_edits.extend(
        (marker, f"C0_C{row}{column}", 1)
        for row in range(2) for column in range(2)
    )
    mean_value_scale_edits.extend(
        (marker, f"C0_R0{row}", -1) for row in range(2)
    )
mutations["mean-value-coordinated-c-r0-rescaling"] = text_mutation(
    mutate_scaled_records(mean_value_scale_edits)
)

qr_rescaling_edits: list[tuple[str, str, int]] = []
for row in range(2):
    for column in range(2):
        qr_rescaling_edits.extend((
            ("GAUGE_ORIENTED_QR", f"BASIS{row}{column}", 1),
            ("GAUGE_ORIENTED_QR", f"INVERSE_BASIS{row}{column}", -1),
            ("GAUGE_ORIENTED_QR", f"TRANSITION{row}{column}", -1),
            (
                "GAUGE_ORIENTED_QR_CONTINUATION1",
                f"C1_D{row}{column}", 1,
            ),
            (
                "GAUGE_ORIENTED_QR_CONTINUATION1",
                f"C1_HULL{row}{column}", 1,
            ),
            (
                "GAUGE_ORIENTED_QR_LOCAL_P2",
                f"FLOW_TANGENT{row}{column}", 1,
            ),
            ("GAUGE_ORIENTED_QR_LOCAL_P2", f"DP{row}{column}", 1),
            (
                "GAUGE_ORIENTED_QR_LOCAL_P2",
                f"SECTION_DP{row}{column}", 1,
            ),
            (
                "GAUGE_ORIENTED_QR_COMPOSED_P2",
                f"J2_BASIS{row}{column}", 1,
            ),
            (
                "GAUGE_ORIENTED_QR_COMPOSED_P2",
                f"TRANSITION{row}{column}", -1,
            ),
        ))
qr_rescaling_edits.append(
    ("GAUGE_ORIENTED_QR_LOCAL_P2", "DET_IN_BASIS", 2)
)
mutations["qr-coordinated-tangent-gauge-rescaling"] = text_mutation(
    mutate_scaled_records(qr_rescaling_edits)
)

for key in (
    "ANY_GAUGE_SIGN_DEFINITE",
    "LIOUVILLE_DETERMINANT_NEGATIVE",
    "CORRELATED_STATE_COMPONENTWISE_NARROWER",
    "CERTIFICATE_PASS",
):
    value = record_tokens["SUMMARY"][key]
    if value not in {"true", "false"}:
        raise RuntimeError(f"non-boolean summary value: {key}")
    mutations[f"summary-{key.lower()}"] = text_mutation(
        mutate_record(
            "SUMMARY", key, "false" if value == "true" else "true"
        )
    )

for name, data in mutations.items():
    rejected(name, data)

rejected("expected-source", baseline, source_hash="f" * 64)
rejected("expected-input", baseline, input_hash="f" * 64)
rejected("expected-challenge", baseline, challenge="f" * 64)
rejected("expected-receipt", baseline, receipt_hash="f" * 64)
rejected(
    "expected-baseline-receipt", baseline,
    baseline_receipt_hash="f" * 64,
)
rejected(
    "expected-flattened-physical", baseline,
    flattened_physical_hash="f" * 64,
)
negative_count = len(mutations) + 6
if negative_count < 80:
    raise SystemExit(f"insufficient negative-mutation coverage: {negative_count}")
print(f"NEGATIVE_MUTATIONS_REJECTED={negative_count}")

source_text = source.read_text(encoding="ascii")
required_source = (
    "class SectionResidentMap : public IPoincareMap",
    "this->integrateUntilSectionCrossing(before, after, 1);",
    "this->crossSectionInOneStep(before, after, local_time,",
    "this->sectionDerivativesEnclosure.computeOneStepSectionEnclosure(",
    "const ReturnData direct2 = direct_return(input, 2);",
    "C1Rect2Set continuation1_carrier(event1_c0, seed_c1, local1.time);",
    "const ReturnData local2 = resident_return(input, continuation1_carrier);",
    "const IMatrix composed_dp2 = local_section_dp2 * local_section_dp1;",
    "const IVector normalized_delta1 = input.normalized_tile_delta();",
    "const IMatrix event_affine_basis1 =",
    "midpoint_tangent_basis(local_section_dp1);",
    "const IMatrix reconditioned_basis1 =",
    "oriented_qr_tangent_basis(local_section_dp1);",
    "const IMatrix transition1 = inverse_basis1 * local_section_dp1;",
    "const IMatrix affine_transition1 =",
    "inverse_event_affine_basis1 * local_section_dp1;",
    "reconditioned_continuation1_carrier.setC0Factor(",
    "correlated_identity_continuation1_carrier.setC0Factor(",
    "correlated_affine_continuation1_carrier.setC0Factor(",
    "std::numeric_limits<double>::infinity());",
    '<< "DIRECT_FLOW_TANGENT_ROLE=D_FLOW_TIMES_Q0\\n"',
    '<< "FLAT_LOCAL_P2_FLOW_TANGENT_ROLE=D_FLOW_LOCAL_TIMES_SECTION_IDENTITY\\n"',
    '<< "GAUGE_FLOW_TANGENT_ROLE=D_FLOW_LOCAL_TIMES_GAUGE_BASIS\\n"',
    '<< "WIDTH_COMPARISON_FRAME=FIXED_SOURCE_Q0_COORDINATES\\n"',
    '<< "SOURCE_TANGENT_SEED_ROLE=GLOBAL_FRAME_RADII_WITH_ZERO_DUMMY_NORMAL\\n"',
    '<< "COMPOSITION_ORDER=J2_LOCAL_TIMES_J1_IN\\n"',
    '<< "GAUGE_COMPOSITION_ORDER=J2_BASIS_TIMES_BASIS_INVERSE_TIMES_J1_IN\\n"',
    '<< "EVENT1_C0_REPRESENTATION=MEAN_VALUE_DOUBLETON\\n"',
    '<< "TANGENT_GAUGES=IDENTITY,MIDPOINT_M,ORIENTED_QR\\n"',
    '<< "C0_FACTOR_REORGANIZATION=DISABLED_TO_PRESERVE_SOURCE_R0\\n"',
    '<< "SCIENTIFIC_RESULT_CLASS=CORRELATION_PRESERVED_ORIENTATION_UNRESOLVED\\n"',
    '<< "AUTONOMOUS_VECTOR_FIELD=true\\n"',
    '<< "INCOMING_DP_REINJECTED=false\\n"',
    '<< "POSTSECTION_STATE_REUSED=false\\n"',
)
for needle in required_source:
    if needle not in source_text:
        raise SystemExit(f"section-resident source contract missing: {needle}")
runner_text = runner.read_text(encoding="ascii")
required_runner = (
    'compile_args=("$cxx_path" -std=c++17 "${cflags[@]}" -O0',
    'printf \'TANGENT_GAUGES=IDENTITY,MIDPOINT_M,ORIENTED_QR\\n\'',
    '--expected-baseline-receipt-sha256',
    '--expected-flattened-physical-sha256',
    'printf \'OPTIMIZATION_LEVEL=O0\\n\'',
    'printf \'FLATTENED_BASELINE_STABLE_DURING_RUN=true\\n\'',
)
for needle in required_runner:
    if needle not in runner_text:
        raise SystemExit(f"runner contract missing: {needle}")
if re.search(r'compile_args=.*-(?:O1|O2|O3|Ofast)(?:[ "\\])', runner_text):
    raise SystemExit("runner compile command must remain at O0")
print(f"DEPENDENCY_CONTENT_ENTRIES_VERIFIED={dependency_entries}")
print(f"LINK_INPUT_ENTRIES_VERIFIED={link_entries}")
print(f"RUNTIME_LIBRARY_ENTRIES_VERIFIED={runtime_entries}")
print("PROVENANCE_BINDINGS_VERIFIED=true")
print("FLATTENED_BASELINE_VERIFIED=true")
PY

grep -Fq 'frozen N0 tile' "$document"
grep -Fq 'two MinusPlus returns' "$document"
grep -Fq 'INV-20260731-cs6-section-resident-reconditioned-two-return' "$document"
grep -Fq 'mean-value C0' "$document"
grep -Fq 'matrix determinant still crosses zero' "$document"
grep -Fq 'LIOUVILLE_DETERMINANT_NEGATIVE=true' "$document"
grep -Fq 'ANY_GAUGE_SIGN_DEFINITE=false' "$document"
grep -Fq 'PROMOTION_ELIGIBLE=false' "$document"

capd_config="${CS6_SECTION_RESIDENT_RECONDITIONED_TWO_RETURN_CAPD_CONFIG:-${CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}}"
[[ -x "$capd_config" ]] || {
  printf 'fresh replay requires executable CAPD_CONFIG (tried %s)\n' "$capd_config" >&2
  exit 66
}
replay_root="$(mktemp -d)"
trap 'rm -rf "$replay_root"' EXIT
challenge="$(printf '%s:%s:%s\n' "$(date -u +%s%N)" "$$" "$RANDOM" \
  | sha256sum | awk '{print $1}')"
bash "$runner" --capd-config "$capd_config" --run-dir "$replay_root/run" \
  --challenge "$challenge" > "$replay_root/runner-output.txt"
grep -Fxq 'VERIFY_PASS=true' "$replay_root/runner-output.txt"
grep -Fxq "RUN_CHALLENGE=$challenge" "$replay_root/run/run-manifest.txt"
grep -Fxq 'RUN_COMPLETE=true' "$replay_root/run/run-manifest.txt"
grep -Fxq 'ROUNDING_MATH_EFFECTIVE=true' "$replay_root/run/run-manifest.txt"
grep -Fxq 'OPTIMIZATION_LEVEL=O0' "$replay_root/run/run-manifest.txt"
grep -Fxq 'DEPENDENCIES_STABLE_DURING_RUN=true' "$replay_root/run/run-manifest.txt"
grep -Fxq 'FLATTENED_BASELINE_STABLE_DURING_RUN=true' \
  "$replay_root/run/run-manifest.txt"
grep -Fxq 'FLATTENED_BASELINE_RECEIPT_SHA256=14315dd35ada83d13bddaa1c653e0dea86a9da91379559e7f64d69b314077dba' \
  "$replay_root/run/run-manifest.txt"
grep -Fxq 'FLATTENED_BASELINE_PHYSICAL_SHA256=536dea89d9f841e0afedaaeb9ef116f5237fb7dd96f7774340850833b5f4b0b1' \
  "$replay_root/run/run-manifest.txt"
grep -Fxq 'PROMOTION_ELIGIBLE=false' "$replay_root/run/run-manifest.txt"
python3 - "$provenance" "$replay_root/run" "$source_file" "$verifier" \
  "$runner" "$challenge" <<'PY'
import hashlib
import re
import subprocess
import sys
from pathlib import Path

provenance_path, run_dir, source, verifier, runner = map(Path, sys.argv[1:6])
challenge = sys.argv[6]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kv(path):
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in result:
            raise SystemExit(f"bad replay binding file: {path}")
        result[key] = value
    return result


provenance = read_kv(provenance_path)
manifest = read_kv(run_dir / "run-manifest.txt")
repo = provenance_path.parents[2]
canonical_manifest = read_kv(repo / provenance["MANIFEST_PATH"])
for key in ("SOURCE_SHA256", "VERIFIER_SHA256", "RUNNER_SHA256", "INPUT_SHA256"):
    if manifest.get(key) != provenance.get(key):
        raise SystemExit(f"fresh replay drifted from canonical {key}")
if manifest.get("RUN_CHALLENGE") != challenge:
    raise SystemExit("fresh replay challenge mismatch")
if challenge == provenance["RUN_CHALLENGE"]:
    raise SystemExit("fresh replay reused retained challenge")
if manifest.get("RECEIPT_SHA256") == provenance["RECEIPT_SHA256"]:
    raise SystemExit("fresh replay did not produce a challenge-distinct receipt")
stable_environment_keys = (
    "CAPD_CONFIG_SHA256", "CAPD_CFLAGS_SHA256", "CAPD_LIBS_SHA256",
    "CAPD_VERSION_SHA256", "PREPROCESSOR_MACROS_SHA256",
    "EFFECTIVE_OPTIONS_SHA256", "DEPENDENCIES_SHA256",
    "LINK_INPUTS_SHA256", "RUNTIME_LIBRARIES_SHA256", "COMPILER_SHA256",
    "COMPILER_VERSION_SHA256", "PYTHON_SHA256", "PYTHON_VERSION_SHA256",
    "FLATTENED_BASELINE_RECEIPT_SHA256",
    "FLATTENED_BASELINE_PROVENANCE_SHA256",
    "FLATTENED_BASELINE_VERIFIER_SHA256",
    "FLATTENED_BASELINE_PHYSICAL_SHA256",
)
for key in stable_environment_keys:
    if manifest.get(key) != canonical_manifest.get(key):
        raise SystemExit(f"fresh replay environment drifted at {key}")
verification = (run_dir / "verification.txt").read_text(encoding="ascii").splitlines()
physical = f"PHYSICAL_CHAIN_SHA256={provenance['PHYSICAL_CHAIN_SHA256']}"
if verification.count(physical) != 1:
    raise SystemExit("fresh replay physical digest drifted from canonical evidence")
comparison = (
    f"COMPARISON_CHAIN_SHA256={provenance['COMPARISON_CHAIN_SHA256']}"
)
if verification.count(comparison) != 1:
    raise SystemExit("fresh replay comparison digest drifted from canonical evidence")
flattened_physical = (
    "FLATTENED_PHYSICAL_CHAIN_SHA256="
    f"{provenance['FLATTENED_BASELINE_PHYSICAL_SHA256']}"
)
if verification.count(flattened_physical) != 1:
    raise SystemExit("fresh replay flattened physical digest mismatch")
for key in ("PHYSICAL_CHAIN_SHA256", "COMPARISON_CHAIN_SHA256"):
    if manifest.get(key) != provenance.get(key):
        raise SystemExit(f"fresh replay manifest drifted at {key}")
for key, path in (
    ("SOURCE_SHA256", source),
    ("VERIFIER_SHA256", verifier),
    ("RUNNER_SHA256", runner),
):
    if digest(path) != provenance[key]:
        raise SystemExit(f"current file changed during fresh replay: {path.name}")


def expect_binding_rejection(ledger, expected_challenge, expected_receipt):
    result = subprocess.run(
        (
            sys.executable, str(verifier), str(ledger),
            "--expected-source-sha256", provenance["SOURCE_SHA256"],
            "--expected-input-sha256", provenance["INPUT_SHA256"],
            "--expected-run-challenge", expected_challenge,
            "--expected-receipt-sha256", expected_receipt,
            "--expected-baseline-receipt-sha256",
            provenance["FLATTENED_BASELINE_RECEIPT_SHA256"],
            "--expected-flattened-physical-sha256",
            provenance["FLATTENED_BASELINE_PHYSICAL_SHA256"],
        ), capture_output=True, text=True, check=False,
    )
    if result.returncode != 2 or not result.stderr.startswith("VERIFY_ERROR="):
        raise SystemExit("cross-challenge receipt binding did not fail closed")


expect_binding_rejection(
    run_dir / "ledger.txt", provenance["RUN_CHALLENGE"],
    manifest["RECEIPT_SHA256"],
)
expect_binding_rejection(
    repo / provenance["RECEIPT_PATH"], challenge,
    provenance["RECEIPT_SHA256"],
)
print("FRESH_REPLAY_CANONICAL_BINDING=true")
print("FRESH_REPLAY_ENVIRONMENT_BINDING=true")
print("CROSS_CHALLENGE_REJECTIONS=true")
PY
if grep -Fq 'bundle-index.sha256' "$replay_root/run/bundle-index.sha256"; then
  printf 'bundle index contains an invalid self-hash\n' >&2
  exit 1
fi
(cd "$replay_root/run" && sha256sum -c bundle-index.sha256 > /dev/null)
printf 'FRESH_REPLAY_PASS=true\n'
printf 'CS6_SECTION_RESIDENT_RECONDITIONED_TWO_RETURN_GATE_PASS=true\n'
