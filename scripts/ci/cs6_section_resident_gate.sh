#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_section_resident_probe.cpp"
verifier="$repo_root/scripts/research/cs6_section_resident_verify.py"
runner="$repo_root/scripts/research/cs6_section_resident_run.sh"
receipt="$repo_root/scripts/research/cs6_section_resident_receipt_v1.txt"
provenance="$repo_root/scripts/research/cs6_section_resident_provenance_v1.txt"
document="$repo_root/docs/research/cs6_section_resident_return_2026-07-31.md"

for path in "$source_file" "$verifier" "$runner" "$receipt" \
  "$provenance" "$document"; do
  [[ -f "$path" ]] || { printf 'missing gate input: %s\n' "$path" >&2; exit 66; }
done

bash -n "$runner"
python3 - "$verifier" <<'PY'
import sys
from pathlib import Path

compile(Path(sys.argv[1]).read_bytes(), sys.argv[1], "exec")
PY

python3 - "$repo_root" "$source_file" "$verifier" "$runner" \
  "$receipt" "$provenance" <<'PY'
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
sha_re = re.compile(r"^[0-9a-f]{64}$")


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
provenance = read_kv(provenance_path, provenance_keys)
fixed_provenance = {
    "SCHEMA": "sounio.cs6.section-resident-provenance.v1",
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
    ("RECEIPT_SHA256", receipt),
):
    if provenance[key] != digest(path):
        raise SystemExit(f"current artifact does not match provenance: {path.name}")

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
    "RECEIPT": "scripts/research/cs6_section_resident_receipt_v1.txt",
    "INPUT": "scripts/research/receipts/cs6_section_resident_input_v1.txt",
    "MANIFEST": "scripts/research/receipts/cs6_section_resident_run_manifest_v1.txt",
    "VERIFICATION": "scripts/research/receipts/cs6_section_resident_verification_v1.txt",
    "DEPENDENCIES": "scripts/research/receipts/cs6_section_resident_dependencies_v1.sha256",
    "LINK_INPUTS": "scripts/research/receipts/cs6_section_resident_link_inputs_v1.sha256",
    "RUNTIME_LIBRARIES": "scripts/research/receipts/cs6_section_resident_runtime_libraries_v1.sha256",
    "COMPILE_COMMAND": "scripts/research/receipts/cs6_section_resident_compile_command_v1.txt",
    "CAPD_CFLAGS": "scripts/research/receipts/cs6_section_resident_capd_cflags_v1.txt",
    "CAPD_LIBS": "scripts/research/receipts/cs6_section_resident_capd_libs_v1.txt",
    "CAPD_VERSION": "scripts/research/receipts/cs6_section_resident_capd_version_v1.txt",
    "PREPROCESSOR_MACROS": "scripts/research/receipts/cs6_section_resident_preprocessor_macros_v1.txt",
    "EFFECTIVE_OPTIONS": "scripts/research/receipts/cs6_section_resident_effective_options_v1.txt",
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
    "CAPD_CONFIG_SHA256", "CAPD_CFLAGS_SHA256", "CAPD_LIBS_SHA256",
    "CAPD_VERSION_SHA256", "PREPROCESSOR_MACROS_SHA256",
    "EFFECTIVE_OPTIONS_SHA256", "EFFECTIVE_OPTIONS_STDERR_SHA256",
    "DEPENDENCY_PATHS_SHA256", "DEPENDENCIES_SHA256", "LINK_INPUTS_SHA256",
    "RUNTIME_LIBRARIES_SHA256", "COMPILER_SHA256", "COMPILER_VERSION_SHA256",
    "PYTHON_SHA256", "PYTHON_VERSION_SHA256", "RUNTIME_LINKAGE_SHA256",
    "COMPILE_COMMAND_SHA256", "COMPILE_STDERR_SHA256", "PROBE_STDERR_SHA256",
    "GIT_HEAD", "GIT_STATUS_CLEAN", "CAPD_VERSION", "INTERVAL_BACKEND",
    "ROUNDING_MATH_EFFECTIVE", "DEPENDENCIES_STABLE_DURING_RUN",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE", "EXECUTION_PROVENANCE_ATTESTED",
    "INDEPENDENT_REPLAY_REQUIRED", "PROMOTION_ELIGIBLE",
}
manifest = read_kv(resolved["MANIFEST"], manifest_keys)
for key, value in manifest.items():
    if key.endswith("SHA256") and sha_re.fullmatch(value) is None:
        raise SystemExit(f"malformed retained manifest hash: {key}")
fixed_manifest = {
    "MANIFEST_KIND": "CS6_SECTION_RESIDENT_ONE_RETURN_V1",
    "RUN_COMPLETE": "true", "WORKER_EXIT": "0", "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB", "ROUNDING_MATH_EFFECTIVE": "true",
    "DEPENDENCIES_STABLE_DURING_RUN": "true",
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
    "INPUT_SCHEMA=sounio.cs6.section-resident-input.v1\n"
    "SOURCE=N0\n"
    "U_INDEX=20000\n"
    "S_INDEX=15000\n"
    "U_TILES=40000\n"
    "S_TILES=30000\n"
    "ORDER=8\n"
    "RETURN_COUNT=1\n"
    "SECTION=COORDINATE_W_EQUALS_ZERO\n"
    "CROSSING_DIRECTION=MINUS_PLUS\n"
    "VECTOR_FIELD=CS6_FROZEN_22.3274637391\n"
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
    "VERIFY_SCHEMA=sounio.cs6.section-resident-carrier-verification.v1",
    "VERIFY_PASS=true", "SOURCE=N0", "TILE=20000,15000/40000,30000",
    "RETURN_COUNT=1", "RAW_C0_RECONSTRUCTED=true", "RAW_C1_RECONSTRUCTED=true",
    "POINCARE_DP_RECOMPUTED=true", "POINCARE_DP_CONTAINS_RECOMPUTATION=true",
    "EXP_ELL_RECOMPUTED=false",
    f"PHYSICAL_CHAIN_SHA256={provenance['PHYSICAL_CHAIN_SHA256']}",
    "STATE_JOINT_OVERLAP=true", "TIME_JOINT_OVERLAP=true",
    "DP_JOINT_OVERLAP=true", "VELOCITY_JOINT_OVERLAP=true",
    "DETERMINANT_JOINT_OVERLAP=true", "PROMOTION_ELIGIBLE=false",
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
           receipt_hash: str | None = None) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as candidate:
        candidate.write(data)
        candidate.flush()
        return subprocess.run(
            (
                sys.executable, str(verifier), candidate.name,
                "--expected-source-sha256", source_hash or expected_args["source"],
                "--expected-input-sha256", input_hash or expected_args["input"],
                "--expected-run-challenge", challenge or expected_args["challenge"],
                "--expected-receipt-sha256",
                receipt_hash or hashlib.sha256(data).hexdigest(),
            ), capture_output=True, text=True, check=False,
        )


def rejected(name: str, data: bytes, **kwargs: str) -> None:
    result = invoke(data, **kwargs)
    if result.returncode != 2 or not result.stderr.startswith("VERIFY_ERROR="):
        raise SystemExit(
            f"negative mutation did not fail closed: {name}, rc={result.returncode}"
        )


baseline_result = invoke(baseline)
if baseline_result.returncode != 0 or "VERIFY_PASS=true\n" not in baseline_result.stdout:
    raise SystemExit("retained receipt no longer verifies")
if f"PHYSICAL_CHAIN_SHA256={provenance['PHYSICAL_CHAIN_SHA256']}\n" not in baseline_result.stdout:
    raise SystemExit("retained physical digest mismatch")


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
candidate_time = re.search(
    rb"^CANDIDATE TIME=(\[[^\]]+\]) ", baseline, re.MULTILINE
).group(1).decode("ascii")

lines = baseline.decode("ascii").splitlines()
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
    "promotion-lie": text_mutation(replace_once(
        "PROMOTION_ELIGIBLE=false", "PROMOTION_ELIGIBLE=true",
    )),
    "full-source-lie": text_mutation(replace_once(
        "FULL_SOURCE_CARRIER_PROVED=false", "FULL_SOURCE_CARRIER_PROVED=true",
    )),
    "summary-false": text_mutation(replace_once(
        "PROBE_PASS=true", "PROBE_PASS=false",
    )),
    "source-tile": text_mutation(mutate_record("SOURCE_TILE", "SOURCE_U", one)),
    "source-q0-normal": text_mutation(mutate_record("SOURCE_TILE", "Q022", one)),
    "candidate-dp": text_mutation(mutate_record("CANDIDATE", "DP00", one)),
    "section-dp": text_mutation(mutate_record("CANDIDATE", "SECTION_DP00", one)),
    "event-c0-raw": text_mutation(mutate_record("EVENT_CARRIER", "C0_X0", one)),
    "event-c0-normal": text_mutation(mutate_record("EVENT_CARRIER", "C0_HULL2", one)),
    "event-c1-raw": text_mutation(mutate_record("EVENT_CARRIER", "C1_D00", one)),
    "continuation-normal-seed": text_mutation(mutate_record(
        "CONTINUATION_CARRIER", "C1_D22", one,
    )),
    "incoming-dp": text_mutation(mutate_record(
        "CONTINUATION_CARRIER", "INCOMING_DP00", one,
    )),
    "post-time": text_mutation(mutate_record("POSTSECTION", "TIME", candidate_time)),
    "post-x2": text_mutation(mutate_record("POSTSECTION", "X2", zero)),
    "post-sign": text_mutation(mutate_record("POSTSECTION", "SECTION_SIGN", zero)),
    "direct-nu": text_mutation(mutate_record("DIRECT", "NU", zero)),
    "liouville-det": text_mutation(mutate_record("LIOUVILLE", "DET", one)),
    "unknown-marker": text_mutation(replace_once("SUMMARY ", "SUMMERY ")),
}
for name, data in mutations.items():
    rejected(name, data)

rejected("expected-source", baseline, source_hash="f" * 64)
rejected("expected-input", baseline, input_hash="f" * 64)
rejected("expected-challenge", baseline, challenge="f" * 64)
rejected("expected-receipt", baseline, receipt_hash="f" * 64)
print(f"NEGATIVE_MUTATIONS_REJECTED={len(mutations) + 4}")

source_text = source.read_text(encoding="ascii")
required_source = (
    "class SectionResidentMap : public IPoincareMap",
    "this->integrateUntilSectionCrossing(before, after, 1);",
    "this->crossSectionInOneStep(before, after, local_time,",
    "this->sectionDerivativesEnclosure.computeOneStepSectionEnclosure(",
    "C1Rect2Set event_carrier(event_c0, event_c1, candidate.time);",
    "C1Rect2Set continuation_carrier(event_c0, seed_c1, candidate.time);",
    '<< "FLOW_TANGENT_ROLE=D_FLOW_TIMES_Q0\\n"',
    '<< "SOURCE_TANGENT_SEED_ROLE=GLOBAL_FRAME_RADII_WITH_ZERO_DUMMY_NORMAL\\n"',
    '<< "INCOMING_DP_REINJECTED=false\\n"',
    '<< "POSTSECTION_STATE_REUSED=false\\n"',
)
for needle in required_source:
    if needle not in source_text:
        raise SystemExit(f"section-resident source contract missing: {needle}")
print(f"DEPENDENCY_CONTENT_ENTRIES_VERIFIED={dependency_entries}")
print(f"LINK_INPUT_ENTRIES_VERIFIED={link_entries}")
print(f"RUNTIME_LIBRARY_ENTRIES_VERIFIED={runtime_entries}")
print("PROVENANCE_BINDINGS_VERIFIED=true")
PY

grep -Fq 'one frozen N0 tile' "$document"
grep -Fq 'section-resident' "$document"
grep -Fq 'INV-20260731-cs6-section-resident-continuation' "$document"
grep -Fq 'EXP_ELL_RECOMPUTED=false' "$document"

capd_config="${CS6_SECTION_RESIDENT_CAPD_CONFIG:-${CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}}"
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
grep -Fxq 'DEPENDENCIES_STABLE_DURING_RUN=true' "$replay_root/run/run-manifest.txt"
grep -Fxq 'PROMOTION_ELIGIBLE=false' "$replay_root/run/run-manifest.txt"
python3 - "$provenance" "$replay_root/run" "$source_file" "$verifier" \
  "$runner" "$challenge" <<'PY'
import hashlib
import re
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
for key in ("SOURCE_SHA256", "VERIFIER_SHA256", "RUNNER_SHA256", "INPUT_SHA256"):
    if manifest.get(key) != provenance.get(key):
        raise SystemExit(f"fresh replay drifted from canonical {key}")
if manifest.get("RUN_CHALLENGE") != challenge:
    raise SystemExit("fresh replay challenge mismatch")
verification = (run_dir / "verification.txt").read_text(encoding="ascii").splitlines()
physical = f"PHYSICAL_CHAIN_SHA256={provenance['PHYSICAL_CHAIN_SHA256']}"
if verification.count(physical) != 1:
    raise SystemExit("fresh replay physical digest drifted from canonical evidence")
for key, path in (
    ("SOURCE_SHA256", source),
    ("VERIFIER_SHA256", verifier),
    ("RUNNER_SHA256", runner),
):
    if digest(path) != provenance[key]:
        raise SystemExit(f"current file changed during fresh replay: {path.name}")
print("FRESH_REPLAY_CANONICAL_BINDING=true")
PY
if grep -Fq 'bundle-index.sha256' "$replay_root/run/bundle-index.sha256"; then
  printf 'bundle index contains an invalid self-hash\n' >&2
  exit 1
fi
(cd "$replay_root/run" && sha256sum -c bundle-index.sha256 > /dev/null)
printf 'FRESH_REPLAY_PASS=true\n'
printf 'CS6_SECTION_RESIDENT_GATE_PASS=true\n'
