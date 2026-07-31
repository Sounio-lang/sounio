#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
capd_config="${CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}"

source_file="$repo_root/scripts/research/cs6_c1_dependency_probe.cpp"
verifier="$repo_root/scripts/research/cs6_c1_dependency_verify.py"
runner="$repo_root/scripts/research/cs6_c1_dependency_run.sh"
gate="$repo_root/scripts/ci/cs6_c1_dependency_gate.sh"
document="$repo_root/docs/research/cs6_c1_dependency_2026-07-31.md"
registry="$repo_root/docs/internal/concepts/registry.tsv"
offload_log="$repo_root/.claude/llm_offload_log.md"
receipt="$repo_root/scripts/research/cs6_c1_dependency_receipt_v1.txt"
provenance="$repo_root/scripts/research/cs6_c1_dependency_provenance_v1.txt"

for path in "$source_file" "$verifier" "$runner" "$gate" "$document" \
  "$registry" "$offload_log" "$receipt" "$provenance"; do
  [[ -f "$path" ]] || { printf 'missing gate input: %s\n' "$path" >&2; exit 66; }
done
[[ -x "$capd_config" ]] || {
  printf 'CAPD_CONFIG is not executable: %s\n' "$capd_config" >&2
  exit 66
}

bash -n "$runner"
python3 - "$verifier" <<'PY'
import sys
from pathlib import Path

compile(Path(sys.argv[1]).read_bytes(), sys.argv[1], "exec")
PY

python3 - "$repo_root" "$capd_config" <<'PY'
import hashlib
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
capd_config = Path(sys.argv[2]).resolve()
source = repo / "scripts/research/cs6_c1_dependency_probe.cpp"
verifier = repo / "scripts/research/cs6_c1_dependency_verify.py"
runner = repo / "scripts/research/cs6_c1_dependency_run.sh"
gate = repo / "scripts/ci/cs6_c1_dependency_gate.sh"
document = repo / "docs/research/cs6_c1_dependency_2026-07-31.md"
registry = repo / "docs/internal/concepts/registry.tsv"
offload_log = repo / ".claude/llm_offload_log.md"
provenance_path = repo / "scripts/research/cs6_c1_dependency_provenance_v1.txt"
sha_re = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kv(path: Path, expected_keys: set[str]) -> dict[str, str]:
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII key/value artifact: {path}") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise SystemExit(f"noncanonical key/value artifact: {path}")
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
        raise SystemExit(
            f"key grammar mismatch for {path}: missing={missing} extra={extra}"
        )
    return result


provenance_keys = {
    "SCHEMA", "GIT_BASE",
    "SOURCE_PATH", "SOURCE_SHA256",
    "VERIFIER_PATH", "VERIFIER_SHA256",
    "RUNNER_PATH", "RUNNER_SHA256",
    "GATE_PATH", "GATE_SHA256",
    "DOCUMENT_PATH", "DOCUMENT_SHA256",
    "RECEIPT_PATH", "RECEIPT_SHA256",
    "INPUT_PATH", "INPUT_SHA256",
    "MANIFEST_PATH", "MANIFEST_SHA256",
    "VERIFICATION_PATH", "VERIFICATION_SHA256",
    "DEPENDENCIES_PATH", "DEPENDENCIES_SHA256",
    "LINK_INPUTS_PATH", "LINK_INPUTS_SHA256",
    "RUNTIME_LIBRARIES_PATH", "RUNTIME_LIBRARIES_SHA256",
    "COMPILE_COMMAND_PATH", "COMPILE_COMMAND_SHA256",
    "CAPD_CFLAGS_PATH", "CAPD_CFLAGS_SHA256",
    "CAPD_LIBS_PATH", "CAPD_LIBS_SHA256",
    "CAPD_VERSION_ARTIFACT_PATH", "CAPD_VERSION_ARTIFACT_SHA256",
    "PREPROCESSOR_MACROS_PATH", "PREPROCESSOR_MACROS_SHA256",
    "EFFECTIVE_OPTIONS_PATH", "EFFECTIVE_OPTIONS_SHA256",
    "BASELINE_RECEIPT_PATH", "BASELINE_RECEIPT_SHA256",
    "BASELINE_PROVENANCE_PATH", "BASELINE_PROVENANCE_SHA256",
    "BASELINE_PHYSICAL_SHA256",
    "RUN_CHALLENGE", "PHYSICAL_SHA256",
    "CAPD_VERSION", "INTERVAL_BACKEND", "DEPENDENCY_COUNT",
    "LINK_INPUT_COUNT", "RUNTIME_LIBRARY_COUNT", "MUTATION_TESTS",
    "SCIENTIFIC_RESULT_CLASS", "RESULT_SCOPE", "EXECUTION_TRUST_MODEL",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE", "REMOTE_ATTESTATION_PRESENT",
    "INDEPENDENT_REPLAY_REQUIRED", "PROMOTION_ELIGIBLE",
    "FULL_SOURCE_CARRIER_PROVED", "PROJECTIVE_RICCATI_INTEGRATED",
    "HYPERBOLICITY_PROVED", "CHAOTIC_ATTRACTOR_PROVED",
    "LEGACY_BASELINE_KEPT",
}
provenance = read_kv(provenance_path, provenance_keys)

fixed_provenance = {
    "SCHEMA": "sounio.cs6.c1-dependency-provenance.v1",
    "GIT_BASE": "0bcd234e1563be8182b69883d80b3b8ef2fdb257",
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "DEPENDENCY_COUNT": "592",
    "LINK_INPUT_COUNT": "19",
    "RUNTIME_LIBRARY_COUNT": "4",
    "MUTATION_TESTS": "39",
    "SCIENTIFIC_RESULT_CLASS":
        "BOUNDED_TILE_PARAMETERIZED_JACOBIAN_SIGN_CERTIFIED",
    "RESULT_SCOPE": "ONE_FROZEN_N0_TILE_TWO_RETURNS",
    "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
    "DEPENDENCY_CONTENT_HASHES_COMPLETE": "false",
    "REMOTE_ATTESTATION_PRESENT": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "PROMOTION_ELIGIBLE": "false",
    "FULL_SOURCE_CARRIER_PROVED": "false",
    "PROJECTIVE_RICCATI_INTEGRATED": "false",
    "HYPERBOLICITY_PROVED": "false",
    "CHAOTIC_ATTRACTOR_PROVED": "false",
    "LEGACY_BASELINE_KEPT": "true",
}
for key, expected in fixed_provenance.items():
    if provenance[key] != expected:
        raise SystemExit(f"provenance {key} mismatch")
for key, value in provenance.items():
    if key.endswith("SHA256") and sha_re.fullmatch(value) is None:
        raise SystemExit(f"malformed provenance hash: {key}")
if sha_re.fullmatch(provenance["RUN_CHALLENGE"]) is None:
    raise SystemExit("malformed retained challenge")

artifact_specs = {
    "SOURCE": (source, "scripts/research/cs6_c1_dependency_probe.cpp"),
    "VERIFIER": (verifier, "scripts/research/cs6_c1_dependency_verify.py"),
    "RUNNER": (runner, "scripts/research/cs6_c1_dependency_run.sh"),
    "GATE": (gate, "scripts/ci/cs6_c1_dependency_gate.sh"),
    "DOCUMENT": (
        document, "docs/research/cs6_c1_dependency_2026-07-31.md"
    ),
    "RECEIPT": (
        repo / "scripts/research/cs6_c1_dependency_receipt_v1.txt",
        "scripts/research/cs6_c1_dependency_receipt_v1.txt",
    ),
    "INPUT": (
        repo / "scripts/research/receipts/cs6_c1_dependency_input_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_input_v1.txt",
    ),
    "MANIFEST": (
        repo / "scripts/research/receipts/cs6_c1_dependency_run_manifest_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_run_manifest_v1.txt",
    ),
    "VERIFICATION": (
        repo / "scripts/research/receipts/cs6_c1_dependency_verification_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_verification_v1.txt",
    ),
    "DEPENDENCIES": (
        repo / "scripts/research/receipts/cs6_c1_dependency_dependencies_v1.sha256",
        "scripts/research/receipts/cs6_c1_dependency_dependencies_v1.sha256",
    ),
    "LINK_INPUTS": (
        repo / "scripts/research/receipts/cs6_c1_dependency_link_inputs_v1.sha256",
        "scripts/research/receipts/cs6_c1_dependency_link_inputs_v1.sha256",
    ),
    "RUNTIME_LIBRARIES": (
        repo / "scripts/research/receipts/cs6_c1_dependency_runtime_libraries_v1.sha256",
        "scripts/research/receipts/cs6_c1_dependency_runtime_libraries_v1.sha256",
    ),
    "COMPILE_COMMAND": (
        repo / "scripts/research/receipts/cs6_c1_dependency_compile_command_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_compile_command_v1.txt",
    ),
    "CAPD_CFLAGS": (
        repo / "scripts/research/receipts/cs6_c1_dependency_capd_cflags_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_capd_cflags_v1.txt",
    ),
    "CAPD_LIBS": (
        repo / "scripts/research/receipts/cs6_c1_dependency_capd_libs_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_capd_libs_v1.txt",
    ),
    "CAPD_VERSION_ARTIFACT": (
        repo / "scripts/research/receipts/cs6_c1_dependency_capd_version_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_capd_version_v1.txt",
    ),
    "PREPROCESSOR_MACROS": (
        repo / "scripts/research/receipts/cs6_c1_dependency_preprocessor_macros_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_preprocessor_macros_v1.txt",
    ),
    "EFFECTIVE_OPTIONS": (
        repo / "scripts/research/receipts/cs6_c1_dependency_effective_options_v1.txt",
        "scripts/research/receipts/cs6_c1_dependency_effective_options_v1.txt",
    ),
    "BASELINE_RECEIPT": (
        repo / "scripts/research/cs6_section_resident_reconditioned_two_return_receipt_v1.txt",
        "scripts/research/cs6_section_resident_reconditioned_two_return_receipt_v1.txt",
    ),
    "BASELINE_PROVENANCE": (
        repo / "scripts/research/cs6_section_resident_reconditioned_two_return_provenance_v1.txt",
        "scripts/research/cs6_section_resident_reconditioned_two_return_provenance_v1.txt",
    ),
}
resolved: dict[str, Path] = {}
for prefix, (path, expected_path) in artifact_specs.items():
    path_key = f"{prefix}_PATH"
    sha_key = f"{prefix}_SHA256"
    if provenance[path_key] != expected_path:
        raise SystemExit(f"unexpected artifact path: {prefix}")
    if not path.is_file() or provenance[sha_key] != digest(path):
        raise SystemExit(f"artifact hash mismatch: {prefix}")
    resolved[prefix] = path

expected_baseline_receipt = (
    "3d17e9b8ad09c9b253c56b181a4eab90c0390eb5582e3ca542ccb3dcc44f6956"
)
expected_baseline_provenance = (
    "22fad25dfa795b63f361d45cc9de1d10177b3f7cd812a75252d3f47b1438344d"
)
expected_baseline_physical = (
    "8b5073b5261708991597af9d784b2b1ad998f5355f92a659925ec3f3882b4e3e"
)
if provenance["BASELINE_RECEIPT_SHA256"] != expected_baseline_receipt:
    raise SystemExit("baseline receipt constant mismatch")
if provenance["BASELINE_PROVENANCE_SHA256"] != expected_baseline_provenance:
    raise SystemExit("baseline provenance constant mismatch")
if provenance["BASELINE_PHYSICAL_SHA256"] != expected_baseline_physical:
    raise SystemExit("baseline physical constant mismatch")
baseline_text = resolved["BASELINE_PROVENANCE"].read_text(encoding="ascii")
if f"PHYSICAL_CHAIN_SHA256={expected_baseline_physical}\n" not in baseline_text:
    raise SystemExit("baseline provenance omits expected physical digest")

manifest_keys = {
    "MANIFEST_SCHEMA", "RUN_COMPLETE", "SOURCE_SHA256", "VERIFIER_SHA256",
    "RUNNER_SHA256", "INPUT_SHA256", "RUN_CHALLENGE", "EXECUTABLE_SHA256",
    "RECEIPT_SHA256", "VERIFICATION_SHA256", "PHYSICAL_SHA256",
    "DEPENDENCIES_SHA256", "DEPENDENCY_COUNT", "LINK_INPUTS_SHA256",
    "LINK_INPUT_COUNT", "RUNTIME_LIBRARIES_SHA256", "RUNTIME_LIBRARY_COUNT",
    "BASELINE_RECEIPT_SHA256", "BASELINE_PROVENANCE_SHA256",
    "BASELINE_PHYSICAL_SHA256", "CAPD_VERSION", "INTERVAL_BACKEND",
    "OPTIMIZATION_LEVEL", "EXECUTION_TRUST_MODEL",
    "REMOTE_ATTESTATION_PRESENT", "INDEPENDENT_REPLAY_REQUIRED",
    "PROMOTION_ELIGIBLE",
}
manifest = read_kv(resolved["MANIFEST"], manifest_keys)
fixed_manifest = {
    "MANIFEST_SCHEMA": "sounio.cs6.c1-dependency-run-manifest.v1",
    "RUN_COMPLETE": "true",
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "OPTIMIZATION_LEVEL": "O0",
    "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
    "REMOTE_ATTESTATION_PRESENT": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "PROMOTION_ELIGIBLE": "false",
}
for key, expected in fixed_manifest.items():
    if manifest[key] != expected:
        raise SystemExit(f"retained manifest {key} mismatch")
for key, value in manifest.items():
    if key.endswith("SHA256") and sha_re.fullmatch(value) is None:
        raise SystemExit(f"malformed retained manifest hash: {key}")
manifest_bindings = {
    "SOURCE_SHA256": provenance["SOURCE_SHA256"],
    "VERIFIER_SHA256": provenance["VERIFIER_SHA256"],
    "RUNNER_SHA256": provenance["RUNNER_SHA256"],
    "INPUT_SHA256": provenance["INPUT_SHA256"],
    "RUN_CHALLENGE": provenance["RUN_CHALLENGE"],
    "RECEIPT_SHA256": provenance["RECEIPT_SHA256"],
    "VERIFICATION_SHA256": provenance["VERIFICATION_SHA256"],
    "PHYSICAL_SHA256": provenance["PHYSICAL_SHA256"],
    "DEPENDENCIES_SHA256": provenance["DEPENDENCIES_SHA256"],
    "DEPENDENCY_COUNT": provenance["DEPENDENCY_COUNT"],
    "LINK_INPUTS_SHA256": provenance["LINK_INPUTS_SHA256"],
    "LINK_INPUT_COUNT": provenance["LINK_INPUT_COUNT"],
    "RUNTIME_LIBRARIES_SHA256": provenance["RUNTIME_LIBRARIES_SHA256"],
    "RUNTIME_LIBRARY_COUNT": provenance["RUNTIME_LIBRARY_COUNT"],
    "BASELINE_RECEIPT_SHA256": expected_baseline_receipt,
    "BASELINE_PROVENANCE_SHA256": expected_baseline_provenance,
    "BASELINE_PHYSICAL_SHA256": expected_baseline_physical,
}
for key, expected in manifest_bindings.items():
    if manifest[key] != expected:
        raise SystemExit(f"manifest/provenance binding mismatch: {key}")

expected_input = (
    "INPUT_SCHEMA=sounio.cs6.c1-dependency-input.v1\n"
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
    "ROUTE_A=C2_AFFINE_JACOBIAN_CARRIER\n"
    "ROUTE_B=FINAL_COLUMN_PROJECTIVE_SLOPE_CONTROL\n"
    f"BASELINE_RECEIPT_SHA256={expected_baseline_receipt}\n"
    f"BASELINE_PHYSICAL_SHA256={expected_baseline_physical}\n"
).encode("ascii")
if resolved["INPUT"].read_bytes() != expected_input:
    raise SystemExit("retained input grammar or value mismatch")


def verify_content_manifest(path: Path, expected_count: int, allow_source: bool) -> None:
    data = path.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        raise SystemExit(f"noncanonical content manifest: {path}")
    seen: set[str] = set()
    for number, line in enumerate(data.decode("ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SystemExit(f"malformed content manifest line {number}: {path}")
        expected, name = match.groups()
        if name in seen:
            raise SystemExit(f"duplicate content-manifest path: {name}")
        seen.add(name)
        if name == "BUNDLE/probe-source.cpp" and allow_source:
            target = source
        elif name.startswith("BUNDLE/"):
            raise SystemExit(f"unsupported bundle-relative dependency: {name}")
        else:
            target = Path(name)
        if not target.is_file() or digest(target) != expected:
            raise SystemExit(f"dependency content mismatch: {name}")
    if len(seen) != expected_count:
        raise SystemExit(f"content-manifest count mismatch: {path}")
    if allow_source and "BUNDLE/probe-source.cpp" not in seen:
        raise SystemExit("dependency manifest omits canonical source snapshot")


verify_content_manifest(resolved["DEPENDENCIES"], 592, True)
verify_content_manifest(resolved["LINK_INPUTS"], 19, False)
verify_content_manifest(resolved["RUNTIME_LIBRARIES"], 4, False)

if resolved["CAPD_VERSION_ARTIFACT"].read_bytes() != b"5.3.0\n":
    raise SystemExit("retained CAPD version artifact mismatch")
cflags = resolved["CAPD_CFLAGS"].read_text(encoding="ascii")
if "-D__USE_FILIB__" not in cflags or "-frounding-math" not in cflags:
    raise SystemExit("retained CAPD flags omit FILIB or rounding math")
macros = resolved["PREPROCESSOR_MACROS"].read_text(encoding="ascii")
if re.search(r"^#define __USE_FILIB__(?: 1)?$", macros, re.MULTILINE) is None:
    raise SystemExit("retained preprocessor state omits FILIB")
options = resolved["EFFECTIVE_OPTIONS"].read_text(encoding="ascii")
if re.search(r"^\s*-frounding-math\s+\[enabled\]", options, re.MULTILINE) is None:
    raise SystemExit("retained effective options disable rounding math")


def invoke_verifier(
    candidate: Path, source_hash: str, input_hash: str, challenge: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable, str(verifier), str(candidate),
            "--source-sha", source_hash,
            "--input-sha", input_hash,
            "--challenge", challenge,
            "--self-test-mutations",
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def expected_verification(receipt_hash: str, physical_hash: str) -> str:
    return (
        "VERIFICATION_SCHEMA=sounio.cs6.c1-dependency-verification.v1\n"
        f"RECEIPT_SHA256={receipt_hash}\n"
        f"PHYSICAL_SHA256={physical_hash}\n"
        "MUTATION_TESTS=39\n"
        "MUTATIONS_REJECTED=39\n"
        "CERTIFICATE_PASS=true\n"
    )


retained_result = invoke_verifier(
    resolved["RECEIPT"],
    provenance["SOURCE_SHA256"],
    provenance["INPUT_SHA256"],
    provenance["RUN_CHALLENGE"],
)
if retained_result.returncode != 0 or retained_result.stderr:
    raise SystemExit("retained receipt no longer verifies")
retained_verification = expected_verification(
    provenance["RECEIPT_SHA256"], provenance["PHYSICAL_SHA256"]
)
if retained_result.stdout != retained_verification:
    raise SystemExit("retained verifier output is not canonical")
if resolved["VERIFICATION"].read_text(encoding="ascii") != retained_verification:
    raise SystemExit("retained verification transcript mismatch")

receipt_data = resolved["RECEIPT"].read_bytes()
receipt_lines = receipt_data.splitlines()
raw_mutations = {
    "missing-line": b"\n".join(receipt_lines[:-1]) + b"\n",
    "extra-line": receipt_data + b"UNKNOWN=true\n",
    "crlf": receipt_data.replace(b"\n", b"\r\n"),
    "fixed-header": receipt_data.replace(b"SOURCE=N0\n", b"SOURCE=N1\n", 1),
    "non-ascii": receipt_data.replace(b"SOURCE=N0", "SOURCE=N\u2080".encode(), 1),
}
for name, mutation in raw_mutations.items():
    with tempfile.NamedTemporaryFile(suffix=".txt") as candidate:
        candidate.write(mutation)
        candidate.flush()
        result = invoke_verifier(
            Path(candidate.name),
            provenance["SOURCE_SHA256"],
            provenance["INPUT_SHA256"],
            provenance["RUN_CHALLENGE"],
        )
    if result.returncode != 1 or not result.stderr.startswith("verification error:"):
        raise SystemExit(f"raw grammar mutation escaped: {name}")

doc_text = document.read_text(encoding="ascii")
required_doc_fragments = (
    "SCIENTIFIC_RESULT_CLASS=BOUNDED_TILE_PARAMETERIZED_JACOBIAN_SIGN_CERTIFIED",
    "A_B_OUTCOME=AFFINE_ONLY",
    "It is not an intrinsic, coordinate-independent",
    "PROJECTIVE_RICCATI_INTEGRATED=false",
    "FULL_SOURCE_CARRIER_PROVED=false",
    "HYPERBOLICITY_PROVED=false",
    "CHAOTIC_ATTRACTOR_PROVED=false",
    "independent-replay fields are substantive limitations",
)
for fragment in required_doc_fragments:
    if fragment not in doc_text:
        raise SystemExit(f"document omits required boundary: {fragment}")
registry_row = (
    "SOUNIO-CS6-C1-SOURCE-DEPENDENCY\thypothesis\tfounder\t"
    "docs/research/cs6_c1_dependency_2026-07-31.md\t"
    "docs/research/cs6_c1_dependency_2026-07-31.md\t"
    "full-source-tiling-and-rigorous-projective-channel"
)
if registry.read_text(encoding="ascii").splitlines().count(registry_row) != 1:
    raise SystemExit("semantic registry row missing or duplicated")
offload_rows = [
    line for line in offload_log.read_text(encoding="utf-8").splitlines()
    if "docs/research/cs6_c1_dependency_2026-07-31.md" in line
]
required_offload_fragments = (
    "xai/grok-4.3; zai/GLM-5.2 (dual, per M1 policy)",
    "math-review + focused CAPD coefficient review",
    "scripts/research/cs6_c1_dependency_probe.cpp",
    "scripts/research/cs6_c1_dependency_verify.py",
    "PASS_WITH_DECLARED_TCB_BOUNDARY",
)
if not any(
    all(fragment in row for fragment in required_offload_fragments)
    for row in offload_rows
):
    raise SystemExit("CS6 C1 dependency dual-provider offload evidence missing")

with tempfile.TemporaryDirectory(prefix="cs6-c1-dependency-gate-", dir="/tmp") as tmp:
    fresh_dir = Path(tmp) / "run"
    fresh_challenge = secrets.token_hex(32)
    fresh_run = subprocess.run(
        (
            str(runner), "--capd-config", str(capd_config),
            "--run-dir", str(fresh_dir), "--challenge", fresh_challenge,
        ),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if fresh_run.returncode != 0:
        raise SystemExit(
            "fresh runner failed:\n" + fresh_run.stdout + fresh_run.stderr
        )
    fresh_manifest = read_kv(fresh_dir / "manifest.txt", manifest_keys)
    fresh_bindings = {
        "SOURCE_SHA256": provenance["SOURCE_SHA256"],
        "VERIFIER_SHA256": provenance["VERIFIER_SHA256"],
        "RUNNER_SHA256": provenance["RUNNER_SHA256"],
        "INPUT_SHA256": provenance["INPUT_SHA256"],
        "RUN_CHALLENGE": fresh_challenge,
        "PHYSICAL_SHA256": provenance["PHYSICAL_SHA256"],
        "DEPENDENCIES_SHA256": provenance["DEPENDENCIES_SHA256"],
        "DEPENDENCY_COUNT": "592",
        "LINK_INPUTS_SHA256": provenance["LINK_INPUTS_SHA256"],
        "LINK_INPUT_COUNT": "19",
        "RUNTIME_LIBRARIES_SHA256": provenance["RUNTIME_LIBRARIES_SHA256"],
        "RUNTIME_LIBRARY_COUNT": "4",
        "BASELINE_RECEIPT_SHA256": expected_baseline_receipt,
        "BASELINE_PROVENANCE_SHA256": expected_baseline_provenance,
        "BASELINE_PHYSICAL_SHA256": expected_baseline_physical,
    }
    for key, expected in {**fixed_manifest, **fresh_bindings}.items():
        if fresh_manifest[key] != expected:
            raise SystemExit(f"fresh manifest mismatch: {key}")
    fresh_receipt = fresh_dir / "ledger.txt"
    fresh_receipt_hash = digest(fresh_receipt)
    if fresh_manifest["RECEIPT_SHA256"] != fresh_receipt_hash:
        raise SystemExit("fresh receipt hash mismatch")
    fresh_verification = expected_verification(
        fresh_receipt_hash, provenance["PHYSICAL_SHA256"]
    )
    if (fresh_dir / "verification.txt").read_text(encoding="ascii") != fresh_verification:
        raise SystemExit("fresh verification transcript mismatch")
    fresh_result = invoke_verifier(
        fresh_receipt,
        provenance["SOURCE_SHA256"],
        provenance["INPUT_SHA256"],
        fresh_challenge,
    )
    if fresh_result.returncode != 0 or fresh_result.stderr:
        raise SystemExit("fresh receipt does not verify with repository verifier")
    if fresh_result.stdout != fresh_verification:
        raise SystemExit("fresh verifier output is not canonical")
    verify_content_manifest(fresh_dir / "dependencies.sha256", 592, True)
    verify_content_manifest(fresh_dir / "link-inputs.sha256", 19, False)
    verify_content_manifest(fresh_dir / "runtime-libraries.sha256", 4, False)

print("CS6_C1_DEPENDENCY_GATE_PASS=true")
print(f"RETAINED_RECEIPT_SHA256={provenance['RECEIPT_SHA256']}")
print(f"FRESH_RECEIPT_SHA256={fresh_receipt_hash}")
print(f"PHYSICAL_SHA256={provenance['PHYSICAL_SHA256']}")
print("MUTATIONS_REJECTED=39")
print("DEPENDENCY_COUNT=592")
print("LINK_INPUT_COUNT=19")
print("RUNTIME_LIBRARY_COUNT=4")
print("PROMOTION_ELIGIBLE=false")
PY
