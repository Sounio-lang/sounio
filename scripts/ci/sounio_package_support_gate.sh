#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="${SOUNIO_PACKAGE_SUPPORT_ARTIFACT_ROOT:-$(mktemp -d /tmp/sounio-package-support.XXXXXX)}"
mkdir -p "$ARTIFACT_ROOT"

log() {
  printf '[package-support] %s\n' "$*"
}

run_step() {
  local name="$1"
  shift
  local log_path="$ARTIFACT_ROOT/$name.log"
  log ">>> $name"
  if "$@" >"$log_path" 2>&1; then
    log "<<< $name PASS log=$log_path"
  else
    local rc=$?
    log "<<< $name FAIL rc=$rc log=$log_path" >&2
    sed -n '1,220p' "$log_path" >&2
    exit "$rc"
  fi
}

run_compiled_fixture() {
  local fixture="$1"
  local expect="$2"
  local name
  name="$(basename "$fixture" .sio)"
  local out_dir="$ARTIFACT_ROOT/$name"
  mkdir -p "$out_dir"

  local lean_souc="$ROOT_DIR/bin/souc-lean-single-x86_64"
  if [[ ! -x "$lean_souc" ]]; then
    lean_souc="$ROOT_DIR/bin/souc-linux-x86_64"
  fi
  if [[ ! -x "$lean_souc" ]]; then
    echo "lean_single compiler ELF not found" >&2
    return 1
  fi

  "$lean_souc" "$fixture" "$out_dir/main" >"$out_dir/compile.log" 2>&1
  chmod +x "$out_dir/main"
  "$out_dir/main" >"$out_dir/run.log" 2>&1
  grep -qF "$expect" "$out_dir/run.log"
}

run_sounio_pkg_smoke() {
  local smoke_dir="$ARTIFACT_ROOT/sounio-pkg-smoke"
  rm -rf "$smoke_dir"
  mkdir -p "$smoke_dir"

  "$ROOT_DIR/tools/sounio-pkg/sounio-pkg" version | tee "$smoke_dir/version.out"
  grep -qF 'sounio-pkg 0.1.0' "$smoke_dir/version.out"

  (
    cd "$smoke_dir"
    "$ROOT_DIR/tools/sounio-pkg/sounio-pkg" new smoke_pkg
    cd smoke_pkg
    "$ROOT_DIR/tools/sounio-pkg/sounio-pkg" build
    "$ROOT_DIR/tools/sounio-pkg/sounio-pkg" check
    "$ROOT_DIR/tools/sounio-pkg/sounio-pkg" test
  )
}

check_public_package_wording() {
  python3 - "$ROOT_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
required = {
    "docs/compiler/KNOWN_LIMITATIONS.md": [
        "Local `~/.sounio/registry/` only. No public registry.",
    ],
    "docs/ecosystem/REGISTRY_ARCHITECTURE.md": [
        "Status: design reference only; not launched as a public registry.",
    ],
    "docs/ecosystem/REGISTRY_ATTESTATION_SPEC.md": [
        "Status: executable R2.6 local policy contract; public registry publishing is disabled.",
        "unsigned-local-policy-evaluation",
        "publication-status = \"disabled\"",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_INVENTORY.md": [
        "Status: executable R3 ownership and file-identity inventory; physical extraction is not executed.",
        "physical-extraction-planning-snapshot",
        "r3-physical-extraction-materialization",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_MATERIALIZATION.md": [
        "Status: executable R3 local exact-copy boundary; canonical repository extraction is not executed.",
        "materialization_status = copied-and-verified",
        "source_removal_status = not-authorized",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION.md": [
        "Status: executable R3 temporary-copy authorization boundary; canonical source removal is not executed.",
        "authorization_status = authorized-not-executed",
        "source_removal_execution_status = not-executed",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION.md": [
        "Status: executable R3 policy-bound local execution interface; canonical repository cutover is not executed.",
        "execution_status = executed-and-verified",
        "source_removal_status = executed",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL.md": [
        "Status: executable R3 Git-state and rehearsal approval interface; canonical repository cutover is not executed.",
        "canonical_cutover_approval_status = approved-not-executed",
        "canonical_cutover_execution_status = not-executed",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION.md": [
        "Status: executable R3 policy-bound canonical Git cutover interface; exercised only in disposable fixtures for this repository.",
        "canonical_cutover_approval_status = consumed",
        "canonical_cutover_execution_status = executed-and-verified",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT.md": [
        "Status: executable R3 non-authorizing prerequisite observation; production policy, approval, human decision, and execution remain absent.",
        "execution_authority = none",
        "canonical_cutover_execution_status = not-executed",
    ],
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION.md": [
        "Status: executable R3 non-authorizing mapping-selection processing; repository creation, production approval, and cutover execution remain absent.",
        "execution_authority = none",
        "proposal_status = proposed-not-approved",
    ],
    "docs/ecosystem/SOUNIO_TOML_SPEC.md": [
        "Status: Draft/local package manifest contract; public registry publishing is not launched.",
    ],
    "docs/ecosystem/CURATED_PACKAGES.md": [
        "Status: roadmap/design list; these are not published public-registry packages.",
    ],
    "docs/guide/programming.md": [
        "Local package support is available through the checked `tools/sounio-pkg/sounio-pkg` wrapper",
    ],
    "docs/FAQ.md": [
        "There is no launched public package registry yet.",
    ],
    "tools/registry/README.md": [
        "This directory is a registry design/reference scaffold, not a launched official public registry.",
    ],
    "tools/sounio-pkg/README.md": [
        "Supported contract: local package creation, local build/check/test, and local package-import workflows.",
    ],
    "tools/pkg/README.md": [
        "Legacy experimental prototype; not part of the release support contract.",
    ],
}

errors = []
for rel, needles in required.items():
    text = (root / rel).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            errors.append(f"{rel}: missing required wording: {needle}")

for rel in [
    "docs/ecosystem/REGISTRY_ARCHITECTURE.md",
    "docs/ecosystem/REGISTRY_ATTESTATION_SPEC.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_INVENTORY.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_MATERIALIZATION.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT.md",
    "docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION.md",
    "docs/ecosystem/SOUNIO_TOML_SPEC.md",
    "docs/ecosystem/CURATED_PACKAGES.md",
    "docs/guide/programming.md",
    "docs/FAQ.md",
    "tools/registry/README.md",
    "tools/sounio-pkg/README.md",
    "tools/pkg/README.md",
]:
    text = (root / rel).read_text(encoding="utf-8")
    forbidden = [
        r"Production:\s*`https://registry",
        r"official package registry",
        r"\bsouc pkg publish\b",
        r"\bsounio-pkg publish\b",
        r"\bsounio-pkg login\b",
        r"\bsou pkg login\b",
        r"\bsou pkg search\b",
    ]
    for pat in forbidden:
        if re.search(pat, text, flags=re.IGNORECASE):
            errors.append(f"{rel}: forbidden launched-registry wording matched {pat!r}")

if errors:
    print("SOUNIO_PACKAGE_WORDING_FAIL", file=sys.stderr)
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

print("SOUNIO_PACKAGE_WORDING_PASS")
PY
}

echo 'SOUNIO_PACKAGE_SUPPORT_GATE_START'
echo "repo=$ROOT_DIR"
echo "artifact_root=$ARTIFACT_ROOT"

run_step package-import-science env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  bash "$ROOT_DIR/scripts/ci/package_import_science_gate.sh"
run_step pkg-manifest-fixture run_compiled_fixture tests/run-pass/pkg_manifest_parse_e2e.sio 'pkg_manifest_parse_e2e: ALL PASS'
run_step pkg-registry-fixture run_compiled_fixture tests/run-pass/pkg_registry_basic.sio 'pkg_registry_basic: ALL PASS'
run_step sounio-pkg-smoke run_sounio_pkg_smoke
run_step public-package-wording check_public_package_wording
run_step physical-extraction-inventory python3 "$ROOT_DIR/scripts/ci/physical_extraction_inventory_gate.py"
run_step physical-extraction-materialization python3 "$ROOT_DIR/scripts/ci/physical_extraction_materialization_gate.py"
run_step physical-extraction-source-removal-authorization \
  python3 "$ROOT_DIR/scripts/ci/physical_extraction_source_removal_authorization_gate.py"
run_step physical-extraction-source-removal-execution \
  python3 "$ROOT_DIR/scripts/ci/physical_extraction_source_removal_execution_gate.py"
run_step physical-extraction-canonical-cutover-approval \
  python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_cutover_approval_gate.py"
run_step physical-extraction-canonical-cutover-execution \
  python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_cutover_execution_gate.py"
run_step physical-extraction-canonical-production-gap \
    python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_production_gap_gate.py"
run_step physical-extraction-canonical-production-repository-catalog \
    python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_production_repository_catalog_gate.py"
run_step physical-extraction-canonical-production-mapping-decision \
  python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_production_mapping_decision_gate.py"

echo 'SOUNIO_PACKAGE_SUPPORT_GATE_PASS'
