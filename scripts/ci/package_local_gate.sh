#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="${SOUNIO_PACKAGE_LOCAL_GATE_ARTIFACT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-package-local-gate.XXXXXX")}"
LOG_DIR="$ARTIFACT_ROOT/logs"
SUMMARY="$ARTIFACT_ROOT/summary.v1.tsv"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"
CLAIM_REGISTRY="$ROOT_DIR/docs/serious-language/public-claim-registry.v1.tsv"
KNOWN_LIMITS="$ROOT_DIR/docs/compiler/KNOWN_LIMITATIONS.md"
REGISTRY_README="$ROOT_DIR/tools/registry/README.md"
REGISTRY_OPENAPI="$ROOT_DIR/tools/registry/openapi.yaml"

mkdir -p "$LOG_DIR"

run_capture() {
  local name="$1"
  shift
  local stdout="$LOG_DIR/$name.stdout"
  local stderr="$LOG_DIR/$name.stderr"
  set +e
  "$@" >"$stdout" 2>"$stderr"
  local rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "${stdout#$ARTIFACT_ROOT/}" "${stderr#$ARTIFACT_ROOT/}" >>"$SUMMARY"
  return "$rc"
}

require_contains() {
  local path="$1"
  local needle="$2"
  local label="$3"
  if ! grep -Fq "$needle" "$path"; then
    echo "package local gate failed: missing $label: $needle" >&2
    echo "  in: $path" >&2
    exit 1
  fi
}

printf 'step\texit\tstdout_log\tstderr_log\n' >"$SUMMARY"

if [[ ! -x "$ROOT_DIR/bin/souc" ]]; then
  echo "package local gate failed: bin/souc is missing or not executable" >&2
  exit 2
fi

if ! run_capture pkg-self-test env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/souc pkg self-test
then
  echo "package local gate failed: bin/souc pkg self-test failed" >&2
  exit 1
fi

if ! run_capture pkg-help env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/souc pkg --help
then
  echo "package local gate failed: bin/souc pkg --help failed" >&2
  exit 1
fi

require_contains "$LOG_DIR/pkg-self-test.stdout" "SPM self-tests: ALL PASSED" "SPM self-test success"
require_contains "$LOG_DIR/pkg-help.stdout" "souc pkg <subcommand>" "pkg help usage"

python3 - "$CLAIM_REGISTRY" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

registry = Path(sys.argv[1])
with registry.open(newline="", encoding="utf-8") as handle:
    rows = {row["claim_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

row = rows.get("tooling.package")
if row is None:
    raise SystemExit("package local gate failed: missing tooling.package registry row")
expected = {
    "claim_level": "prototype",
    "closure_status": "downgraded",
    "evidence_kind": "gate",
    "evidence_ref": "scripts/ci/package_local_gate.sh",
}
for key, value in expected.items():
    if (row.get(key) or "").strip() != value:
        raise SystemExit(f"package local gate failed: tooling.package {key}={row.get(key)!r}, expected {value!r}")
wording = (row.get("public_wording") or "").lower()
if "no public registry launch" not in wording:
    raise SystemExit("package local gate failed: tooling.package wording must forbid public registry launch")
PY

require_contains "$KNOWN_LIMITS" "No public registry." "known-limits public registry downgrade"
require_contains "$REGISTRY_README" "Prototype status" "registry README prototype status"
require_contains "$REGISTRY_README" "not a launched public service" "registry README launch disclaimer"
require_contains "$REGISTRY_OPENAPI" "Prototype API scaffold" "registry OpenAPI prototype status"

cat >"$RESULTS" <<EOF
# Sounio Package Local Gate

| Field | Value |
|---|---|
| artifact_root | \`$ARTIFACT_ROOT\` |
| status | \`pass\` |

This gate validates the release-supported package-manager scope:

- checked \`bin/souc\` exists and is executable
- \`bin/souc pkg self-test\` reports \`SPM self-tests: ALL PASSED\`
- \`bin/souc pkg --help\` exposes the package command surface
- \`tooling.package\` remains \`prototype / downgraded\` in the public claim registry
- public wording explicitly forbids claiming a launched public registry
- \`docs/compiler/KNOWN_LIMITATIONS.md\`, \`tools/registry/README.md\`, and
  \`tools/registry/openapi.yaml\` carry the same downgrade

This gate does not validate a public registry launch, remote package downloads,
or hosted registry support.
EOF

echo "Package local gate passed. See $RESULTS"
