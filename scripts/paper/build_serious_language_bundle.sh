#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUC_BIN="${SOUNIO_SERIOUS_LANGUAGE_SOUC_BIN:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_SERIOUS_LANGUAGE_STDLIB_PATH:-$ROOT_DIR/stdlib}"

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_ROOT="${SOUNIO_SERIOUS_LANGUAGE_ARTIFACT_ROOT:-$ROOT_DIR/artifacts/conference-serious-language/$timestamp}"
LOG_DIR="$ARTIFACT_ROOT/logs"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"
MANIFEST="$ARTIFACT_ROOT/manifest.json"
FULL_MODE="${SOUNIO_SERIOUS_LANGUAGE_FULL:-0}"
FAILED_REQUIRED=0
FAILED_OPTIONAL=0

mkdir -p "$LOG_DIR"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))' 2>/dev/null || printf '"unavailable"'
}

file_sha256() {
  local file="$1"
  if [[ -f "$file" ]]; then
    sha256sum "$file" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$file" 2>/dev/null | awk '{print $1}'
  else
    printf 'not_found'
  fi
}

file_size() {
  local file="$1"
  if [[ -f "$file" ]]; then
    stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || printf 'unknown'
  else
    printf 'not_found'
  fi
}

tool_version() {
  local name="$1"
  shift
  if command -v "$name" >/dev/null 2>&1; then
    "$@" 2>&1 | head -n1
  else
    printf 'not_found'
  fi
}

run_step() {
  local requirement="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/$name.log"
  local display
  display="$(printf '%q ' "$@")"

  printf '## %s\n\n' "$name" >>"$RESULTS"
  printf '%s\n' "- requirement: \`$requirement\`" >>"$RESULTS"
  printf '%s\n' "- command: \`$display\`" >>"$RESULTS"

  echo "[serious-language] running $name"
  "$@" >"$log" 2>&1
  local rc=$?

  local log_rel="${log#$ARTIFACT_ROOT/}"
  printf '%s\n' "- exit_code: \`$rc\`" >>"$RESULTS"
  printf '%s\n\n' "- log: \`$log_rel\`" >>"$RESULTS"

  if [[ "$rc" -ne 0 ]]; then
    if [[ "$requirement" == "required" ]]; then
      FAILED_REQUIRED=1
    else
      FAILED_OPTIONAL=1
    fi
  fi
}

branch="$(git branch --show-current 2>/dev/null || true)"
commit="$(git rev-parse HEAD 2>/dev/null || true)"
host_os="$(uname -s 2>/dev/null || echo unknown)"
host_arch="$(uname -m 2>/dev/null || echo unknown)"
souc_version="$("$SOUC_BIN" --version 2>/dev/null || true)"
souc_info="$("$SOUC_BIN" info 2>/dev/null || true)"
selected_bin="$(printf '%s\n' "$souc_info" | awk -F= '/^selected_bin=/{print $2; exit}')"
souc_wrapper_sha256="$(file_sha256 "$SOUC_BIN")"
souc_wrapper_bytes="$(file_size "$SOUC_BIN")"
selected_bin_sha256="$(file_sha256 "$selected_bin")"
selected_bin_bytes="$(file_size "$selected_bin")"
git_version="$(tool_version git git --version)"
bash_version="${BASH_VERSION:-unknown}"
python_version="$(tool_version python3 python3 --version)"
lake_version="$(tool_version lake lake --version)"

cat >"$RESULTS" <<EOF
# Sounio Serious-Language Evidence Bundle

Generated at: $timestamp

| Field | Value |
|---|---|
| branch | \`$branch\` |
| commit | \`$commit\` |
| host | \`$host_os/$host_arch\` |
| souc_bin | \`$SOUC_BIN\` |
| souc_bin_sha256 | \`$souc_wrapper_sha256\` |
| stdlib_path | \`$SOUNIO_STDLIB_PATH\` |
| selected_bin | \`$selected_bin\` |
| selected_bin_sha256 | \`$selected_bin_sha256\` |
| full_mode | \`$FULL_MODE\` |

## Summary

This bundle is for conference and paper-readiness review. Required smoke steps must pass before claims are presented as stable. Optional/full steps may fail, but failures must downgrade or block the affected claims in \`docs/serious-language/readiness-ledger.md\`.

EOF

cat >"$MANIFEST" <<EOF
{
  "schema": "sounio.serious_language_bundle.v1",
  "generated_at": "$timestamp",
  "branch": $(printf '%s' "$branch" | json_escape),
  "commit": $(printf '%s' "$commit" | json_escape),
  "host_os": $(printf '%s' "$host_os" | json_escape),
  "host_arch": $(printf '%s' "$host_arch" | json_escape),
  "souc_bin": $(printf '%s' "$SOUC_BIN" | json_escape),
  "stdlib_path": $(printf '%s' "$SOUNIO_STDLIB_PATH" | json_escape),
  "souc_bin_sha256": $(printf '%s' "$souc_wrapper_sha256" | json_escape),
  "souc_bin_bytes": $(printf '%s' "$souc_wrapper_bytes" | json_escape),
  "selected_bin": $(printf '%s' "$selected_bin" | json_escape),
  "selected_bin_sha256": $(printf '%s' "$selected_bin_sha256" | json_escape),
  "selected_bin_bytes": $(printf '%s' "$selected_bin_bytes" | json_escape),
  "full_mode": $(printf '%s' "$FULL_MODE" | json_escape),
  "souc_version": $(printf '%s' "$souc_version" | json_escape),
  "souc_info": $(printf '%s' "$souc_info" | json_escape),
  "tool_versions": {
    "git": $(printf '%s' "$git_version" | json_escape),
    "bash": $(printf '%s' "$bash_version" | json_escape),
    "python3": $(printf '%s' "$python_version" | json_escape),
    "lake": $(printf '%s' "$lake_version" | json_escape)
  },
  "claim_coverage": {
    "checked_compiler_entry": ["souc-version.log", "souc-info.log"],
    "linux_x86_64_compile_run": ["hello-check.log", "hello-run.log"],
    "real_world_defensibility": ["docs/serious-language/real-world-defensibility.md", "docs/serious-language/readiness-ledger.md"],
    "spec_evidence_drift": ["spec-drift.log", "spec-drift/RESULTS.md", "spec-drift/summary.v1.json", "spec-drift/conformance.log", "spec-drift/conformance/summary.v1.tsv", "docs/serious-language/spec-evidence-matrix.v1.tsv"],
    "claim_closure": ["claim-closure.log", "claim-closure/RESULTS.md", "claim-closure/summary.v1.json", "docs/serious-language/public-claim-registry.v1.tsv", "docs/serious-language/doc-claim-surface.v1.tsv", "docs/serious-language/claim-line-annotations.v1.tsv"],
    "docs_governance": ["docs-consistency.log", "docs-registry.log"],
    "serious_conformance": ["serious-conformance.log", "serious-conformance/summary.v1.tsv", "serious-conformance/summary.v1.json"],
    "sounio_suite": ["sio-suite.log"],
    "selfhost": ["selfhost-host-gate.log"],
    "ontology": ["ontology-rebuilt.log"],
    "epistemic_gum": ["package-pbpk-gum.log", "stdlib-science.log"],
    "stdlib_reliability": ["stdlib-reliability.log"],
    "formal_lean_status": ["lean-sorry-audit.log", "lean-build.log"]
  }
}
EOF

cat >>"$RESULTS" <<EOF
## Tool Versions

| Tool | Version |
|---|---|
| git | \`$git_version\` |
| bash | \`$bash_version\` |
| python3 | \`$python_version\` |
| lake | \`$lake_version\` |

EOF

run_step required git-status git status --short --branch
run_step required souc-version "$SOUC_BIN" --version
run_step required souc-info "$SOUC_BIN" info
run_step required hello-check "$SOUC_BIN" check examples/hello.sio
run_step required hello-run "$SOUC_BIN" run examples/hello.sio
run_step required docs-consistency bash scripts/dev/check_docs_consistency.sh
run_step required docs-registry bash scripts/dev/check_docs_registry.sh
run_step required spec-drift env SOUNIO_SPEC_DRIFT_ARTIFACT_ROOT="$ARTIFACT_ROOT/spec-drift" bash scripts/ci/serious_language_spec_drift_gate.sh
run_step required claim-closure env SOUNIO_CLAIM_CLOSURE_ARTIFACT_ROOT="$ARTIFACT_ROOT/claim-closure" bash scripts/ci/serious_language_claim_closure_gate.sh

if [[ "$FULL_MODE" == "1" ]]; then
  run_step optional serious-conformance env SOUNIO_SERIOUS_CONFORMANCE_ARTIFACT_ROOT="$ARTIFACT_ROOT/serious-conformance" SOUNIO_SERIOUS_CONFORMANCE_SOUC_BIN="$SOUC_BIN" SOUNIO_SERIOUS_CONFORMANCE_STDLIB_PATH="$SOUNIO_STDLIB_PATH" bash scripts/ci/serious_language_conformance_gate.sh
  run_step optional sio-suite bash scripts/run_sio_test_suite.sh --jobs "${SOUNIO_TEST_JOBS:-4}"
  run_step optional selfhost-host-gate bash scripts/ci/selfhost_host_gate.sh
  run_step optional ontology-rebuilt bash scripts/ci/run_ontology_validation.sh --mode rebuilt ontology
  run_step optional package-pbpk-gum bash scripts/ci/package_pbpk_gum_gate.sh
  run_step optional stdlib-science bash scripts/stdlib_science_pipeline_gate.sh
  run_step optional stdlib-reliability bash scripts/stdlib_reliability_gate.sh
  run_step optional lean-sorry-audit python3 scripts/ci/lean_proof_status_audit.py
  if command -v lake >/dev/null 2>&1; then
    run_step optional lean-build bash -lc 'cd formal/lean4 && lake build'
  else
    printf '## lean-build\n\n- requirement: `optional`\n- command: `cd formal/lean4 && lake build`\n- exit_code: `not_run`\n- reason: `lake not found on PATH`\n\n' >>"$RESULTS"
    FAILED_OPTIONAL=1
  fi
fi

cat >>"$RESULTS" <<EOF
## Claim Review Checklist

- Compare every external claim against \`docs/serious-language/readiness-ledger.md\`.
- Confirm public PL claims close through \`docs/serious-language/public-claim-registry.v1.tsv\` and \`docs/serious-language/doc-claim-surface.v1.tsv\`.
- Confirm high-value public claims have exact anchors in \`docs/serious-language/claim-line-annotations.v1.tsv\`.
- Downgrade any claim whose supporting command failed or was not run.
- For external-facing papers/slides/abstracts, run the required \`bin/llm-offload\` review and log the result.
- Record remaining blockers before submission.

## Final Status

| Required failures | Optional/full-lane failures |
|---|---|
| \`$FAILED_REQUIRED\` | \`$FAILED_OPTIONAL\` |

EOF

echo "Wrote serious-language bundle to $ARTIFACT_ROOT"

if [[ "$FAILED_REQUIRED" -ne 0 ]]; then
  echo "One or more required serious-language evidence steps failed." >&2
  exit 1
fi

if [[ "$FAILED_OPTIONAL" -ne 0 ]]; then
  echo "Optional/full-lane steps had failures or were not run; inspect RESULTS.md." >&2
fi

exit 0
