#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${SOUNIO_CLAUDE_CONTRACT_OUT:-$ROOT_DIR/artifacts/omega/claude_operational_contract_status.v1.json}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_PATH")"

run_check() {
  local name="$1"
  shift
  local log="$TMP_DIR/${name}.log"
  if "$@" >"$log" 2>&1; then
    printf '{"name":"%s","status":"pass","log":"%s"}' "$name" "$log"
    return 0
  fi
  printf '{"name":"%s","status":"fail","log":"%s"}' "$name" "$log"
  return 1
}

CHECKS=()

check_json="$(run_check prompt_execution_contract bash scripts/check_prompt_execution_contract.sh)"
CHECKS+=("$check_json")

check_json="$(run_check plan_consistency bash scripts/check_claude_plan_consistency.sh)"
CHECKS+=("$check_json")

checks_joined=""
for item in "${CHECKS[@]}"; do
  if [[ -n "$checks_joined" ]]; then
    checks_joined+=","
  fi
  checks_joined+="$item"
done

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -cn \
  --arg generated_at_utc "$generated_at_utc" \
  --argjson checks "[$checks_joined]" \
  '{
    schema: "sounio.claude.operational.contract.v1",
    generated_at_utc: $generated_at_utc,
    canonical_precedence: [
      "PLAN_ORIGINAL.md",
      ".claude/offload-specs/*.md",
      "artifacts/omega/selfhost_compiler_progress.v1.json",
      "artifacts/omega/parallel_cutover_status.v1.json"
    ],
    locked_track_b_order: [
      "data_structures.md",
      "gpu_ir_expansion.md",
      "hlir_lowering.md",
      "metal_msl_codegen.md",
      "ptx_regalloc_expansion.md"
    ],
    checks: $checks,
    status: (if ($checks | all(.status == "pass")) then "pass" else "fail" end)
  }' >"$OUT_PATH"

if ! jq -e '.status == "pass"' "$OUT_PATH" >/dev/null; then
  echo "error: claude operational contract gate failed" >&2
  cat "$OUT_PATH" >&2
  exit 1
fi

echo "CLAUDE_OPERATIONAL_CONTRACT_PASS out=$OUT_PATH"
