#!/usr/bin/env bash
# Run all independent stages before aggregating. A failed stage never hides E2E.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_JSON="${STDLIB_RELIABILITY_STATUS_OUT:-$ROOT_DIR/artifacts/stdlib/stdlib_reliability_status.v1.json}"
E2E_JSON="${STDLIB_RELIABILITY_E2E_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_e2e_result.v1.json}"
INVENTORY_JSON="${STDLIB_RELIABILITY_INVENTORY_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_inventory.v1.json}"
SCIENCE_STATUS_JSON="${STDLIB_RELIABILITY_SCIENCE_STATUS_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_science_pipeline_status.v1.json}"
HYPER_STATUS_JSON="${STDLIB_RELIABILITY_HYPER_STATUS_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_hyper_execution_status.v1.json}"
while [[ $# -gt 0 ]];do
  case "$1" in
    --help|-h) echo "stdlib_reliability_gate.sh [--out-json PATH] [--e2e-json PATH] [--inventory-json PATH] [--science-json PATH] [--hyper-json PATH]";exit 0;;
    --out-json|--e2e-json|--inventory-json|--science-json|--hyper-json)
      [[ $# -ge 2 && -n $2 && $2 != --* ]] || { echo "missing path for $1" >&2;exit 2; }
      case "$1" in --out-json)OUT_JSON=$2;;--e2e-json)E2E_JSON=$2;;--inventory-json)INVENTORY_JSON=$2;;--science-json)SCIENCE_STATUS_JSON=$2;;--hyper-json)HYPER_STATUS_JSON=$2;;esac;shift 2;;
    *)echo "unknown argument: $1" >&2;exit 2;;
  esac
done
cd "$ROOT_DIR"
# Always create fresh stage paths so a failed producer cannot reuse an old pass.
work=$(mktemp -d "${TMPDIR:-/tmp}/stdlib-reliability.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
scan_rc=0;hyper_rc=0;e2e_rc=0
bash scripts/stdlib/scan_stdlib.sh --json-out "$work/inventory.json" --quiet || scan_rc=$?
STDLIB_HYPER_INVENTORY_JSON="$work/inventory.json" bash scripts/stdlib/stdlib_hyper_execution_gate.sh --out-json "$work/hyper.json" || hyper_rc=$?
bash scripts/stdlib/run_stdlib_e2e.sh --json-out "$work/e2e.json" || e2e_rc=$?
# Preserve a fresh explicit not_run receipt for missing stage output.
publish() {
  node -e 'const fs=require("fs"),p=require("path");const[a,b,s]=process.argv.slice(1);fs.mkdirSync(p.dirname(p.resolve(b)),{recursive:true});if(fs.existsSync(a))fs.copyFileSync(a,b);else fs.writeFileSync(b,JSON.stringify({schema:s,status_summary:"not_run",reason:"producer_did_not_write_fresh_output"})+"\n")' "$1" "$2" "$3"
}
publish "$work/inventory.json" "$INVENTORY_JSON" sounio.stdlib.inventory.v1
publish "$work/hyper.json" "$HYPER_STATUS_JSON" sounio.stdlib.hyper_execution_status.v1
publish "$work/e2e.json" "$E2E_JSON" sounio.stdlib.e2e.result.v1
aggregate_rc=0
node scripts/stdlib/reliability_status.mjs "$ROOT_DIR" "$E2E_JSON" "$INVENTORY_JSON" "$OUT_JSON" "$HYPER_STATUS_JSON" "$SCIENCE_STATUS_JSON" "$scan_rc" "$hyper_rc" "$e2e_rc" || aggregate_rc=$?
exit "$aggregate_rc"
