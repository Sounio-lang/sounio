#!/usr/bin/env bash
# Regeneratable repo scale measurements for README, llms.txt, and audit docs.
# Usage:
#   bash scripts/dev/measure_repo_scale.sh
#   bash scripts/dev/measure_repo_scale.sh --json artifacts/audit/repo_scale.v1.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${1:-}"
if [[ "${1:-}" == "--json" ]]; then
  OUT_JSON="${2:-$ROOT_DIR/artifacts/audit/repo_scale.v1.json}"
fi

count_sio_lines() {
  local nfiles=0
  local nlines=0
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    nfiles=$((nfiles + 1))
    nlines=$((nlines + $(wc -l < "$f")))
  done < <(git ls-files '*.sio')
  printf '%s %s\n' "$nfiles" "$nlines"
}

subdir_lines() {
  local prefix="$1"
  local nfiles=0
  local nlines=0
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    nfiles=$((nfiles + 1))
    nlines=$((nlines + $(wc -l < "$f")))
  done < <(git ls-files "${prefix}/**/*.sio" 2>/dev/null || true)
  printf '%s %s\n' "$nfiles" "$nlines"
}

read -r SIO_FILES SIO_LINES <<< "$(count_sio_lines)"
read -r SH_FILES SH_LINES <<< "$(subdir_lines self-hosted)"
read -r STDLIB_FILES STDLIB_LINES <<< "$(subdir_lines stdlib)"
read -r TEST_FILES TEST_LINES <<< "$(subdir_lines tests)"
read -r EXAMPLE_FILES EXAMPLE_LINES <<< "$(subdir_lines examples)"
read -r BOOTSTRAP_FILES BOOTSTRAP_LINES <<< "$(subdir_lines bootstrap)"

GATE_SCRIPTS="$(find scripts/ci -name '*gate*.sh' 2>/dev/null | wc -l | tr -d ' ')"
REPO_DISK="$(du -sh . 2>/dev/null | awk '{print $1}')"
GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

if [[ -f artifacts/audit/stdlib_module_audit_refined.v1.json ]] && command -v python3 >/dev/null 2>&1; then
  STDLIB_AUDIT="$(python3 - <<'PY'
import json
from collections import Counter
p = "artifacts/audit/stdlib_module_audit_refined.v1.json"
data = json.load(open(p))
mods = []
for cluster, items in data.get("clusters", {}).items():
    for m in items:
        mods.append(m)
c = Counter(m.get("status", "?") for m in mods)
print(json.dumps({"modules": len(mods), "works": c.get("works", 0), "scaffold": c.get("scaffold", 0), "missing": c.get("missing", 0)}))
PY
)"
else
  STDLIB_AUDIT='{"modules": null, "works": null, "scaffold": null, "missing": null}'
fi

if [[ -f artifacts/audit/ci_gates_works_wiring_a4.v1.json ]] && command -v python3 >/dev/null 2>&1; then
  GATE_WIRING="$(python3 - <<'PY'
import json
p = "artifacts/audit/ci_gates_works_wiring_a4.v1.json"
s = json.load(open(p)).get("summary", {})
print(json.dumps(s))
PY
)"
else
  GATE_WIRING='{}'
fi

if [[ -f artifacts/stdlib/stdlib_reliability_status.v1.json ]]; then
  STDLIB_RELIABILITY="$(python3 - <<'PY'
import json
d = json.load(open("artifacts/stdlib/stdlib_reliability_status.v1.json"))
inv = d.get("inventory", {})
print(json.dumps({
    "pass": inv.get("pass"),
    "fail": inv.get("fail"),
    "skip": inv.get("skip"),
    "total": inv.get("total"),
    "note": "stdlib_reliability_gate inventory; not the same as 814/910 README badge"
}))
PY
 2>/dev/null || echo '{}')"
else
  STDLIB_RELIABILITY='{}'
fi

mkdir -p "$(dirname "${OUT_JSON:-/dev/null}")"

emit_json() {
  cat <<EOF
{
  "schema": "sounio.repo_scale.v1",
  "generated_at_utc": "$GENERATED_AT",
  "git": {"branch": "$BRANCH", "commit": "$COMMIT"},
  "tracked_sio": {
    "files": $SIO_FILES,
    "lines_raw": $SIO_LINES
  },
  "subtrees": {
    "self_hosted": {"files": $SH_FILES, "lines": $SH_LINES},
    "stdlib": {"files": $STDLIB_FILES, "lines": $STDLIB_LINES},
    "tests": {"files": $TEST_FILES, "lines": $TEST_LINES},
    "examples": {"files": $EXAMPLE_FILES, "lines": $EXAMPLE_LINES},
    "bootstrap": {"files": $BOOTSTRAP_FILES, "lines": $BOOTSTRAP_LINES}
  },
  "ci_gates": {"scripts_ci_gate_sh": $GATE_SCRIPTS},
  "repo_disk": "$REPO_DISK",
  "stdlib_audit": $STDLIB_AUDIT,
  "gate_wiring_audit_a4": $GATE_WIRING,
  "stdlib_reliability": $STDLIB_RELIABILITY,
  "regenerate": "bash scripts/dev/measure_repo_scale.sh --json artifacts/audit/repo_scale.v1.json"
}
EOF
}

if [[ -n "$OUT_JSON" ]]; then
  emit_json | python3 -m json.tool > "$OUT_JSON"
  echo "[measure_repo_scale] wrote $OUT_JSON" >&2
fi

cat <<EOF

Sounio repo scale ($(git rev-parse --short HEAD 2>/dev/null || echo ?))
  tracked .sio: $SIO_FILES files, $SIO_LINES lines (git ls-files sum)
  self-hosted/: $SH_FILES files, $SH_LINES lines
  stdlib/:      $STDLIB_FILES files, $STDLIB_LINES lines
  tests/:       $TEST_FILES files, $TEST_LINES lines
  examples/:    $EXAMPLE_FILES files, $EXAMPLE_LINES lines
  CI gate scripts: $GATE_SCRIPTS under scripts/ci/
  working tree disk: $REPO_DISK

Do not describe this repo as "~200k LOC" or "stdlib-only size".
See docs/audit/README.md and llms.txt § Repository scale.
EOF
