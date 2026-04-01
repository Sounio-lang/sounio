#!/usr/bin/env bash
# sprint19_web_gate.sh — website and playground assets gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${SOUNIO_SPRINT19_WEB_GATE_OUT:-$ROOT_DIR/artifacts/sprint19/web_gate.v1.json}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint19_web_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

# --- File existence checks ---

if [ -f "$ROOT_DIR/website/public/wasm/sounio_compiler.js" ]; then
  record "website_wasm_sounio_compiler_js_exists" "pass" "found"
else
  record "website_wasm_sounio_compiler_js_exists" "not_run" "asset_not_yet_built"
fi

if [ -f "$ROOT_DIR/website/src/components/showcases/EpistemicFlowViz.tsx" ]; then
  record "website_epistemic_flow_viz_tsx_exists" "pass" "found"
else
  record "website_epistemic_flow_viz_tsx_exists" "not_run" "component_not_yet_created"
fi

if [ -f "$ROOT_DIR/website/src/components/SounioCode.astro" ]; then
  record "website_sounio_code_astro_exists" "pass" "found"
else
  record "website_sounio_code_astro_exists" "not_run" "component_not_yet_created"
fi

if [ -f "$ROOT_DIR/scripts/gen_stdlib_docs.sh" ] && [ -x "$ROOT_DIR/scripts/gen_stdlib_docs.sh" ]; then
  record "scripts_gen_stdlib_docs_sh_exists_executable" "pass" "found"
else
  record "scripts_gen_stdlib_docs_sh_exists_executable" "not_run" "script_not_yet_created"
fi

if [ -f "$ROOT_DIR/website/src/data/registry.json" ]; then
  record "website_data_registry_json_exists" "pass" "found"
else
  record "website_data_registry_json_exists" "not_run" "registry_not_yet_created"
fi

if [ -f "$ROOT_DIR/scripts/build_playground_wasm.sh" ] && [ -x "$ROOT_DIR/scripts/build_playground_wasm.sh" ]; then
  record "scripts_build_playground_wasm_sh_exists_executable" "pass" "found"
else
  record "scripts_build_playground_wasm_sh_exists_executable" "not_run" "script_not_yet_created"
fi

# --- PlaygroundEditor examples count ---

PLAYGROUND_TSX="$ROOT_DIR/website/src/components/playground/PlaygroundEditor.tsx"
if [ -f "$PLAYGROUND_TSX" ]; then
  example_count="$(grep -c "name:" "$PLAYGROUND_TSX" 2>/dev/null || echo 0)"
  if [ "$example_count" -ge 10 ]; then
    record "playground_editor_has_10_plus_examples" "pass" "count=$example_count"
  else
    record "playground_editor_has_10_plus_examples" "not_run" "count_below_10_or_not_yet_implemented"
  fi
else
  record "playground_editor_has_10_plus_examples" "not_run" "playground_editor_tsx_not_yet_created"
fi

python3 - "$OUT_JSON" "$CASES_TSV" << 'PY'
import json, datetime as dt, sys
from pathlib import Path

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])

cases = []
counts = {"pass": 0, "fail": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, reason = raw.split("\t", 2)
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({"name": name, "status": status, "reason": reason})

overall = "pass" if counts["fail"] == 0 and counts["not_run"] == 0 else \
          "not_run" if counts["fail"] == 0 else "fail"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint19.web_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "metrics": {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "not_run": counts["not_run"],
    },
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint19_web_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
