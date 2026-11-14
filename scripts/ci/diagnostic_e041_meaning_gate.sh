#!/usr/bin/env bash
# #3552: E041 has one meaning on Madaros; legacy lean_single is documented.
# With --souc, also check the emitted code/text on a source-built compiler.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC=""
if [[ $# -gt 0 ]]; then
  [[ $# -eq 2 && "$1" == --souc ]] || { echo "usage: $0 [--souc WRAPPER]" >&2; exit 2; }
  SOUC="$2"
fi

python3 - <<'PY'
from pathlib import Path
import re
import runpy

registry = Path('self-hosted/diagnostics/mod.sio').read_text()
checker = Path('self-hosted/check/check.sio').read_text()
parser = Path('self-hosted/parser/types.sio').read_text()
catalog = Path('docs/llm-guide/error-catalog.md').read_text()

def require(ok, message):
    if not ok:
        raise SystemExit('diagnostic_e041_meaning: FAIL ' + message)

for code, meaning in [(41, 'unit_mismatch'), (285, 'ontology_subsumption'),
                      (286, 'rust_mut_reference'), (287, 'use_after_move')]:
    require(re.search(rf'^let E{code:03}: i64 = {code}\s+// {meaning}$', registry, re.M),
            f'E{code:03} registry meaning must be {meaning}')
    require(re.search(rf'^\| E{code:03} \|', catalog, re.M), f'E{code:03} index row missing')
require('if code == E041 { return 1953066613 }' in registry, 'E041 name must pack unit')
require('var d = diag_error(E287)' in registry, 'use-after-move constructor must use E287')
require('d_uam.code == E287' in registry, 'registry self-test must pin E287')
require('else if code == 41 { print("unit mismatch") }' in checker,
        'E041 checker message must be unit mismatch')
require('else if code == 285 { print("ontology subsumption could not be verified") }' in checker,
        'ontology error must use E285')
require(len(re.findall(r'checker_report_error_at_inplace\([^\n]+, 285, 0, 0, 0\)', checker)) == 1,
        'ontology error emitter must use E285')
require(not re.search(r'(?:report_error_at[^\n]*|report_error\([^\n]*), 41, 0, 0, 0\)', checker),
        'ontology error must not reuse E041')
unit_prints = re.findall(r'print\("error\[E041\]: ([^"\n]+)', checker)
require(len(unit_prints) == 4 and all(s.startswith('unit mismatch') for s in unit_prints),
        'direct E041 checker messages must describe unit mismatch')
require('error[E286]: Sounio uses `&!T` for exclusive references, not `&mut T`' in parser,
        'Madaros &mut parser refusal must use E286')
require('error[E041]' not in parser, 'Madaros parser must not reuse E041')
explanation = Path('docs/llm-guide/explanations/E041.md').read_text()
require('title: Unit mismatch' in explanation and 'lean_single' in explanation,
        'E041 explanation must describe units and disclose legacy divergence')
get_error_doc = runpy.run_path('tools/mcp/sounio_mcp/errors_catalog.py')['get_error_doc']
for code, title in [('E041', 'Unit mismatch'), ('E285', 'Ontology subsumption'),
                    ('E286', 'Rust &mut reference syntax'), ('E287', 'Use after move')]:
    require(get_error_doc(code).startswith(f'# {code}: {title}'),
            f'{code} MCP explanation must agree with registry')
print('diagnostic_e041_meaning: STATIC_PASS E041=unit_mismatch E285=ontology E286=&mut E287=use_after_move')
PY

if [[ -n "$SOUC" ]]; then
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/diagnostic-e041.XXXXXX")"
  trap 'rm -rf "$WORK"' EXIT
  for fixture in tests/compile-fail/diagnostic_e041_unit_meaning.sio \
                 tests/compile-fail/diagnostic_e285_ontology_meaning.sio \
                 tests/compile-fail/diagnostic_e286_mut_ref.sio; do
    expected="$(sed -n 's|^//@ error-pattern: ||p' "$fixture")"
    rc=0
    "$SOUC" check "$fixture" >"$WORK/output" 2>&1 || rc=$?
    if [[ "$rc" -eq 0 || "$rc" -ge 128 ]] || ! grep -Fq "$expected" "$WORK/output"; then
      cat "$WORK/output" >&2
      echo "diagnostic_e041_meaning: FAIL $fixture rc=$rc expected=$expected" >&2
      exit 1
    fi
    if [[ "$fixture" == *ontology* ]]; then
      grep -Fq 'ontology subsumption could not be verified' "$WORK/output"
      grep -Fq 'strengthen the child property' "$WORK/output"
      if grep -Fq 'convert units' "$WORK/output"; then
        echo 'diagnostic_e041_meaning: FAIL ontology error still has unit help' >&2
        exit 1
      fi
    fi
    echo "diagnostic_e041_meaning: LIVE_PASS $fixture rc=$rc $expected"
  done
fi
