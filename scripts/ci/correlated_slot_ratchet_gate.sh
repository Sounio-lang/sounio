#!/usr/bin/env bash
# R6: freeze Knowledge slot-identity debt that would require `with Correlated`.
# May only shrink. See correlated_slot_ratchet.frozen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FROZEN="$ROOT/scripts/ci/correlated_slot_ratchet.frozen"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "correlated_slot_ratchet"

python3 - <<'PY'
import re
from pathlib import Path
meas = re.compile(r'\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*Knowledge[^=]*)?=\s*measure\s*\(')
know_let = re.compile(r'\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*Knowledge')
op_pat = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*([+*/])\s*\1\b')
skip_parts = {'.git', 'archive', 'bootstrap', 'docs', 'tests'}
sites = []
for p in Path('.').rglob('*.sio'):
    if any(x in p.parts for x in skip_parts):
        continue
    try:
        t = p.read_text(errors='ignore')
    except Exception:
        continue
    names = set()
    for m in meas.finditer(t):
        names.add(m.group(1))
    for m in know_let.finditer(t):
        names.add(m.group(1))
    if not names:
        continue
    for i, line in enumerate(t.splitlines(), 1):
        if line.strip().startswith('//'):
            continue
        for m in op_pat.finditer(line):
            if m.group(1) in names:
                # Heuristic: skip obvious f64 helper params (abs_f64 etc.)
                if 'fn ' in t[max(0, t.find(line)-200):t.find(line)] and f'({m.group(1)}: f64)' in t:
                    continue
                sites.append(f'{p}:{i}:{m.group(0)}')
# Filter known false positives: pure f64 math helpers in same file as Knowledge comments
filtered = []
for s in sites:
    path = s.split(':')[0]
    text = Path(path).read_text(errors='ignore')
    # if the binding is clearly f64 function param, drop
    name = s.rsplit(':', 1)[-1].split()[0]
    if re.search(rf'fn\s+\w+\s*\([^)]*\b{re.escape(name)}\s*:\s*f64', text):
        continue
    filtered.append(s)
print(len(filtered))
for s in filtered:
    print(s)
PY
COUNT=$(python3 - <<'PY'
import re
from pathlib import Path
meas = re.compile(r'\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*Knowledge[^=]*)?=\s*measure\s*\(')
know_let = re.compile(r'\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*Knowledge')
op_pat = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*([+*/])\s*\1\b')
skip_parts = {'.git', 'archive', 'bootstrap', 'docs', 'tests'}
sites = []
for p in Path('.').rglob('*.sio'):
    if any(x in p.parts for x in skip_parts):
        continue
    try:
        t = p.read_text(errors='ignore')
    except Exception:
        continue
    names = set(m.group(1) for m in meas.finditer(t)) | set(m.group(1) for m in know_let.finditer(t))
    if not names:
        continue
    for i, line in enumerate(t.splitlines(), 1):
        if line.strip().startswith('//'):
            continue
        for m in op_pat.finditer(line):
            if m.group(1) in names:
                sites.append((str(p), m.group(1), s:=f'{p}:{i}:{m.group(0)}'))
filtered = []
for path, name, s in sites:
    text = Path(path).read_text(errors='ignore')
    if re.search(rf'fn\s+\w+\s*\([^)]*\b{re.escape(name)}\s*:\s*f64', text):
        continue
    filtered.append(s)
print(len(filtered))
PY
)

require_nonempty "$COUNT" "the live slot-identity scan produced no count — the extraction broke, this is not a measured zero"
FROZEN_TOTAL=$(awk -F= '/^total=/{print $2}' "$FROZEN")
require_nonempty "$FROZEN_TOTAL" "no total= line in $FROZEN — the frozen baseline is unreadable, not zero"
echo "correlated_slot_ratchet: live=$COUNT frozen=$FROZEN_TOTAL"
if [[ "$COUNT" -gt "$FROZEN_TOTAL" ]]; then
  echo "FAIL: slot-identity Knowledge debt rose ($COUNT > $FROZEN_TOTAL). Migrate or raise only with founder leave." >&2
  exit 1
fi
if [[ "$COUNT" -lt "$FROZEN_TOTAL" ]]; then
  echo "NOTE: debt shrank ($COUNT < $FROZEN_TOTAL). Lower total= in correlated_slot_ratchet.frozen."
fi
echo "CORRELATED_SLOT_RATCHET_OK"
