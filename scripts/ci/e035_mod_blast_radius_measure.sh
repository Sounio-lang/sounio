#!/usr/bin/env bash
# Static re-run of the E035 / Mod blast-radius census.
# Does not apply the substitution. Does not edit self-hosted/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${E035_MOD_BLAST_OUT:-/tmp/e035_mod_static.json}"
python3 "$ROOT/scripts/dev/e035_mod_blast_radius.py" --root "$ROOT" --json-out "$OUT"
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cg = d["call_graph"]
print(
    "E035_MOD_BLAST "
    f"sha={d['sha'][:12]} "
    f"mod_fns={cg['affected_functions']} "
    f"closure_fns={cg['closure_functions']} "
    f"need_mut={cg['closure_need_new_mut']} "
    f"need_files={len(d['need_mut_files'])} "
    f"depth={cg['max_depth']} "
    f"compiler={cg['reaches_compiler']}"
)
print("status=measured")
print(
    "metrics "
    f"{{total={cg['closure_functions']}, "
    f"passed={cg['closure_get_mut_from_mod']}, "
    f"failed={cg['closure_need_new_mut']}, "
    f"not_run=0}}"
)
PY
