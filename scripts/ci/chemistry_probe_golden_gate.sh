#!/usr/bin/env bash
# Golden-output gate for the Sounio replica-side probes of the GRI-Mech 3.0
# cross-validation (benchmarks/chemistry/RESULTS.md §2.6, §7.3, §7.7).  Each
# probe is run under lean_single -- the engine every number in RESULTS.md was
# produced on -- and its stdout is compared byte-for-byte with the committed
# golden file.  A diff means a number in the paper moved: re-derive it, do not
# regenerate the golden to make the gate pass (CLAUDE.md §6.6).
#   regenerate (only after the change is understood):
#     REGEN=1 bash scripts/ci/chemistry_probe_golden_gate.sh
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
GOLD="$ROOT_DIR/benchmarks/chemistry/golden"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/chem-probe-golden.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
PROBES=(rep_traj_bug rep_stagnation rep_adiabatic_bug)
FAILS=0
echo "[chem-golden] lean_single: $(md5sum "$ROOT_DIR/bin/souc-lean-single-x86_64" 2>/dev/null | cut -c1-8) bin/souc-lean-single-x86_64"
for p in "${PROBES[@]}"; do
  src="examples/chemistry/$p.sio"; gold="$GOLD/$p.lean_single.txt"; out="$WORK/$p.txt"
  if ! SOUNIO_SOUC_ENGINE=lean_single timeout 1500 "$SOUC" run "$src" >"$out" 2>"$WORK/$p.err"; then
    echo "[chem-golden] FAIL $p: run exited non-zero" >&2; tail -5 "$WORK/$p.err" >&2; FAILS=$((FAILS+1)); continue
  fi
  if [[ "${REGEN:-0}" == "1" ]]; then cp "$out" "$gold"; echo "[chem-golden] wrote $gold"; continue; fi
  if [[ ! -f "$gold" ]]; then echo "[chem-golden] FAIL $p: no golden at $gold" >&2; FAILS=$((FAILS+1)); continue; fi
  if diff -u "$gold" "$out" >"$WORK/$p.diff"; then echo "[chem-golden] ok   $p ($(wc -l <"$out") lines identical)"
  else echo "[chem-golden] FAIL $p: output differs from golden" >&2; head -40 "$WORK/$p.diff" >&2; FAILS=$((FAILS+1)); fi
done
[[ $FAILS -eq 0 ]] && echo "[chem-golden] all ${#PROBES[@]} probes match" || echo "[chem-golden] $FAILS probe(s) diverge" >&2
[[ $FAILS -eq 0 ]]
