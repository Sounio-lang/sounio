#!/usr/bin/env bash
# Build of the BootstrapSporosVm library (the native Sporos VM `sporos_vm` simulates
# `Sporos.run`, and its TMR dispatch voters) and the axiom audit of its main theorems.
#
# Not run by the pull-request CI: the 28 modules take about 18 minutes of kernel time
# (12 minutes on four cores, measured 2026-11-22 on Lean 4.33.0), which the Lean Proofs
# job cannot absorb. Run it by hand or on the lab cluster.
#
# Every audited theorem must exist and depend on no axiom beyond propext,
# Classical.choice and Quot.sound.
#
#   bootstrap/model/sporos_vm_proofs.sh [--selftest]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LEAN_BIN:+$LEAN_BIN/}lake"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

THEOREMS=(
  SporosVmCore.table_ok
  SporosVmReady.mkVmReadyState_rel
  SporosVmSim.sporos_vm_step_sim
  SporosVmSim.sporos_vm_lrun_sim
  SporosVmSim.sporos_vm_sim_noPhys
  SporosVmSim.sporos_vm_sim_noPhys_noNomem
  SporosVmSim.sporos_vm_sim_checked_no_static_fail
  SporosVmFault.all_inline_voters_ok
  SporosVmFault.vm_dispatch_tri_phys_bound
  SporosVmFault.sporos_vm_dispatch_tmr_safe
  SporosVmWitness.witness_vm_echo
  SporosVmWitness.witness_vm_hello
  SporosVmWitness.witness_vm_upper
  SporosVmWitness.witness_vm_fib
  SporosVmWitness.witness_vm_provDemo_nonzero
  SporosVmWitness.witness_vm_provDemo_zero
  SporosVmWitness.witness_vm_provDemo_demand_fail
  SporosVmWitness.witness_tmr_vote_single_disturbance
  SporosVmWitness.witness_tmr_vote_three_way_exit3
)

# check_axioms LOG: exit 0 iff every theorem in THEOREMS appears exactly once and each
# rests only on the standard axioms (or none). Lean may wrap a long list over lines.
check_axioms() {
  local names
  names="$(printf '%s\n' "${THEOREMS[@]}")"
  awk -v names="$names" '
    BEGIN {
      n = split(names, want, "\n")
      for (i = 1; i <= n; i++) need[want[i]] = 1
      ok["propext"] = 1; ok["Classical.choice"] = 1; ok["Quot.sound"] = 1
      bad = 0; cur = ""
    }
    function close_entry() { if (cur != "") { cur = "" } }
    {
      line = $0; sub(/\r$/, "", line)
      if (line ~ /sorryAx|ofReduceBool|declaration uses/) bad = 1
      if (match(line, /^\x27[^\x27]+\x27/)) {
        name = substr(line, 2, RLENGTH - 2)
        rest = substr(line, RLENGTH + 1)
        if (!(name in need)) { bad = 1; next }
        seen[name]++
        if (rest == " does not depend on any axioms") { cur = ""; next }
        if (substr(rest, 1, 21) != " depends on axioms: [") { bad = 1; next }
        body = substr(rest, 22); cur = name
      } else if (cur != "") {
        body = line
      } else {
        if (line != "") bad = 1
        next
      }
      closed = (index(body, "]") > 0)
      sub(/\].*$/, "", body)
      k = split(body, toks, ",")
      for (i = 1; i <= k; i++) {
        t = toks[i]; gsub(/^[ \t]+|[ \t]+$/, "", t)
        if (t != "" && !(t in ok)) bad = 1
      }
      if (closed) cur = ""
    }
    END {
      if (cur != "") bad = 1
      for (i = 1; i <= n; i++) if (seen[want[i]] != 1) bad = 1
      exit bad
    }
  ' "$1"
}

selftest() {
  local t
  : > "$W/ok.log"
  for t in "${THEOREMS[@]}"; do
    echo "'$t' depends on axioms: [propext, Classical.choice, Quot.sound]" >> "$W/ok.log"
  done
  check_axioms "$W/ok.log" || { echo "FAIL: positive control rejected" >&2; return 1; }

  { printf "'%s' depends on axioms: [propext,\n  Classical.choice, Quot.sound]\n" "${THEOREMS[0]}"; tail -n +2 "$W/ok.log"; } > "$W/wrap.log"
  check_axioms "$W/wrap.log" || { echo "FAIL: wrapped list rejected" >&2; return 1; }

  sed '3s/Quot.sound\]/sorryAx]/' "$W/ok.log" > "$W/sorry.log"
  if check_axioms "$W/sorry.log"; then echo "FAIL: sorryAx accepted" >&2; return 1; fi
  sed '4s/Quot.sound\]/Lean.ofReduceBool]/' "$W/ok.log" > "$W/nd.log"
  if check_axioms "$W/nd.log"; then echo "FAIL: Lean.ofReduceBool accepted" >&2; return 1; fi
  sed '5s/Quot.sound\]/Foo.ax]/' "$W/ok.log" > "$W/custom.log"
  if check_axioms "$W/custom.log"; then echo "FAIL: custom axiom accepted" >&2; return 1; fi
  sed '6d' "$W/ok.log" > "$W/missing.log"
  if check_axioms "$W/missing.log"; then echo "FAIL: missing theorem accepted" >&2; return 1; fi
  { cat "$W/ok.log"; sed -n '2p' "$W/ok.log"; } > "$W/dup.log"
  if check_axioms "$W/dup.log"; then echo "FAIL: duplicated theorem accepted" >&2; return 1; fi
  printf "'%s.fake' depends on axioms: [propext]\n" "${THEOREMS[0]}" | cat "$W/ok.log" - > "$W/extra.log"
  if check_axioms "$W/extra.log"; then echo "FAIL: unexpected theorem accepted" >&2; return 1; fi

  echo "PASS: sporos_vm_proofs selftest"
}

case "${1:-}" in
  --selftest) selftest; exit $? ;;
  "") ;;
  *) echo "usage: $0 [--selftest]" >&2; exit 2 ;;
esac

selftest > /dev/null

start=$(date +%s)
( cd "$ROOT/formal" && "$LAKE" build BootstrapSporosVm )
echo "build ok ($(( $(date +%s) - start ))s)"

{
  echo "import Bootstrap.SporosVmBin"
  for t in "${THEOREMS[@]}"; do echo "#print axioms $t"; done
} > "$W/axioms.lean"
( cd "$ROOT/formal" && "$LAKE" env lean "$W/axioms.lean" ) > "$W/axioms.log" 2>&1 || {
  cat "$W/axioms.log" >&2; echo "FAIL: axiom query did not run" >&2; exit 1
}
cat "$W/axioms.log"
check_axioms "$W/axioms.log" || { echo "FAIL: a theorem is missing or rests on another axiom" >&2; exit 1; }
echo "SPOROS_VM_PROOFS_OK theorems=${#THEOREMS[@]}"
