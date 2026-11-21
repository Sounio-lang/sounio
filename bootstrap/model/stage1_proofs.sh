#!/usr/bin/env bash
# Kernel-checked proofs of the stage-1 checker over its real image: the
# BootstrapSporosStage1 library (formal/Bootstrap/SporosX1Final*.lean). Not built
# on every pull request (tens of minutes, several GB); run it on any pull request
# that changes those files or what they import (X86Int, X86Dec, X86Env, Sporos).
#
# Checks:
#   1. `lake build BootstrapSporosStage1` succeeds.
#   2. SporosX1Final.lean, compiled on its own, warns `declaration uses 'sorry'`
#      exactly as many times as formal/AXIOM_INVENTORY.md records for it.
#   3. Of the three targets (stage1_oracle_exists_target, stage1_final_target,
#      stage1_final_exit2_target), every one that does not rest on sorryAx rests
#      on no axiom beyond propext, Classical.choice and Quot.sound, and their
#      number is 3 minus the sorry count.
#
#   bootstrap/model/stage1_proofs.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
lake="${LEAN_BIN:+$LEAN_BIN/}lake"
if ! command -v "$lake" >/dev/null 2>&1 && [[ -x "$HOME/.elan/bin/lake" ]]; then
  lake="$HOME/.elan/bin/lake"
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/stage1_proofs.XXXXXX")"
trap 'rm -rf "$work"' EXIT
fail() { echo "stage1_proofs: FAIL: $*" >&2; echo "STAGE1_PROOFS_FAIL"; exit 1; }
say() { echo "stage1_proofs: $*"; }

cd "$root/formal"
start=$(date +%s)
"$lake" build BootstrapSporosStage1 > "$work/build.log" 2>&1 || { tail -c 3000 "$work/build.log"; fail "lake build BootstrapSporosStage1"; }
say "build ok ($(( $(date +%s) - start ))s)"

"$lake" env lean Bootstrap/SporosX1Final.lean > "$work/final.log" 2>&1 || { cat "$work/final.log"; fail "SporosX1Final.lean"; }
got=$(grep -c "declaration uses .sorry." "$work/final.log" || true)
want=$(grep -F '| `formal/Bootstrap/SporosX1Final.lean` |' "$root/formal/AXIOM_INVENTORY.md" | grep -oE '\| [0-9]+ \|$' | grep -oE '[0-9]+' || echo 0)
[[ "$got" == "$want" ]] || fail "SporosX1Final.lean has $got sorry, AXIOM_INVENTORY.md records $want"
say "sorry count $got (matches AXIOM_INVENTORY.md)"

printf '%s\n' 'import Bootstrap.SporosX1Final' \
  '#print axioms SporosX1.stage1_oracle_exists_target' \
  '#print axioms SporosX1.stage1_final_target' \
  '#print axioms SporosX1.stage1_final_exit2_target' > "$work/axioms.lean"
"$lake" env lean "$work/axioms.lean" > "$work/axioms.log" 2>&1 || { cat "$work/axioms.log"; fail "axiom audit"; }
cat "$work/axioms.log"
clean=$(grep -c "depends on axioms: \[propext, Classical.choice, Quot.sound\]$" "$work/axioms.log" || true)
dirty=$(grep -c "sorryAx" "$work/axioms.log" || true)
[[ $(( clean + dirty )) == 3 ]] || fail "a target rests on an axiom other than the standard three or sorryAx"
[[ "$clean" == $(( 3 - got )) ]] || fail "$clean proved targets, expected $(( 3 - got ))"
say "proved targets: $clean of 3, each on propext, Classical.choice, Quot.sound only"
echo "STAGE1_PROOFS_OK"
