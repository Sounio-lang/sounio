#!/usr/bin/env bash
# A witness must declare what would kill it.
#
# Why this exists
# ---------------
# souc-build-remote.sh --gate witness has carried a positive control since the
# i256 work: build the fixture with SOUNIO_WIDE_MUL_SABOTAGE=1, and if it still
# passes, refuse to certify it. That control is correct and it is the only
# instrument in this repository that reports honestly about its own blind spot.
# It is also, on its own, unusable outside one family: hand it an epistemic
# fixture and the sabotage is inert, the fixture passes anyway, and the gate
# says CONTROL_FAIL -- which is true, and useless, because there is no way to
# tell it what SHOULD have killed this particular witness.
#
# So the sabotage moves from the gate to the witness:
#
#   //@ sabotage: quotient-hessian
#
# The gate then needs to know nothing about domains. It reads the declaration,
# sets the matching switch, rebuilds, and requires the witness to FAIL. A
# witness that passes under its own declared mutilation is not measuring what
# it claims to measure.
#
# What this gate does NOT do
# --------------------------
# It does not fail an undeclared witness. Most of the corpus declares nothing,
# and pretending otherwise would turn one honest red into hundreds of
# meaningless ones. Undeclared witnesses are counted as UNVERIFIED and written
# to the artifact. That count is the number this gate exists to move.
#
#   SOUNIO_WITNESS_SABOTAGE_MADAROS=/path/to/madaros   # required
#   SOUNIO_WITNESS_SABOTAGE_CENSUS_ONLY=1              # skip the builds
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "witness_declares_its_sabotage"

# token -> environment switch. Adding a row here is the only coupling between
# this gate and the compiler; the witnesses name tokens, never variables.
sab_env_for() {
  case "$1" in
    quotient-hessian) echo "SOUNIO_SABOTAGE_QUOTIENT_HESSIAN" ;;
    hessian-chain)    echo "SOUNIO_SABOTAGE_HESSIAN_CHAIN" ;;
    correlate-rho)    echo "SOUNIO_SABOTAGE_CORRELATE_RHO" ;;
    wide-mul)         echo "SOUNIO_WIDE_MUL_SABOTAGE" ;;
    *)                echo "" ;;
  esac
}

ART_DIR="$ROOT_DIR/artifacts/gates"
mkdir -p "$ART_DIR"
ART="$ART_DIR/witness_declares_its_sabotage.json"

mapfile -t ALL_WITNESSES < <(find tests/run-pass -name '*.sio' -type f | sort)
declared=(); undeclared=()
for w in "${ALL_WITNESSES[@]}"; do
  tok="$(grep -m1 -oE '^//@ sabotage:[[:space:]]*[a-z0-9-]+' "$w" 2>/dev/null | awk '{print $NF}')"
  if [[ -n "$tok" ]]; then declared+=("$w|$tok"); else undeclared+=("$w"); fi
done

total_w=${#ALL_WITNESSES[@]}
n_decl=${#declared[@]}
n_undecl=${#undeclared[@]}

# Non-vacuity, and it is not a formality.
#
# The first run of this gate reported `status=pass declared=0 ... of=0` and
# printed OK, because tests/ was missing from the remote payload. A gate whose
# entire purpose is refusing witnesses that measure nothing passed by measuring
# nothing. There is no version of that which is acceptable, so an empty corpus
# is a hard failure and a corpus with no declarations is one too: both mean the
# gate ran and learned nothing.
if [[ $total_w -eq 0 ]]; then
  gate_fail "inspected ZERO witnesses -- tests/run-pass is missing or unreadable from $(pwd)"
fi
if [[ $n_decl -eq 0 ]]; then
  gate_fail "$total_w witnesses found and not one declares a sabotage -- the declarations were lost, not absent"
fi

# The census is the point even when the builds are skipped.
if [[ "${SOUNIO_WITNESS_SABOTAGE_CENSUS_ONLY:-0}" == "1" ]]; then
  # passed=0, not_run=total. Census mode executes nothing, and an artifact that
  # reports passes it did not observe is the exact defect this gate exists to
  # find -- it was written that way first, and caught in review of its own JSON.
  printf '{"status":"pass","mode":"census","metrics":{"total":%d,"declared":%d,"unverified":%d,"passed":0,"failed":0,"not_run":%d}}\n' \
    "$total_w" "$n_decl" "$n_undecl" "$total_w" > "$ART"
  echo "witness_declares_its_sabotage: census only -- $n_decl declared, $n_undecl unverified, of $total_w"
  gate_pass "census written to $ART"
  exit 0
fi

MADAROS="${SOUNIO_WITNESS_SABOTAGE_MADAROS:-}"
if [[ -z "$MADAROS" || ! -x "$MADAROS" ]]; then
  echo "witness_declares_its_sabotage: SOUNIO_WITNESS_SABOTAGE_MADAROS must name an executable Madaros ELF"
  echo "  (run with SOUNIO_WITNESS_SABOTAGE_CENSUS_ONLY=1 for the declaration census alone)"
  gate_fail "no compiler to run the sabotage against"
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
WORK="$(mktemp -d /tmp/witness-sabotage.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
ulimit -s 524288 2>/dev/null || true

total=0; passed=0; failed=0; unknown_token=0; broken=0
for row in "${declared[@]}"; do
  src="${row%%|*}"; tok="${row##*|}"
  total=$((total + 1))
  env_var="$(sab_env_for "$tok")"
  if [[ -z "$env_var" ]]; then
    echo "  UNKNOWN-TOKEN  $src declares '$tok', which no switch implements"
    unknown_token=$((unknown_token + 1)); failed=$((failed + 1)); continue
  fi

  # How this witness reports death.
  #
  # Not the exit code. Most of the corpus signals failure by printing a marker
  # the suite matches with `//@ expect-stdout`, and returns 0 either way. The
  # first version of this gate judged by exit status alone and reported
  # SURVIVED for a witness whose sabotaged run printed
  # MADAROS_HESSIAN_TRANSCENDENTAL_FAIL with h_sin=0.000000 -- it had died
  # exactly as declared, said so, and was accused of not measuring anything.
  # A wrong notion of death blames the measured thing for the instrument.
  marker="$(grep -m1 -oE '^//@ expect-stdout(-contains)?:[[:space:]]*\S+' "$src" 2>/dev/null | awk '{print $NF}')"

  # Control A -- unsabotaged, must PASS. A witness that is already red proves
  # nothing when it goes red under sabotage.
  out="$WORK/clean.elf"
  if ! timeout 300 "$MADAROS" build "$src" "$out" >"$WORK/b.log" 2>&1; then
    echo "  CLEAN-BUILD-FAIL  $src did not compile unsabotaged"
    broken=$((broken + 1)); failed=$((failed + 1)); continue
  fi
  chmod +x "$out"
  timeout 120 "$out" >"$WORK/r.log" 2>&1
  clean_rc=$?
  if [[ $clean_rc -ne 0 ]] || { [[ -n "$marker" ]] && ! grep -qF "$marker" "$WORK/r.log"; }; then
    echo "  CLEAN-RUN-FAIL    $src fails without any sabotage (rc=$clean_rc)"
    sed 's/^/      clean| /' "$WORK/r.log" | head -8
    broken=$((broken + 1)); failed=$((failed + 1)); continue
  fi

  # Control B -- sabotaged, must FAIL. This is the whole gate.
  bad="$WORK/bad.elf"
  if ! env "$env_var=1" timeout 300 "$MADAROS" build "$src" "$bad" >"$WORK/b2.log" 2>&1; then
    # A refused compile is a legitimate death: the witness did not survive.
    echo "  ok   $src  died at compile under $tok"
    passed=$((passed + 1)); continue
  fi
  chmod +x "$bad"
  env "$env_var=1" timeout 120 "$bad" >"$WORK/r2.log" 2>&1
  bad_rc=$?
  survived=0
  if [[ $bad_rc -eq 0 ]]; then
    if [[ -z "$marker" ]]; then
      survived=1
    elif grep -qF "$marker" "$WORK/r2.log"; then
      survived=1
    fi
  fi
  if [[ $survived -eq 1 ]]; then
    echo "  SURVIVED  $src passed with $tok sabotaged -- it does not measure what it declares"
    # Print what it printed. A survival has two causes -- a witness that does
    # not discriminate, or a switch that does not bite -- and they are not
    # distinguishable without the numbers. Discarding this output would make
    # the gate accuse the witness for the gate's own defect.
    sed 's/^/      sabotaged| /' "$WORK/r2.log" | head -12
    sed 's/^/      clean    | /' "$WORK/r.log" | head -12
    failed=$((failed + 1))
  else
    echo "  ok   $src  died at run under $tok"
    passed=$((passed + 1))
  fi
done

status=$([[ $((failed - broken)) -eq 0 ]] && echo pass || echo fail)
printf '{"status":"%s","mode":"full","metrics":{"total":%d,"declared":%d,"unverified":%d,"unjudgeable":%d,"passed":%d,"failed":%d,"not_run":%d}}\n' \
  "$status" "$total_w" "$n_decl" "$n_undecl" "$broken" "$passed" "$((failed - broken))" "$n_undecl" > "$ART"

echo "witness_declares_its_sabotage: status=$status declared=$n_decl passed=$passed failed=$failed unverified=$n_undecl of=$total_w"
if [[ $unknown_token -ne 0 ]]; then
  echo "  $unknown_token witness(es) name a sabotage token with no switch behind it"
fi
# Survival and brokenness are different accusations and must not be reported
# as one. A witness that survives its sabotage is claiming more than it
# measures; a witness that is already red says nothing either way, and calling
# that "survived" points at the wrong defect.
survived=$((failed - broken))

# What this gate blocks on, and what it only reports.
#
# It owns one question: does a witness that declares a sabotage actually die
# under it? A witness that survives is claiming more than it measures, and that
# blocks.
#
# A witness that cannot run at all answers a different question, and it is not
# this gate's to enforce. Blocking on it would make this gate red on main from
# the day it lands, and a gate that lives red gets ignored -- which is exactly
# how madaros_corpus_regression_gate.sh spent twelve days telling every branch
# it had broken 21 tests that were already broken. Introducing a second one of
# those would cost more than it found.
#
# So they are counted, named, and written to the artifact, loudly enough that
# nobody has to grep for them. #2148 is the first.
if [[ $broken -gt 0 ]]; then
  echo "  NOT JUDGED: $broken witness(es) fail with no sabotage applied and cannot be assessed here."
  echo "              They are counted in the artifact as \"unjudgeable\"; each needs its own issue."
fi
if [[ $survived -gt 0 ]]; then
  gate_fail "$survived witness(es) survived their own declared sabotage"
fi
gate_pass "$passed/$total declared witnesses died as declared; $n_undecl of $total_w still unverified"
exit 0
