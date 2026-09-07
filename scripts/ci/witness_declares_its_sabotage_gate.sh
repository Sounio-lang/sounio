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
# Merge Conflicts and Derived Artefacts (#2391)
# ---------------------------------------------
# artifacts/gates/witness_declares_its_sabotage.json is a derived artefact.
# When concurrent branches add witnesses, the cardinality and corpus digest
# will conflict on merge. NEVER resolve the conflict by picking HEAD or main:
# always re-derive via:
#   SOUNIO_WITNESS_SABOTAGE_CENSUS_ONLY=1 bash scripts/ci/witness_declares_its_sabotage_gate.sh
#
# Usage:
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

# Tracked witnesses only — Finder "foo 2.sio" duplicates must not inflate the census.
mapfile -t ALL_WITNESSES < <(git -C "$ROOT_DIR" ls-files 'tests/run-pass/*.sio' | LC_ALL=C sort)

# Map declared witnesses quickly without spawning thousands of separate subshells.
declare -A DECL_MAP=()
while IFS=: read -r file match; do
  [[ -z "$file" ]] && continue
  tok="$(awk '{print $NF}' <<<"$match")"
  DECL_MAP["$file"]="$tok"
done < <(grep -r -m1 -E '^//@ sabotage:[[:space:]]*[a-z0-9-]+' tests/run-pass 2>/dev/null || true)

declared=(); undeclared=()
for w in "${ALL_WITNESSES[@]}"; do
  if [[ -n "${DECL_MAP["$w"]:-}" ]]; then
    declared+=("$w|${DECL_MAP["$w"]}")
  else
    undeclared+=("$w")
  fi
done

total_w=${#ALL_WITNESSES[@]}
n_decl=${#declared[@]}
n_undecl=${#undeclared[@]}

# Corpus identity digest: SHA-256 over sorted "path + content-hash" pairs.
#
# #2391 landed this as a digest of the sorted PATHS alone, which fixes
# cardinality and naming but not identity: rewriting a witness's body while
# keeping its filename leaves the digest unchanged. That is precisely the
# "corpus swap with cardinality preserved" case this census is meant to make
# impossible, so the content hash is folded in here. The digest necessarily
# changes value when this line lands — it is a different, stronger claim, and
# the pinned value in scripts/ci/fixtures/measured_claims.tsv is re-derived in
# the same commit.
# One hasher invocation over the whole (already sorted) list: its output is
# "<content-hash>  <path>" per line, so the inner digest covers content AND
# naming AND ordering in a single pass. Hashing file-by-file in a shell loop
# would spawn ~1900 processes here and cost seconds per gate run.
corpus_sha="$(
  { sha256sum "${ALL_WITNESSES[@]}" 2>/dev/null || shasum -a 256 "${ALL_WITNESSES[@]}"; } \
    | (sha256sum 2>/dev/null || shasum -a 256) | awk '{print $1}'
)"

# Per-class counts of run-pass directives (#2391).
n_check_only="$(grep -r -m1 -l '^//@ check-only' tests/run-pass 2>/dev/null | wc -l | tr -d ' ')"
n_known_fail="$(grep -r -m1 -l '^//@ known-failure' tests/run-pass 2>/dev/null | wc -l | tr -d ' ')"
n_req_madaros="$(grep -r -m1 -l '^//@ requires:[[:space:]]*madaros' tests/run-pass 2>/dev/null | wc -l | tr -d ' ')"

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
  printf '{"_comment":"Derived artifact: on merge conflict re-derive with SOUNIO_WITNESS_SABOTAGE_CENSUS_ONLY=1 bash scripts/ci/witness_declares_its_sabotage_gate.sh, never resolve by side","status":"pass","mode":"census","corpus_sha256":"%s","metrics":{"total":%d,"declared":%d,"unverified":%d,"passed":0,"failed":0,"not_run":%d},"classes":{"check_only":%d,"known_failure":%d,"requires_madaros":%d}}\n' \
    "$corpus_sha" "$total_w" "$n_decl" "$n_undecl" "$total_w" "$n_check_only" "$n_known_fail" "$n_req_madaros" | gate_write_artifact "$ART"
  echo "witness_declares_its_sabotage: census only -- $n_decl declared, $n_undecl unverified, of $total_w (corpus_sha256=$corpus_sha)"
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
# Death has kinds, and "died" alone is not evidence.
#
# A witness that stops compiling under sabotage did not necessarily detect the
# removed property -- the switch may simply have broken the compiler. A witness
# killed by a signal or a timeout says even less. Only a clean run-failure, or a
# refusal that names a diagnostic, shows the witness noticing what was taken
# away. All four were counted as one "ok" before, which let the strongest and
# the weakest evidence share a line of output.
#
# Measured on the run that landed the gate: all five deaths were clean run
# failures, so this classification changes no verdict today. It exists so the
# first unclean death is visible on the day it appears rather than on the day
# someone asks.
d_run=0; d_compile=0; d_crash=0; d_timeout=0; d_misattributed=0
for row in "${declared[@]}"; do
  src="${row%%|*}"; tok="${row##*|}"

  # What kind of death this witness claims, and optionally which diagnostic.
  #
  # Without this, ANY error[Ennn] under sabotage counted as the witness noticing.
  # A quotient-Hessian switch that happens to provoke an unrelated E035 would be
  # certified as a clean kill -- the failure would be real and the attribution
  # invented. Declaring the expected class makes the attribution load-bearing:
  #
  #   //@ sabotage: quotient-hessian
  #   //@ sabotage-expect: run-fail            (default when absent)
  #   //@ sabotage-expect: compile-refused
  #   //@ sabotage-error-pattern: E242         (implies compile-refused)
  #
  # Absent, it defaults to run-fail, which is what all five witnesses on main do
  # today -- so this tightens without moving a single current verdict.
  want_class="$(grep -m1 -oE '^//@ sabotage-expect:[[:space:]]*[a-z-]+' "$src" 2>/dev/null | awk '{print $NF}')"
  want_diag="$(grep -m1 -oE '^//@ sabotage-error-pattern:[[:space:]]*[A-Za-z0-9_\[\]-]+' "$src" 2>/dev/null | awk '{print $NF}')"
  [[ -n "$want_diag" && -z "$want_class" ]] && want_class="compile-refused"
  [[ -z "$want_class" ]] && want_class="run-fail"
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
  env "$env_var=1" timeout 300 "$MADAROS" build "$src" "$bad" >"$WORK/b2.log" 2>&1
  build_rc=$?
  if [[ $build_rc -ne 0 ]]; then
    if [[ $build_rc -eq 124 ]]; then
      echo "  UNCLEAN  $src  compile TIMED OUT under $tok -- not evidence the witness noticed"
      d_timeout=$((d_timeout + 1))
    elif [[ $build_rc -gt 128 ]]; then
      echo "  UNCLEAN  $src  compiler died by signal $((build_rc - 128)) under $tok"
      d_crash=$((d_crash + 1))
    elif grep -qE 'error\[E[0-9]+\]' "$WORK/b2.log"; then
      got_diag="$(grep -oE 'error\[E[0-9]+\]' "$WORK/b2.log" | head -1)"
      if [[ "$want_class" != "compile-refused" ]]; then
        echo "  MISATTRIBUTED  $src declares $want_class but died at COMPILE with $got_diag under $tok"
        d_misattributed=$((d_misattributed + 1))
      elif [[ -n "$want_diag" ]] && ! grep -qF "$want_diag" "$WORK/b2.log"; then
        echo "  MISATTRIBUTED  $src declares $want_diag but got $got_diag under $tok"
        d_misattributed=$((d_misattributed + 1))
      else
        echo "  ok       $src  refused at compile under $tok ($got_diag)"
        d_compile=$((d_compile + 1)); passed=$((passed + 1))
      fi
    else
      echo "  UNCLEAN  $src  compile failed under $tok with no diagnostic (rc=$build_rc)"
      tail -3 "$WORK/b2.log" | sed 's/^/      /'
      d_crash=$((d_crash + 1))
    fi
    continue
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
  elif [[ $bad_rc -eq 124 ]]; then
    echo "  UNCLEAN  $src  TIMED OUT under $tok -- not evidence the witness noticed"
    d_timeout=$((d_timeout + 1))
  elif [[ $bad_rc -gt 128 ]]; then
    echo "  UNCLEAN  $src  died by signal $((bad_rc - 128)) under $tok, not by its own assertion"
    d_crash=$((d_crash + 1))
  elif [[ "$want_class" != "run-fail" ]]; then
    echo "  MISATTRIBUTED  $src declares $want_class but died at RUN under $tok"
    d_misattributed=$((d_misattributed + 1))
  else
    echo "  ok       $src  died at run under $tok"
    d_run=$((d_run + 1)); passed=$((passed + 1))
  fi
done

# Ratchet: unjudgeable is debt, and debt that can grow silently is not debt.
#
# Today one witness fails before any sabotage is applied
# (epistemic_hessian_transcendentals.sio, SIGILL, #2148). Counting it and moving
# on was right -- blocking would have made the gate red on main from the day it
# landed, and a gate that lives red gets ignored. But nothing stopped that count
# from becoming six, at which point the gate would still report OK while
# verifying almost nothing: every witness would be excused before it was tested.
#
# So the count is frozen at what it was when measured, and only ever lowered by
# editing this line -- which puts each removal in a diff, next to the issue that
# fixed it.
UNJUDGEABLE_CEILING="${SOUNIO_WITNESS_UNJUDGEABLE_CEILING:-1}"

# An unclean death is not a pass and not brokenness: it is a death whose cause
# we cannot attribute to the witness noticing the sabotage. Same argument, same
# treatment -- counted, ceilinged, and visible.
UNCLEAN_CEILING="${SOUNIO_WITNESS_UNCLEAN_CEILING:-0}"
unclean=$((d_crash + d_timeout + d_misattributed))

# The artifact must agree with the exit code. Survival, an unjudgeable count
# over its ceiling, and an unclean death all fail below, so all three have to
# show as fail here -- an artifact reporting "pass" beside a red gate is the
# same lie in a different file.
status=pass
if [[ $((failed - broken)) -ne 0 ]] || [[ $broken -gt ${SOUNIO_WITNESS_UNJUDGEABLE_CEILING:-1} ]] \
   || [[ $((d_crash + d_timeout + d_misattributed)) -gt ${SOUNIO_WITNESS_UNCLEAN_CEILING:-0} ]]; then
  status=fail
fi
printf '{"_comment":"Derived artifact: on merge conflict re-derive with SOUNIO_WITNESS_SABOTAGE_CENSUS_ONLY=1 bash scripts/ci/witness_declares_its_sabotage_gate.sh, never resolve by side","status":"%s","mode":"full","corpus_sha256":"%s","metrics":{"total":%d,"declared":%d,"unverified":%d,"unjudgeable":%d,"passed":%d,"failed":%d,"not_run":%d},"deaths":{"run":%d,"compile_refused":%d,"crash":%d,"timeout":%d,"misattributed":%d},"classes":{"check_only":%d,"known_failure":%d,"requires_madaros":%d}}\n' \
  "$status" "$corpus_sha" "$total_w" "$n_decl" "$n_undecl" "$broken" "$passed" "$((failed - broken))" "$n_undecl" \
  "$d_run" "$d_compile" "$d_crash" "$d_timeout" "$d_misattributed" \
  "$n_check_only" "$n_known_fail" "$n_req_madaros" | gate_write_artifact "$ART"

echo "witness_declares_its_sabotage: status=$status declared=$n_decl passed=$passed failed=$failed unverified=$n_undecl of=$total_w"
echo "  deaths: run=$d_run compile-refused=$d_compile crash=$d_crash timeout=$d_timeout misattributed=$d_misattributed | unjudgeable=$broken (ceiling $UNJUDGEABLE_CEILING)"
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

if [[ $broken -gt $UNJUDGEABLE_CEILING ]]; then
  gate_fail "unjudgeable witnesses rose $UNJUDGEABLE_CEILING -> $broken; each one is excused before it is tested"
fi
if [[ $unclean -gt $UNCLEAN_CEILING ]]; then
  gate_fail "$unclean death(s) were not the witness noticing the sabotage ($d_crash crash, $d_timeout timeout, $d_misattributed misattributed); ceiling is $UNCLEAN_CEILING"
fi
if [[ $survived -gt 0 ]]; then
  gate_fail "$survived witness(es) survived their own declared sabotage"
fi
gate_pass "$passed/$total declared witnesses died as declared; $n_undecl of $total_w still unverified"
exit 0
