#!/usr/bin/env bash
# Derive the effect archaeology ladder from executable pass/refuse pairs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INDEX="${SOUNIO_EFFECT_ARCHAEOLOGY_INDEX:-$ROOT_DIR/tests/effects/archaeology/index.tsv}"
COMPILER="${SOUNIO_EFFECT_ARCHAEOLOGY_BIN:-$ROOT_DIR/bin/souc}"
REPORT_ONLY="${SOUNIO_EFFECT_ARCHAEOLOGY_REPORT_ONLY:-0}"

fail() {
  echo "EFFECT_ARCHAEOLOGY_GATE_FAIL reason=$1" >&2
  exit 1
}

[[ -f "$INDEX" ]] || fail "missing_index"
[[ -x "$COMPILER" ]] || fail "missing_compiler"

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-effect-archaeology.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

failures=0
rows=0
printf 'kind\tposition\tdeepest_named_layer\tpass_rc\trefuse_rc\texpected_diagnostic\n'

while IFS=$'\t' read -r kind pass_fixture refuse_fixture expected_diag deepest_layer; do
  [[ "$kind" == "kind" ]] && continue
  [[ -z "$kind" ]] && continue
  rows=$((rows + 1))

  if [[ -z "$pass_fixture" || -z "$refuse_fixture" ]]; then
    printf '%s\tGarden\t%s\tNA\tNA\t%s\n' "$kind" "$deepest_layer" "$expected_diag"
    continue
  fi

  pass_path="$ROOT_DIR/$pass_fixture"
  refuse_path="$ROOT_DIR/$refuse_fixture"
  if [[ ! -f "$pass_path" || ! -f "$refuse_path" ]]; then
    printf '%s\tGarden\t%s\tNA\tNA\t%s\n' "$kind" "$deepest_layer" "$expected_diag"
    failures=$((failures + 1))
    continue
  fi

  set +e
  "$COMPILER" run "$pass_path" >"$work_dir/${kind}.pass.log" 2>&1
  pass_rc=$?
  "$COMPILER" check "$refuse_path" >"$work_dir/${kind}.refuse.log" 2>&1
  refuse_rc=$?
  set -e

  pass_has_diag=0
  refuse_has_diag=0
  grep -Fq "error[$expected_diag]" "$work_dir/${kind}.pass.log" && pass_has_diag=1
  grep -Fq "error[$expected_diag]" "$work_dir/${kind}.refuse.log" && refuse_has_diag=1

  position=Hypothesis
  if ((pass_rc == 0)); then
    if ((refuse_rc != 0 && refuse_has_diag == 1)); then
      position=Claim-ready
    elif ((refuse_rc == 0)); then
      position=Executable
      failures=$((failures + 1))
      echo "EFFECT_ARCHAEOLOGY_XPASS kind=$kind fixture=$refuse_fixture" >&2
    else
      position=Executable
      failures=$((failures + 1))
      echo "EFFECT_ARCHAEOLOGY_WRONG_DIAGNOSTIC kind=$kind expected=$expected_diag" >&2
    fi
  elif ((refuse_rc != 0 && refuse_has_diag == 1 && pass_has_diag == 1)); then
    position=Reserva
  else
    failures=$((failures + 1))
    echo "EFFECT_ARCHAEOLOGY_PASS_REGRESSION kind=$kind fixture=$pass_fixture rc=$pass_rc" >&2
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$position" "$deepest_layer" "$pass_rc" "$refuse_rc" "$expected_diag"
done <"$INDEX"

echo "EFFECT_ARCHAEOLOGY_SUMMARY rows=$rows failures=$failures"
if ((failures > 0)) && [[ "$REPORT_ONLY" != "1" ]]; then
  exit 1
fi
