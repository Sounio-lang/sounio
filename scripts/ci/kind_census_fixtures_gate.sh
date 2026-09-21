#!/usr/bin/env bash
# Derive kind-census positions from pass/refuse fixtures.
# The index does not store a position. This script is the table.
#
# Precedence: scripts/ci/known_failure_madaros_recheck.sh (XPASS = signal)
#             and scripts/ci/effect_archaeology_gate.sh (codex-1, 16 founder effects).
# This gate owns only tests/kind-census/index.tsv — extra ladder names +
# CD/ExactRing surfaces measured by grok-cli3. It does not touch
# tests/effects/archaeology or tests/typekind.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INDEX="${SOUNIO_KIND_CENSUS_INDEX:-$ROOT_DIR/tests/kind-census/index.tsv}"
COMPILER="${SOUNIO_KIND_CENSUS_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
REPORT_ONLY="${SOUNIO_KIND_CENSUS_REPORT_ONLY:-0}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

fail() {
  echo "KIND_CENSUS_GATE_FAIL reason=$1" >&2
  exit 1
}

[[ -f "$INDEX" ]] || fail "missing_index"
[[ -x "$COMPILER" ]] || fail "missing_compiler"

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-kind-census.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

failures=0
rows=0
printf 'kind\tposition\tdeepest_named_layer\tpass_rc\trefuse_rc\texpected_diagnostic\n'

while IFS=$'\t' read -r kind pass_fixture refuse_fixture expected_diag deepest_layer; do
  [[ "$kind" == "kind" ]] && continue
  [[ -z "${kind:-}" ]] && continue
  rows=$((rows + 1))

  pass_fixture="${pass_fixture:-}"
  refuse_fixture="${refuse_fixture:-}"
  expected_diag="${expected_diag:-}"
  deepest_layer="${deepest_layer:-}"
  [[ "$pass_fixture" == "-" ]] && pass_fixture=""
  [[ "$refuse_fixture" == "-" ]] && refuse_fixture=""
  [[ "$expected_diag" == "-" ]] && expected_diag=""

  if [[ -z "$pass_fixture" && -z "$refuse_fixture" ]]; then
    printf '%s\tGarden\t%s\tNA\tNA\t%s\n' "$kind" "$deepest_layer" "$expected_diag"
    continue
  fi

  pass_rc=NA
  refuse_rc=NA
  pass_has_diag=0
  refuse_has_diag=0

  if [[ -n "$pass_fixture" ]]; then
    pass_path="$ROOT_DIR/$pass_fixture"
    if [[ ! -f "$pass_path" ]]; then
      echo "KIND_CENSUS_MISSING kind=$kind fixture=$pass_fixture" >&2
      failures=$((failures + 1))
      printf '%s\tGarden\t%s\tNA\tNA\t%s\n' "$kind" "$deepest_layer" "$expected_diag"
      continue
    fi
    set +e
    "$COMPILER" run "$pass_path" >"$work_dir/${kind}.pass.log" 2>&1
    pass_rc=$?
    set -e
    if [[ -n "$expected_diag" ]]; then
      grep -Fq "error[$expected_diag]" "$work_dir/${kind}.pass.log" && pass_has_diag=1
    fi
  fi

  if [[ -n "$refuse_fixture" ]]; then
    refuse_path="$ROOT_DIR/$refuse_fixture"
    if [[ ! -f "$refuse_path" ]]; then
      echo "KIND_CENSUS_MISSING kind=$kind fixture=$refuse_fixture" >&2
      failures=$((failures + 1))
      printf '%s\tGarden\t%s\t%s\tNA\t%s\n' "$kind" "$deepest_layer" "$pass_rc" "$expected_diag"
      continue
    fi
    set +e
    "$COMPILER" check "$refuse_path" >"$work_dir/${kind}.refuse.log" 2>&1
    refuse_rc=$?
    set -e
    if [[ -n "$expected_diag" ]]; then
      grep -Fq "error[$expected_diag]" "$work_dir/${kind}.refuse.log" && refuse_has_diag=1
    fi
  fi

  position=Hypothesis
  if [[ -n "$pass_fixture" && -n "$refuse_fixture" ]]; then
    if [[ "$pass_rc" == 0 && "$refuse_rc" != 0 && "$refuse_has_diag" == 1 ]]; then
      position=Claim-ready
    elif [[ "$pass_rc" == 0 && "$refuse_rc" == 0 ]]; then
      position=Executable
      failures=$((failures + 1))
      echo "KIND_CENSUS_XPASS kind=$kind fixture=$refuse_fixture" >&2
    elif [[ "$pass_rc" != 0 && "$refuse_rc" != 0 && "$pass_has_diag" == 1 && "$refuse_has_diag" == 1 ]]; then
      position=Reserva
    elif [[ "$pass_rc" != 0 && "$refuse_rc" != 0 && "$refuse_has_diag" == 1 ]]; then
      # correct construct refused, wrong use also refused — Reserva even if
      # the pass log uses a different E-code than the index expected.
      position=Reserva
    elif [[ "$pass_rc" == 0 ]]; then
      position=Executable
      failures=$((failures + 1))
      echo "KIND_CENSUS_WRONG_DIAGNOSTIC kind=$kind expected=$expected_diag" >&2
    else
      failures=$((failures + 1))
      echo "KIND_CENSUS_PASS_REGRESSION kind=$kind fixture=$pass_fixture rc=$pass_rc" >&2
    fi
  elif [[ -n "$pass_fixture" ]]; then
    if [[ "$pass_rc" == 0 ]]; then
      position=Hypothesis
    else
      failures=$((failures + 1))
      echo "KIND_CENSUS_PASS_REGRESSION kind=$kind fixture=$pass_fixture rc=$pass_rc" >&2
    fi
  else
    if [[ "$refuse_rc" != 0 && "$refuse_has_diag" == 1 ]]; then
      position=Reserva
    else
      position=Hypothesis
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$position" "$deepest_layer" "$pass_rc" "$refuse_rc" "$expected_diag"
done <"$INDEX"

echo "KIND_CENSUS_SUMMARY rows=$rows failures=$failures"
if ((failures > 0)) && [[ "$REPORT_ONLY" != "1" ]]; then
  fail "derived_failures=$failures"
fi
echo "KIND_CENSUS_GATE_PASS rows=$rows"
