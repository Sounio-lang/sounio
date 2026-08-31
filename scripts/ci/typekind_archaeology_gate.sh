#!/usr/bin/env bash
# Derive TypeKind ladder position from pass/refuse fixtures (PROTOCOLO v3).
# The index does NOT store position — this script is the ruler.
#
# Usage:
#   bash scripts/ci/typekind_archaeology_gate.sh [index.tsv]
#
# Derived positions:
#   empty pass+refuse paths              -> Garden
#   pass OK  + refuse FAIL+diag          -> Claim-ready
#   pass FAIL+diag + refuse FAIL+diag    -> Reserved
#   pass OK  + refuse PASS               -> Executable (XPASS fail)
#   pass FAIL (not Reserved)              -> PASS_REGRESSION fail
#   refuse FAIL without expected diag    -> WRONG_DIAGNOSTIC fail
#
# Env:
#   SOUNIO_TYPEKIND_ARCHAEOLOGY_BIN          compiler (default: bin/souc)
#   SOUNIO_TYPEKIND_ARCHAEOLOGY_REPORT_ONLY  1 = print table, exit 0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INDEX="${1:-$ROOT_DIR/tests/typekind/index.tsv}"
COMPILER="${SOUNIO_TYPEKIND_ARCHAEOLOGY_BIN:-$ROOT_DIR/bin/souc}"
REPORT_ONLY="${SOUNIO_TYPEKIND_ARCHAEOLOGY_REPORT_ONLY:-0}"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
ulimit -s 1048576 2>/dev/null || true

fail() {
  echo "TYPEKIND_ARCHAEOLOGY_GATE_FAIL reason=$1" >&2
  exit 1
}

[[ -f "$INDEX" ]] || fail "missing_index path=$INDEX"
[[ -x "$COMPILER" ]] || fail "missing_compiler path=$COMPILER"

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-typekind-archaeology.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

failures=0
rows=0
xpass=0
regress=0
wrong_diag=0

printf 'kind\tposition\tdeepest_named_layer\tpass_rc\trefuse_rc\texpected_diagnostic\tpass_fixture\trefuse_fixture\n'

while IFS=$'\t' read -r kind pass_fixture refuse_fixture expected_diag deepest_layer || [[ -n "${kind:-}" ]]; do
  # skip header / blank / comments
  [[ -z "${kind:-}" ]] && continue
  [[ "$kind" == "kind" ]] && continue
  [[ "$kind" == \#* ]] && continue

  rows=$((rows + 1))
  pass_fixture="${pass_fixture:-}"
  refuse_fixture="${refuse_fixture:-}"
  expected_diag="${expected_diag:-}"
  deepest_layer="${deepest_layer:-}"

  # Garden: both paths empty (declared absence)
  if [[ -z "$pass_fixture" && -z "$refuse_fixture" ]]; then
    printf '%s\tGarden\t%s\tNA\tNA\t%s\t\t\n' "$kind" "$deepest_layer" "$expected_diag"
    continue
  fi

  # Incomplete pair without files → still Garden if neither path resolves;
  # if index lists a path that is missing, that is a packaging error.
  pass_path=""
  refuse_path=""
  [[ -n "$pass_fixture" ]] && pass_path="$ROOT_DIR/$pass_fixture"
  [[ -n "$refuse_fixture" ]] && refuse_path="$ROOT_DIR/$refuse_fixture"

  if [[ -z "$pass_fixture" || -z "$refuse_fixture" ]]; then
    # one-sided index row: treat as Garden (no complete executable pair)
    printf '%s\tGarden\t%s\tNA\tNA\t%s\t%s\t%s\n' \
      "$kind" "$deepest_layer" "$expected_diag" "$pass_fixture" "$refuse_fixture"
    continue
  fi

  if [[ ! -f "$pass_path" || ! -f "$refuse_path" ]]; then
    echo "TYPEKIND_ARCHAEOLOGY_MISSING_FIXTURE kind=$kind pass=$pass_fixture refuse=$refuse_fixture" >&2
    failures=$((failures + 1))
    printf '%s\tGarden\t%s\tNA\tNA\t%s\t%s\t%s\n' \
      "$kind" "$deepest_layer" "$expected_diag" "$pass_fixture" "$refuse_fixture"
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
  if [[ -n "$expected_diag" ]]; then
    grep -Fq "error[$expected_diag]" "$work_dir/${kind}.pass.log" && pass_has_diag=1
    grep -Fq "error[$expected_diag]" "$work_dir/${kind}.refuse.log" && refuse_has_diag=1
  fi

  position=Hypothesis
  note=""

  if ((pass_rc == 0)); then
    if ((refuse_rc != 0 && refuse_has_diag == 1)); then
      position=Claim-ready
    elif ((refuse_rc == 0)); then
      position=Executable
      note=XPASS
      xpass=$((xpass + 1))
      failures=$((failures + 1))
      echo "TYPEKIND_ARCHAEOLOGY_XPASS kind=$kind fixture=$refuse_fixture" >&2
    else
      position=Executable
      note=WRONG_DIAGNOSTIC
      wrong_diag=$((wrong_diag + 1))
      failures=$((failures + 1))
      echo "TYPEKIND_ARCHAEOLOGY_WRONG_DIAGNOSTIC kind=$kind expected=$expected_diag fixture=$refuse_fixture" >&2
    fi
  elif ((refuse_rc != 0 && refuse_has_diag == 1 && pass_has_diag == 1)); then
    # Both programs refused with the reserved diagnostic — Reserved (v2 off-ladder)
    position=Reserved
  else
    position=Hypothesis
    note=PASS_REGRESSION
    regress=$((regress + 1))
    failures=$((failures + 1))
    echo "TYPEKIND_ARCHAEOLOGY_PASS_REGRESSION kind=$kind fixture=$pass_fixture rc=$pass_rc" >&2
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$position" "$deepest_layer" "$pass_rc" "$refuse_rc" \
    "$expected_diag" "$pass_fixture" "$refuse_fixture"
done <"$INDEX"

echo "TYPEKIND_ARCHAEOLOGY_SUMMARY rows=$rows failures=$failures xpass=$xpass pass_regression=$regress wrong_diagnostic=$wrong_diag"
if ((failures > 0)) && [[ "$REPORT_ONLY" != "1" ]]; then
  exit 1
fi
exit 0
