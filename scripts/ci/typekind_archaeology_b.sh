#!/usr/bin/env bash
# Derive TypeKind ladder positions for family B from fixtures (protocol v3).
# The index does NOT store a position. This gate prints one and fails on
# regression of a named refuse or of the Contest Claim-ready pass check.
#
# Derivation:
#   no pass path and no refuse path                         -> Garden
#   pass check OK  + refuse FAIL with expected_diagnostic  -> Claim-ready
#   pass check FAIL + refuse FAIL with expected_diagnostic  -> Reserved
#   pass check OK  + refuse check OK                         -> Executable
#   anything else with a fixture pair                       -> Hypothesis
#
# Fail the gate when:
#   - index / souc / listed fixture files are missing
#   - a refuse starts passing (xpass) or misses its named diagnostic
#   - Contest (the measured Claim-ready row) stops checking clean
#
# Reserved rows intentionally fail their pass fixtures — that is not a
# regression. Claim-ready is a *check* fact; pass_run is recorded but does
# not fail the gate (Contest still dies at the native-v2 bridge).
#
# Usage:
#   bash scripts/ci/typekind_archaeology_b.sh
#   TYPEKIND_B_REPORT_ONLY=1 bash scripts/ci/typekind_archaeology_b.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=../lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUC_BIN SOUNIO_SOUC_ENGINE || true
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
ulimit -s 1048576 2>/dev/null || true

INDEX="${TYPEKIND_B_INDEX:-$ROOT_DIR/docs/audit/TYPE_ARCHAEOLOGY_FAMILY_B_2026-08-19.tsv}"
SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
REPORT_ONLY="${TYPEKIND_B_REPORT_ONLY:-0}"

fail() {
  echo "TYPEKIND_B_FAIL reason=$1" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "missing_souc path=$SOUC"
[[ -f "$INDEX" ]] || fail "missing_index path=$INDEX"

sha_main="$(git -C "$ROOT_DIR" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-typekind-b.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

check_one() {
  local src="$1" log="$2"
  set +e
  "$SOUC" check "$src" >"$log" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc"
}

rows=0
garden=0
claim_ready=0
reserved=0
executable=0
hypothesis=0
refuse_xpass=0
refuse_nodiag=0
contest_pass_fail=0

echo "# derived by scripts/ci/typekind_archaeology_b.sh sha=$sha_main souc=$SOUC"
echo -e "kind\tderived\tpass_check\trefuse_check\trefuse_named\tpass_run\tdeepest_layer\tpass_path\trefuse_path"

while IFS=$'\t' read -r kind pass_path refuse_path expected_diag deepest rest || [[ -n "${kind:-}" ]]; do
  [[ -z "${kind:-}" || "$kind" == \#* || "$kind" == kind ]] && continue
  rows=$((rows + 1))

  local_pass_rc="-"
  local_refuse_rc="-"
  refuse_named="-"
  pass_run="-"
  derived=""

  has_pass=0
  has_refuse=0
  [[ -n "${pass_path:-}" && "$pass_path" != "-" ]] && has_pass=1
  [[ -n "${refuse_path:-}" && "$refuse_path" != "-" ]] && has_refuse=1

  if (( has_pass == 0 && has_refuse == 0 )); then
    derived="Garden"
    garden=$((garden + 1))
    echo -e "${kind}\t${derived}\t-\t-\t-\t-\t${deepest:--}\t-\t-"
    continue
  fi

  if (( has_pass == 1 )); then
    if [[ ! -f "$ROOT_DIR/$pass_path" ]]; then
      fail "missing_pass kind=$kind path=$pass_path"
    fi
    local_pass_rc="$(check_one "$ROOT_DIR/$pass_path" "$workdir/${kind}.pass.log")"
    if [[ "$local_pass_rc" == "0" ]]; then
      set +e
      "$SOUC" run "$ROOT_DIR/$pass_path" >"$workdir/${kind}.pass.run.log" 2>&1
      pass_run=$?
      set -e
    fi
  fi

  if (( has_refuse == 1 )); then
    if [[ ! -f "$ROOT_DIR/$refuse_path" ]]; then
      fail "missing_refuse kind=$kind path=$refuse_path"
    fi
    local_refuse_rc="$(check_one "$ROOT_DIR/$refuse_path" "$workdir/${kind}.refuse.log")"
    if [[ "$local_refuse_rc" != "0" && -n "${expected_diag:-}" && "$expected_diag" != "-" ]]; then
      if grep -F -q -- "$expected_diag" "$workdir/${kind}.refuse.log"; then
        refuse_named="yes"
      else
        refuse_named="no"
        refuse_nodiag=$((refuse_nodiag + 1))
        echo "TYPEKIND_B_REFUSE_NODIAG kind=$kind expect=$expected_diag path=$refuse_path" >&2
        echo "$(tail -n 12 "$workdir/${kind}.refuse.log")" >&2
      fi
    elif [[ "$local_refuse_rc" == "0" ]]; then
      refuse_named="xpass"
      refuse_xpass=$((refuse_xpass + 1))
      echo "TYPEKIND_B_REFUSE_XPASS kind=$kind path=$refuse_path" >&2
    else
      refuse_named="no"
      refuse_nodiag=$((refuse_nodiag + 1))
      echo "TYPEKIND_B_REFUSE_NODIAG kind=$kind expect=${expected_diag:--} path=$refuse_path" >&2
    fi
  fi

  if (( has_pass == 1 && has_refuse == 1 )); then
    if [[ "$local_pass_rc" == "0" && "$local_refuse_rc" != "0" && "$refuse_named" == "yes" ]]; then
      derived="Claim-ready"
      claim_ready=$((claim_ready + 1))
    elif [[ "$local_pass_rc" != "0" && "$local_refuse_rc" != "0" && "$refuse_named" == "yes" ]]; then
      derived="Reserved"
      reserved=$((reserved + 1))
    elif [[ "$local_pass_rc" == "0" && "$local_refuse_rc" == "0" ]]; then
      derived="Executable"
      executable=$((executable + 1))
    else
      derived="Hypothesis"
      hypothesis=$((hypothesis + 1))
    fi
  else
    derived="Hypothesis"
    hypothesis=$((hypothesis + 1))
  fi

  # Contest is the measured Claim-ready row: its pass fixture must keep checking.
  if [[ "$kind" == "Contest" && "$local_pass_rc" != "0" ]]; then
    contest_pass_fail=$((contest_pass_fail + 1))
    echo "TYPEKIND_B_CONTEST_PASS_FAIL path=$pass_path rc=$local_pass_rc" >&2
    echo "$(tail -n 12 "$workdir/${kind}.pass.log")" >&2
  fi

  echo -e "${kind}\t${derived}\t${local_pass_rc}\t${local_refuse_rc}\t${refuse_named}\t${pass_run}\t${deepest:--}\t${pass_path}\t${refuse_path}"
done < "$INDEX"

echo "TYPEKIND_B_DERIVED rows=$rows garden=$garden claim_ready=$claim_ready reserved=$reserved executable=$executable hypothesis=$hypothesis refuse_xpass=$refuse_xpass refuse_nodiag=$refuse_nodiag contest_pass_fail=$contest_pass_fail"

if (( refuse_xpass > 0 || refuse_nodiag > 0 || contest_pass_fail > 0 )); then
  if [[ "$REPORT_ONLY" == "1" ]]; then
    echo "TYPEKIND_B_REPORT_ONLY would_fail refuse_xpass=$refuse_xpass refuse_nodiag=$refuse_nodiag contest_pass_fail=$contest_pass_fail" >&2
    exit 0
  fi
  fail "refuse_xpass=$refuse_xpass refuse_nodiag=$refuse_nodiag contest_pass_fail=$contest_pass_fail"
fi

echo "TYPEKIND_B_PASS rows=$rows claim_ready=$claim_ready reserved=$reserved garden=$garden"
exit 0
