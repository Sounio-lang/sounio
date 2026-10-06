#!/usr/bin/env bash
# Derive family-B ladder positions from fixtures. The index does not store a position.
# Protocol v3: right passes + wrong fails named = Claim-ready; both fail = Reserved;
# missing pair = Garden (no fixtures) or Hypothesis (one fixture only).
#
# Clock: this worktree bin/souc (Madaros). Never inherit SOUC_BIN / SOUNIO_SOUC_ENGINE.
# Usage: bash scripts/dev/typekind_family_b_derive.sh
# Output: TSV on stdout (derived). Does not write the index.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
INDEX="$ROOT/docs/audit/TYPE_ARCHAEOLOGY_FAMILY_B_2026-08-19.tsv"
SOUC="$ROOT/bin/souc"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_GATE_SOUC="$SOUC"
export MADAROS_STACK_KB=524288
unset SOUC_BIN SOUNIO_SOUC_ENGINE
ulimit -s 1048576 2>/dev/null || true

if [[ ! -x "$SOUC" ]]; then
  echo "derive_fail reason=missing_souc path=$SOUC" >&2
  exit 2
fi
if [[ ! -f "$INDEX" ]]; then
  echo "derive_fail reason=missing_index path=$INDEX" >&2
  exit 2
fi

sha_main="$(git -C "$ROOT" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/typekind-b-derive.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

check_one() {
  local src="$1" log="$2"
  "$SOUC" check "$src" >"$log" 2>&1
  echo $?
}

echo "# derived by scripts/dev/typekind_family_b_derive.sh sha=$sha_main souc=$SOUC"
echo -e "kind\tderived\tpass_check\trefuse_check\trefuse_named\tpass_run\tdeepest_layer\tpass_path\trefuse_path"

while IFS=$'\t' read -r kind pass_path refuse_path expected_diag deepest rest; do
  [[ -z "${kind:-}" || "$kind" == \#* || "$kind" == kind ]] && continue

  local_pass_rc="-"
  local_refuse_rc="-"
  refuse_named="-"
  pass_run="-"
  derived=""

  has_pass=0
  has_refuse=0
  [[ -n "$pass_path" && "$pass_path" != "-" ]] && has_pass=1
  [[ -n "$refuse_path" && "$refuse_path" != "-" ]] && has_refuse=1

  if (( has_pass == 0 && has_refuse == 0 )); then
    derived="Garden"
    echo -e "${kind}\t${derived}\t-\t-\t-\t-\t${deepest}\t-\t-"
    continue
  fi

  if (( has_pass == 1 )); then
    if [[ ! -f "$ROOT/$pass_path" ]]; then
      echo "derive_fail reason=missing_pass kind=$kind path=$pass_path" >&2
      exit 2
    fi
    local_pass_rc="$(check_one "$ROOT/$pass_path" "$workdir/${kind}.pass.log")"
    if [[ "$local_pass_rc" == "0" ]]; then
      "$SOUC" run "$ROOT/$pass_path" >"$workdir/${kind}.pass.run.log" 2>&1
      pass_run=$?
    fi
  fi

  if (( has_refuse == 1 )); then
    if [[ ! -f "$ROOT/$refuse_path" ]]; then
      echo "derive_fail reason=missing_refuse kind=$kind path=$refuse_path" >&2
      exit 2
    fi
    local_refuse_rc="$(check_one "$ROOT/$refuse_path" "$workdir/${kind}.refuse.log")"
    if [[ "$local_refuse_rc" != "0" && -n "$expected_diag" && "$expected_diag" != "-" ]]; then
      if grep -F -q -- "$expected_diag" "$workdir/${kind}.refuse.log"; then
        refuse_named="yes"
      else
        refuse_named="no"
      fi
    elif [[ "$local_refuse_rc" == "0" ]]; then
      refuse_named="xpass"
    else
      refuse_named="no"
    fi
  fi

  if (( has_pass == 1 && has_refuse == 1 )); then
    if [[ "$local_pass_rc" == "0" && "$local_refuse_rc" != "0" && "$refuse_named" == "yes" ]]; then
      derived="Claim-ready"
    elif [[ "$local_pass_rc" != "0" && "$local_refuse_rc" != "0" && "$refuse_named" == "yes" ]]; then
      derived="Reserved"
    elif [[ "$local_pass_rc" == "0" && "$local_refuse_rc" == "0" ]]; then
      derived="Executable"
    elif [[ "$local_pass_rc" != "0" && "$local_refuse_rc" == "0" ]]; then
      derived="Hypothesis"
    else
      derived="Hypothesis"
    fi
  else
    derived="Hypothesis"
  fi

  echo -e "${kind}\t${derived}\t${local_pass_rc}\t${local_refuse_rc}\t${refuse_named}\t${pass_run}\t${deepest}\t${pass_path}\t${refuse_path}"
done < "$INDEX"
