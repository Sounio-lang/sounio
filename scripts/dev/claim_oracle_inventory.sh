#!/usr/bin/env bash
# scripts/dev/claim_oracle_inventory.sh
#
# Scan gate / oracle scripts and emit a provisional claim-oracle inventory
# (ADR-008). Classifications are heuristic — overrides may be supplied later.
#
# Output: artifacts/audit/claim_oracle_inventory.tsv
# Schema: docs/decisions/claim_oracle_inventory.schema.md
#
# Exit 0 always on successful scan (inventory is observational).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="$ROOT_DIR/artifacts/audit"
OUT_TSV="$OUT_DIR/claim_oracle_inventory.tsv"
mkdir -p "$OUT_DIR"

UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Collect candidate paths (gates + named oracles + parity refs)
mapfile -t CANDIDATES < <(
  {
    find scripts -type f \( -name '*gate*.sh' -o -name '*_gate.sh' \) 2>/dev/null
    find scripts -type f \( -name '*oracle*.py' -o -name '*_oracle.py' \) 2>/dev/null
    find scripts/parity -type f -name '*_ref.py' 2>/dev/null || true
    find scripts/ci -type f -name '*gate*.sh' 2>/dev/null
  } | sed 's|^\./||' | sort -u
)

classify_one() {
  local path="$1"
  local text
  if [[ ! -f "$path" ]]; then
    return 0
  fi
  # Cap read size for huge files
  text="$(head -c 200000 "$path" 2>/dev/null || true)"

  local kind="other"
  case "$path" in
    *oracle*.py|*_oracle.py) kind="oracle" ;;
    scripts/parity/*_ref.py) kind="parity_ref" ;;
    *gate*) kind="gate" ;;
  esac

  local tier="other"
  case "$path" in
    scripts/ci/*) tier="ci" ;;
    scripts/dev/*) tier="dev" ;;
    scripts/research/*) tier="research" ;;
    scripts/selfhost/*) tier="selfhost" ;;
    scripts/*) tier="root_scripts" ;;
  esac

  local has_py=0 has_souc=0 has_diff=0 has_fail=0 has_mpmath=0 has_skip=0
  local has_all_pass=0 has_trust=0 has_fixed=0 has_lean=0 has_optional=0

  echo "$text" | grep -qE 'python3|python |[.]py["'"'"' ]' && has_py=1 || true
  echo "$text" | grep -qE 'bin/souc|SOUC|souc run|souc check' && has_souc=1 || true
  echo "$text" | grep -qE '\bdiff\b|/usr/bin/diff' && has_diff=1 || true
  echo "$text" | grep -qE 'fail=1|exit 1|GATE FAILED|ORACLE MISMATCH' && has_fail=1 || true
  echo "$text" | grep -qiE 'mpmath|scipy|numpy' && has_mpmath=1 || true
  echo "$text" | grep -qE 'SKIP:|exit 0.*mpmath|not installed' && has_skip=1 || true
  echo "$text" | grep -qE 'ALL PASS|GUM_TRUST_OK|_OK"|expect-stdout' && has_all_pass=1 || true
  echo "$text" | grep -qiE 'epistemic_trust|GUM_TRUST' && has_trust=1 || true
  echo "$text" | grep -qiE 'gen2|gen3|fixed.point|fixed-point|boot4' && has_fixed=1 || true
  echo "$text" | grep -qiE '\bleach\b|lake build|formal/' && has_lean=1 || true
  echo "$text" | grep -qiE 'optional.*oracle|oracle.*optional|corroboration' && has_optional=1 || true

  local foreign="none"
  local fr=()
  [[ $has_py -eq 1 ]] && fr+=("python3")
  [[ $has_mpmath -eq 1 ]] && fr+=("mpmath_or_scipy")
  if [[ ${#fr[@]} -gt 0 ]]; then
    foreign=$(IFS=,; echo "${fr[*]}")
  fi

  local foreign_hard="no"
  local sounio_wit="no"
  local oclass="unknown"
  local migration="none"
  local notes=""

  if [[ $has_souc -eq 1 || $has_all_pass -eq 1 || $has_trust -eq 1 ]]; then
    sounio_wit="partial"
  fi
  if [[ $has_all_pass -eq 1 || $has_trust -eq 1 ]] && [[ $has_py -eq 0 ]]; then
    sounio_wit="yes"
  fi

  if [[ $has_fixed -eq 1 && $has_py -eq 0 ]]; then
    oclass="bootstrap_integrity"
    migration="keep"
    notes="fixed-point_or_bootstrap_signals"
  elif [[ $has_lean -eq 1 && $has_souc -eq 0 && $has_py -eq 0 ]]; then
    oclass="formal_only"
    migration="keep"
    notes="lean_or_formal_path"
  elif [[ $has_py -eq 1 && $has_fail -eq 1 && ( $has_diff -eq 1 || $has_mpmath -eq 1 ) ]]; then
    # Foreign mismatch can fail the gate
    if [[ $has_optional -eq 1 ]]; then
      oclass="external_corroboration_only"
      foreign_hard="unknown"
      migration="demote_corroboration"
      notes="python_present_with_fail_but_optional_markers"
    else
      oclass="forbidden_as_claim_oracle"
      foreign_hard="yes"
      migration="rehome_sounio"
      notes="python_diff_or_mpmath_on_fail_path"
    fi
    if [[ $has_souc -eq 1 ]]; then
      sounio_wit="partial"
    fi
  elif [[ $has_py -eq 1 && $has_fail -eq 0 ]]; then
    oclass="external_corroboration_only"
    foreign_hard="no"
    migration="demote_corroboration"
    notes="python_without_clear_fail_pairing"
  elif [[ $has_souc -eq 1 && $has_py -eq 0 ]]; then
    oclass="sounio_native_expected"
    foreign_hard="no"
    migration="keep"
    notes="souc_without_python_judge"
    sounio_wit="yes"
  elif [[ $kind == "oracle" || $kind == "parity_ref" ]]; then
    oclass="forbidden_as_claim_oracle"
    foreign_hard="unknown"
    foreign="python3"
    migration="rehome_sounio"
    notes="standalone_oracle_or_parity_ref"
    has_py=1
  else
    oclass="unknown"
    migration="none"
    notes="insufficient_signal"
  fi

  # Escape notes for TSV
  notes="${notes//$'\t'/ }"
  notes="${notes//$'\n'/ }"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
    "$foreign" "$tier" "$notes" "$migration" "$UTC"
}

{
  printf '# claim_oracle_inventory.tsv — provisional classifications (ADR-008)\n'
  printf '# schema: docs/decisions/claim_oracle_inventory.schema.md\n'
  printf '# scanner: scripts/dev/claim_oracle_inventory.sh\n'
  printf '# scanned_utc=%s\n' "$UTC"
  printf 'gate_id\tkind\toracle_class\tforeign_hard_fail\tsounio_witness\tforeign_runtimes\tci_tier\tnotes\tmigration\tscanned_utc\n'

  for p in "${CANDIDATES[@]}"; do
    classify_one "$p"
  done
} > "$OUT_TSV"

# Summary
total=$(grep -v '^#' "$OUT_TSV" | grep -v '^gate_id' | wc -l | tr -d ' ')
echo "[claim-oracle-inventory] wrote $OUT_TSV ($total rows)"
echo "[claim-oracle-inventory] by oracle_class:"
grep -v '^#' "$OUT_TSV" | grep -v '^gate_id' | cut -f3 | sort | uniq -c | sort -rn
echo "[claim-oracle-inventory] foreign_hard_fail=yes:"
grep -v '^#' "$OUT_TSV" | grep -v '^gate_id' | awk -F'\t' '$4=="yes"{print $1}' | head -40
n_hard=$(grep -v '^#' "$OUT_TSV" | grep -v '^gate_id' | awk -F'\t' '$4=="yes"' | wc -l | tr -d ' ')
echo "[claim-oracle-inventory] foreign_hard_fail=yes count=$n_hard"

exit 0
