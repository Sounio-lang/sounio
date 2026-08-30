#!/usr/bin/env bash
# scripts/dev/claim_oracle_inventory.sh
#
# Scan gate / oracle scripts and emit a provisional claim-oracle inventory
# (ADR-008). Classifications are heuristic - overrides may be supplied later.
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
  local has_adr008_soft=0

  echo "$text" | grep -qE 'python3|python |[.]py["'"'"' ]' && has_py=1 || true
  # resolve_souc / SOUC_BIN / sounio_require_souc count as Sounio surface
  echo "$text" | grep -qE 'bin/souc|SOUC|souc run|souc check|resolve_souc|sounio_require_souc|SOUC_BIN' && has_souc=1 || true
  echo "$text" | grep -qE '\bdiff\b|/usr/bin/diff' && has_diff=1 || true
  echo "$text" | grep -qE 'fail=1|exit 1|GATE FAILED|ORACLE MISMATCH' && has_fail=1 || true
  echo "$text" | grep -qiE 'mpmath|scipy|numpy' && has_mpmath=1 || true
  echo "$text" | grep -qE 'SKIP:|exit 0.*mpmath|not installed' && has_skip=1 || true
  echo "$text" | grep -qE 'ALL PASS|GUM_TRUST_OK|_OK"|expect-stdout' && has_all_pass=1 || true
  echo "$text" | grep -qiE 'epistemic_trust|GUM_TRUST' && has_trust=1 || true
  echo "$text" | grep -qiE 'gen2|gen3|fixed.point|fixed-point|boot4' && has_fixed=1 || true
  echo "$text" | grep -qiE '\bleach\b|lake build|formal/' && has_lean=1 || true
  echo "$text" | grep -qiE 'optional.*oracle|oracle.*optional|corroboration' && has_optional=1 || true
  # ADR-008 demotion markers: foreign path soft unless SOUNIO_FOREIGN_ORACLE_HARD=1
  echo "$text" | grep -qE 'lib_sounio_claim_oracle|SOUNIO_FOREIGN_ORACLE_HARD|sounio_foreign_mismatch|sounio_foreign_diff|ADR-008' && has_adr008_soft=1 || true
  # C research contracts (l8/l9 census) still count as non-peer-language claim clocks
  local has_c_contract=0
  echo "$text" | grep -qE '_VERDICT PASS|l8_zd_census|l9_zd_census|cc -O|CC_BIN' && has_c_contract=1 || true

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

  # Path-based residual classes (not dual Sounio+foreign claim judges).
  case "$path" in
    scripts/archive/*)
      oclass="research_harness"
      foreign_hard="no"
      migration="none"
      notes="archived_gate_not_active_claim_clock"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
        "$foreign" "$tier" "$notes" "$migration" "$UTC"
      return 0
      ;;
    scripts/bootstrap/*|scripts/selfhost/*|*bootstrap_chain*|*bootstrap_full*|*bootstrap_kernel*)
      oclass="bootstrap_integrity"
      foreign_hard="no"
      migration="keep"
      notes="bootstrap_or_selfhost_integrity"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
        "$foreign" "$tier" "$notes" "$migration" "$UTC"
      return 0
      ;;
    scripts/ci/fixtures/*)
      oclass="research_harness"
      foreign_hard="no"
      migration="none"
      notes="fixture_not_product_gate"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
        "$foreign" "$tier" "$notes" "$migration" "$UTC"
      return 0
      ;;
    *suffering_aware*|*self_falsifying_compilation_line*|*sac_llm*|*mercyful_machine*|*moonshot_*|*federated_san*|*san_real_patient*|*chingon_zd*|*ade_wildgen*|*e_series_semantic*|*falsification_ledger*|*168_biology*|*associator_gum*|*cd_zd_graph*|*cd_tower_nullity*|*functor_f_g2*|*g2_zd_fibers*|*garden_to_claim*)
      if [[ $has_souc -eq 0 ]]; then
        oclass="research_harness"
        foreign_hard="no"
        migration="none"
        notes="python_or_shell_research_contract_not_language_claim"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
          "$foreign" "$tier" "$notes" "$migration" "$UTC"
        return 0
      fi
      ;;
  esac

  # native_v2 golden-stdout gates: Sounio ELF vs expected file, not peer-language oracle
  case "$path" in
    *native_v2_*_gate.sh)
      oclass="sounio_native_expected"
      foreign_hard="no"
      migration="keep"
      notes="native_v2_sounio_golden_or_self_check"
      sounio_wit="yes"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
        "$foreign" "$tier" "$notes" "$migration" "$UTC"
      return 0
      ;;
  esac

  # claim_ast / claim_native: Sounio typecheck is claim; python preprocessor is tooling
  if [[ "$path" == *claim_ast_gate* || "$path" == *claim_native_gate* ]]; then
    oclass="sounio_native_expected"
    foreign_hard="no"
    migration="keep"
    notes="claim_sounio_check_plus_optional_tooling"
    sounio_wit="partial"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
      "$foreign" "$tier" "$notes" "$migration" "$UTC"
    return 0
  fi

  # Shell meta / ops gates with no Sounio and no Python: operational, not claim clocks
  if [[ $has_souc -eq 0 && $has_py -eq 0 && $has_c_contract -eq 0 ]]; then
    oclass="research_harness"
    foreign_hard="no"
    migration="none"
    notes="shell_meta_or_ops_gate_no_claim_clock"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$path" "$kind" "$oclass" "$foreign_hard" "$sounio_wit" \
      "$foreign" "$tier" "$notes" "$migration" "$UTC"
    return 0
  fi

  if [[ $has_fixed -eq 1 && $has_py -eq 0 ]]; then
    oclass="bootstrap_integrity"
    migration="keep"
    notes="fixed-point_or_bootstrap_signals"
  elif [[ $has_lean -eq 1 && $has_souc -eq 0 && $has_py -eq 0 ]]; then
    oclass="formal_only"
    migration="keep"
    notes="lean_or_formal_path"
  elif [[ $has_adr008_soft -eq 1 ]]; then
    # Demoted dual clock (Sounio and/or C claim + soft foreign)
    foreign_hard="no"
    migration="keep"
    if [[ $has_souc -eq 1 || $has_c_contract -eq 1 ]]; then
      oclass="external_corroboration_only"
      notes="adr008_claim_plus_soft_foreign"
      sounio_wit="partial"
      [[ $has_souc -eq 1 || $has_c_contract -eq 1 ]] && sounio_wit="yes"
    else
      oclass="external_corroboration_only"
      notes="adr008_soft_foreign_only"
    fi
  elif [[ $has_py -eq 1 && $has_fail -eq 1 ]] && { [[ $has_diff -eq 1 ]] || [[ $has_mpmath -eq 1 ]]; }; then
    # Numeric foreign judge on a fail path. Prefer soft inventory when Sounio
    # already witnesses the claim (ADR-008 single clock); hard-flag only pure
    # foreign claim clocks still pending demotion.
    if [[ $has_souc -eq 1 || $has_c_contract -eq 1 || $has_optional -eq 1 ]]; then
      oclass="external_corroboration_only"
      foreign_hard="no"
      migration="demote_corroboration"
      notes="numeric_foreign_with_sounio_or_optional_soft"
      sounio_wit="partial"
      [[ $has_souc -eq 1 || $has_c_contract -eq 1 ]] && sounio_wit="yes"
    else
      oclass="forbidden_as_claim_oracle"
      foreign_hard="yes"
      migration="rehome_sounio"
      notes="python_diff_or_mpmath_on_fail_path"
    fi
  elif [[ $has_souc -eq 1 && $has_py -eq 1 ]]; then
    # Dual surface without numeric foreign judge (no diff/mpmath): Sounio is
    # the claim clock; Python is tooling / helper (LSP, packaging, meta).
    oclass="sounio_native_expected"
    foreign_hard="no"
    migration="keep"
    notes="dual_souc_python_tooling_not_numeric_judge"
    sounio_wit="yes"
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
  elif [[ $has_c_contract -eq 1 && $has_souc -eq 0 ]]; then
    # CUDA / C receipt gates (kretikos phases, FPGA): not peer-language claim clocks
    oclass="research_harness"
    foreign_hard="no"
    migration="none"
    notes="hardware_or_c_receipt_not_language_claim_clock"
  elif [[ $has_lean -eq 1 && $has_souc -eq 0 ]]; then
    oclass="formal_only"
    foreign_hard="no"
    migration="keep"
    notes="lean_formal_with_optional_tooling"
  elif [[ $has_py -eq 1 && $has_souc -eq 0 ]]; then
    # Python-only gate without numeric foreign pairing: paper/meta/research
    oclass="research_harness"
    foreign_hard="no"
    migration="none"
    notes="python_only_gate_without_numeric_claim_pairing"
  elif [[ $kind == "oracle" || $kind == "parity_ref" ]]; then
    # Standalone research oracles: measurement helpers, not claim gates
    oclass="research_harness"
    foreign_hard="no"
    foreign="python3"
    migration="none"
    notes="standalone_measurement_oracle_not_gate"
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
  printf '# claim_oracle_inventory.tsv - provisional classifications (ADR-008)\n'
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
