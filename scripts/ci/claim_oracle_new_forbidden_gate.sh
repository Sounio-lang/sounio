#!/usr/bin/env bash
# claim_oracle_new_forbidden_gate.sh — the enforcement point of ADR-008.
#
# ADR-008 ("single semantic clock") says pass/fail of a language or library
# claim is decided by Sounio witnesses and nowhere else, and that CI "may
# later fail on new rows classified forbidden_as_claim_oracle". The inventory
# (scripts/dev/claim_oracle_inventory.sh -> artifacts/audit/
# claim_oracle_inventory.tsv) has always been observational: it records, it
# does not prevent. This gate is the "later": it turns a NEW foreign judge
# into a red build.
#
# What it checks: after regenerating the inventory, every row whose
# oracle_class is forbidden_as_claim_oracle or whose foreign_hard_fail is
# "yes" must be listed in the grandfather baseline
# (scripts/ci/claim_oracle_grandfathered_forbidden.tsv). As of the wiring
# date (2026-08-31) the baseline is EMPTY because the inventory is clean;
# it exists so a legacy migration in flight can be grandfathered with a
# written justification instead of being silently tolerated.
#
# If this gate fails on your new gate, the fix is one of:
#   1. give the claim a Sounio witness (sounio_native_expected), or
#   2. demote the foreign path to report-only (external_corroboration_only;
#      scripts/ci/lib_sounio_claim_oracle.sh is the shared helper), or
#   3. only for a migration already in flight, add the gate_id to the
#      grandfather baseline with a written reason — that line is a deliberate
#      act in a diff, which is the point.
#
# SOUNIO_CLAIM_ORACLE_TSV overrides the inventory path and
# SOUNIO_CLAIM_ORACLE_BASELINE the baseline path (used by the gate's own
# fixtures/tests); by default the inventory is regenerated first.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

BASELINE="${SOUNIO_CLAIM_ORACLE_BASELINE:-scripts/ci/claim_oracle_grandfathered_forbidden.tsv}"
TSV="${SOUNIO_CLAIM_ORACLE_TSV:-}"

if [[ -z "$TSV" ]]; then
  bash scripts/dev/claim_oracle_inventory.sh
  TSV="artifacts/audit/claim_oracle_inventory.tsv"
fi

[[ -f "$TSV" ]] || { echo "CLAIM_ORACLE_NEW_FORBIDDEN_GATE_FAIL: no inventory at $TSV"; exit 1; }
[[ -f "$BASELINE" ]] || { echo "CLAIM_ORACLE_NEW_FORBIDDEN_GATE_FAIL: missing baseline $BASELINE"; exit 1; }

total=$(awk -F'\t' '$0 !~ /^#/ && $1!="gate_id" && $1!=""' "$TSV" | wc -l)

# Forbidden rows: foreign runtime as sole hard-fail judge of a claim.
forbidden=$(awk -F'\t' '$0 !~ /^#/ && $1!="gate_id" && ($3=="forbidden_as_claim_oracle" || $4=="yes") {print $1}' "$TSV" | sort -u)
grandfathered=$(awk -F'\t' '$0 !~ /^#/ && $1!="" {print $1}' "$BASELINE" | sort -u)

new_rows=$(comm -23 <(printf '%s\n' "$forbidden") <(printf '%s\n' "$grandfathered") | sed '/^$/d')

n_forbidden=$(printf '%s\n' "$forbidden" | sed '/^$/d' | wc -l)
n_grand=$(printf '%s\n' "$grandfathered" | sed '/^$/d' | wc -l)
n_new=$(printf '%s\n' "$new_rows" | sed '/^$/d' | wc -l)

echo "  inventory rows            : $total ($TSV)"
echo "  forbidden claim-oracles   : $n_forbidden"
echo "  grandfathered (baseline)  : $n_grand"
echo "  NEW forbidden             : $n_new"

if [[ -n "$new_rows" ]]; then
  echo
  echo "CLAIM_ORACLE_NEW_FORBIDDEN_GATE_FAIL: new gate(s) hard-fail on a foreign"
  echo "runtime's judgment of a Sounio claim (ADR-008 forbids this):"
  printf '  - %s\n' $new_rows
  echo
  echo "Fix by (1) adding a Sounio witness for the claim, (2) demoting the"
  echo "foreign path to report-only via scripts/ci/lib_sounio_claim_oracle.sh,"
  echo "or (3) only for an in-flight legacy migration, grandfathering the"
  echo "gate_id in $BASELINE with a written justification."
  exit 1
fi

echo "CLAIM_ORACLE_NEW_FORBIDDEN_GATE_OK"
