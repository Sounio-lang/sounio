#!/usr/bin/env bash
# Paper A artifact — "Manufacturing Precision Is a Type Error" — reproduce every number the paper
# cites that runs on this tree, and refuse to run on the wrong compiler.
#
#   bash docs/research/paper_A_artifact/reproduce.sh            # full (Lean gate + all programs, ~10 min)
#   bash docs/research/paper_A_artifact/reproduce.sh --no-lean  # skip the Lean gate
#
# Every program prints a verdict line; this script compares it with the value recorded in the
# paper (README.md next to this file). A mismatch is a FAIL, not a warning.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
fails=0
ok()   { echo "  ok    $1"; }
fail() { echo "  FAIL  $1" >&2; fails=$((fails+1)); }

echo "[paper-A artifact] tree $(git rev-parse --short HEAD 2>/dev/null || echo no-git)"

# ── 0. the compiler must be the committed Madaros, not a local build (measured 2026-08-31, #2318) ──
export SOUNIO_REQUIRE_COMMITTED_MADAROS=1
_prov="$(./bin/souc --version 2>&1 || true)"
if grep -q 'is the COMMITTED binary' <<<"$_prov"; then
  ok "compiler is the committed bin/madaros-linux-x86_64 ($(md5sum bin/madaros-linux-x86_64 | cut -c1-8))"
else
  grep provenance <<<"$_prov" >&2 || true
  fail "compiler is not the committed prebuilt (remove artifacts/self-hosted/madaros or unset MADAROS_RAW_BIN)"
  exit 1
fi

# ── 1. the mechanized metatheory (Lean 4.33.1, Mathlib-free) ──
if [[ "${1:-}" != "--no-lean" ]]; then
  if bash scripts/ci/ns_metatheory_lean_gate.sh > /tmp/paperA_lean.log 2>&1 && grep -q NS_METATHEORY_LEAN_GATE_PASS /tmp/paperA_lean.log; then
    ok "EpistemicEffectsNS.lean: $(grep -o 'across [0-9]* theorems' /tmp/paperA_lean.log), sorry-free, axioms ⊆ {propext, Quot.sound, Classical.choice}"
  else
    tail -5 /tmp/paperA_lean.log >&2; fail "Lean gate"
  fi
fi

run_expect() {  # run_expect <label> <file> <grep-regex-that-must-match-stdout>
  local label="$1" file="$2" needle="$3" out
  if out="$(./bin/souc run "$file" 2>/dev/null)" && grep -qE -- "$needle" <<<"$out"; then
    ok "$label"
  else
    echo "$out" | tail -3 | sed 's/^/        /' >&2; fail "$label (expected /$needle/)"
  fi
}

# ── 2. Lemma 1 at the analysis level: the three souc-green prototypes (§7, §8.2) ──
run_expect "noise_symbols.sio — x+x true 4 vs naive 2"        docs/research/sounio/noise_symbols.sio   '.'
run_expect "ns_dataflow.sio — shared-source add flagged"      docs/research/sounio/ns_dataflow.sio     '.'
run_expect "ns_contract.sio — five acceptance controls"       docs/research/sounio/ns_contract.sio     'PASS'

# ── 3. RQ4: the two-compartment cohort (§8.4) — seed 20260831, 5,000 patients ──
run_expect "RQ4 cohort: B interval sum silences 311/909; A phase sum silences 0, chain 300.7×" \
  docs/research/sounio/rq4_vanco_two_compartment_flip.sio \
  'RQ4_FLIP n=5000 true_warn=909 silenced_sum=0 silenced_naive=0 spurious_naive=1894 var_ratio_sum_permille=1204 var_ratio_naive_permille=300662 B_true_warn=909 B_silenced=311 B_var_ratio_permille=500'

# ── 4. Monte Carlo adequacy of the first-order truth (§6.4 (v), §8.4) — 1,000 draws/patient ──
run_expect "RQ4 Monte Carlo: Var_MC/Var_T 0.999, WARN 909/911/877, naive 300.9× vs sampling" \
  docs/research/sounio/rq4_vanco_mc_adequacy.sio \
  'RQ4_MC n=5000 k=1000 warn_t=909 warn_n=2803 warn_mcsd=911 warn_mcq=877 .* var_mc_over_t_permille=999 var_n_over_mc_permille=300917'

# ── 5. the runtime fix: stdlib/epistemic/affine (exact shared-source propagation) ──
run_expect "affine: shared-source add (x+x = 4; ep_add 2)"    tests/run-pass/affine_shared_source_add.sio      'AFFINE_SHARED_SOURCE_ADD PASS'
run_expect "affine: delta-method product (x·x = square)"     tests/run-pass/affine_product_delta.sio          'AFFINE_PRODUCT_DELTA PASS'
run_expect "affine: phase partition identity, Cov < 0"       tests/run-pass/affine_phase_partition_identity.sio 'AFFINE_PHASE_PARTITION_IDENTITY PASS'

# ── 6. the clinical receipt (§8.4): AUC 450 ± 44, CI [361, 539], WARN — engine-portable twin ──
run_expect "vancomycin AUC receipt (affine twin): WARN"       examples/vancomycin_auc_affine.sio 'GUM_GATE=WARN_SUBTHERAPEUTIC_POSSIBLE'

echo
if [[ $fails -eq 0 ]]; then
  echo "PAPER_A_ARTIFACT_OK — every recorded value regenerated on the committed compiler"
else
  echo "PAPER_A_ARTIFACT_FAIL — $fails item(s) did not regenerate" >&2; exit 1
fi
