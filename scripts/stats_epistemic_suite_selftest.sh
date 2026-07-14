#!/usr/bin/env bash
# scripts/stats_epistemic_suite_selftest.sh
# Runs the inline self-tests of every module in the epistemic-statistics suite
# and the end-to-end demo. Each module prints "ALL PASS" on success.
# Usage:  bash scripts/stats_epistemic_suite_selftest.sh
set -u

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

MODULES=(
  stdlib/stats/student_t.sio
  stdlib/stats/chi_square.sio
  stdlib/stats/fisher_f.sio
  stdlib/stats/wilcoxon.sio
  stdlib/stats/bland_altman.sio
  stdlib/stats/power.sio
  stdlib/stats/qq_normal.sio
  stdlib/stats/anova.sio
  stdlib/stats/reg_bands.sio
  stdlib/stats/correlation.sio
  stdlib/stats/kruskal_wallis.sio
  stdlib/stats/tukey.sio
  stdlib/stats/chi2_independence.sio
  stdlib/stats/effect_size.sio
  stdlib/stats/proportion.sio
  stdlib/stats/densities.sio
  stdlib/stats/bayes_conjugate.sio
  stdlib/stats/diagnostic.sio
  stdlib/stats/cohen_kappa.sio
  stdlib/stats/roc.sio
  stdlib/stats/summary_inference.sio
  stdlib/stats/meta_analysis.sio
  stdlib/stats/epidemiology.sio
  stdlib/stats/sample_size.sio
  stdlib/stats/reliability.sio
  stdlib/stats/concordance.sio
  stdlib/stats/rate_epi.sio
  stdlib/stats/weibull_negbin.sio
  stdlib/stats/mcnemar.sio
  stdlib/stats/friedman.sio
  stdlib/stats/trend.sio
  stdlib/stats/ks_test.sio
  stdlib/stats/gof.sio
  stdlib/stats/normality.sio
  stdlib/stats/km.sio
  stdlib/stats/logrank.sio
  stdlib/stats/exp_survival.sio
  stdlib/stats/robust.sio
  stdlib/stats/hodges_lehmann.sio
  stdlib/stats/outlier.sio
  stdlib/stats/deming.sio
  stdlib/stats/theil_sen.sio
  stdlib/stats/passing_bablok.sio
  stdlib/stats/fisher_exact.sio
  stdlib/stats/cochran_q.sio
  stdlib/stats/fleiss_kappa.sio
)

fail=0
for m in "${MODULES[@]}"; do
  out="$("$SOUC" run "$m" 2>&1 | tail -1)"
  if [ "$out" = "ALL PASS" ]; then
    printf '  [PASS] %s\n' "$m"
  else
    printf '  [FAIL] %s  -> %s\n' "$m" "$out"
    fail=1
  fi
done

# Smoke-run the integration demo (exit 0 = every tool composed cleanly).
for demo in examples/stats/epistemic_suite_demo.sio examples/stats/full_analysis_report.sio \
            examples/stats/multiple_comparisons_test.sio examples/stats/permutation_test.sio \
            examples/stats/forest_plot.sio examples/stats/box_plot.sio \
            examples/stats/funnel_plot.sio; do
  if "$SOUC" run "$demo" >/dev/null 2>&1; then
    printf '  [PASS] %s\n' "$demo"
  else
    printf '  [FAIL] %s\n' "$demo"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "epistemic-statistics suite: ALL GREEN"
else
  echo "epistemic-statistics suite: FAILURES ABOVE"
fi
exit "$fail"
