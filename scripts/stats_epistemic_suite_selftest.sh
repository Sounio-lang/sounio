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
  stdlib/stats/logistic.sio
  stdlib/stats/poisson_reg.sio
  stdlib/stats/wls.sio
  stdlib/stats/runs_test.sio
  stdlib/stats/durbin_watson.sio
  stdlib/stats/autocorr.sio
  stdlib/stats/nelson_aalen.sio
  stdlib/stats/rmst.sio
  stdlib/stats/life_table.sio
  stdlib/stats/kendall_tau.sio
  stdlib/stats/goodman_kruskal.sio
  stdlib/stats/somers_d.sio
  stdlib/stats/bartlett.sio
  stdlib/stats/levene.sio
  stdlib/stats/var_ftest.sio
  stdlib/stats/mann_whitney.sio
  stdlib/stats/sign_test.sio
  stdlib/stats/mood_median.sio
  stdlib/stats/fit_gamma.sio
  stdlib/stats/fit_beta.sio
  stdlib/stats/fit_lognormal.sio
  stdlib/stats/corr_ci.sio
  stdlib/stats/point_biserial.sio
  stdlib/stats/partial_corr.sio
  stdlib/stats/bayes_ab.sio
  stdlib/stats/beta_hdi.sio
  stdlib/stats/bic.sio
  stdlib/stats/mahalanobis.sio
  stdlib/stats/hotelling_t2.sio
  stdlib/stats/pca2.sio
  stdlib/stats/jackknife.sio
  stdlib/stats/rng.sio
  stdlib/stats/bootstrap.sio
  stdlib/stats/perm_test.sio
  stdlib/stats/bootstrap_diff.sio
  stdlib/stats/bootstrap_corr.sio
  stdlib/stats/mantel_haenszel.sio
  stdlib/stats/attributable.sio
  stdlib/stats/standardized_rate.sio
  stdlib/stats/exp_smoothing.sio
  stdlib/stats/holt.sio
  stdlib/stats/ar1.sio
  stdlib/stats/process_capability.sio
  stdlib/stats/control_chart.sio
  stdlib/stats/cusum.sio
  stdlib/stats/effect_convert.sio
  stdlib/stats/effect_from_test.sio
  stdlib/stats/cles.sio
  stdlib/stats/gwet_ac1.sio
  stdlib/stats/krippendorff.sio
  stdlib/stats/icc_forms.sio
  stdlib/stats/poisson_gamma.sio
  stdlib/stats/dirichlet_mult.sio
  stdlib/stats/breslow_day.sio
  stdlib/stats/leverage.sio
  stdlib/stats/cooks_distance.sio
  stdlib/stats/vif.sio
  stdlib/stats/sample_size_survival.sio
  stdlib/stats/sample_size_precision.sio
  stdlib/stats/detectable_effect.sio
  stdlib/stats/central_tendency.sio
  stdlib/stats/dispersion.sio
  stdlib/stats/shape.sio
  stdlib/stats/gk_lambda.sio
  stdlib/stats/gk_tau.sio
  stdlib/stats/uncertainty_coefficient.sio
  stdlib/stats/anderson_darling.sio
  stdlib/stats/cramer_von_mises.sio
  stdlib/stats/bowker.sio
  stdlib/stats/zscore.sio
  stdlib/stats/normalize.sio
  stdlib/stats/rank_transform.sio
  stdlib/stats/weighted_stats.sio
  stdlib/stats/grouped_stats.sio
  stdlib/stats/weighted_quantile.sio
  stdlib/stats/km_greenwood.sio
  stdlib/stats/conditional_survival.sio
  stdlib/stats/cumulative_incidence.sio
  stdlib/stats/linear_contrast.sio
  stdlib/stats/scheffe.sio
  stdlib/stats/welch_anova.sio
  stdlib/stats/jonckheere.sio
  stdlib/stats/page_trend.sio
  stdlib/stats/mann_kendall.sio
  stdlib/stats/huber_location.sio
  stdlib/stats/siegel_regression.sio
  stdlib/stats/yuen.sio
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
