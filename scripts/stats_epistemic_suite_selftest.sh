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
if "$SOUC" run examples/stats/epistemic_suite_demo.sio >/dev/null 2>&1; then
  printf '  [PASS] examples/stats/epistemic_suite_demo.sio\n'
else
  printf '  [FAIL] examples/stats/epistemic_suite_demo.sio\n'
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "epistemic-statistics suite: ALL GREEN"
else
  echo "epistemic-statistics suite: FAILURES ABOVE"
fi
exit "$fail"
