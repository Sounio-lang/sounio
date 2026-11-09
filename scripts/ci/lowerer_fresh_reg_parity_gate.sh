#!/usr/bin/env bash
# Component regression using the actual production allocator bodies. Run on Slurm.
# Optional LOWERER_FRESH_REG_PROBE_GENERATOR supplies a C++23-built generator on
# workers without a C++ compiler; no substitute fixture or allocator is accepted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/lowerer-fresh-reg-parity.XXXXXX")}"
mkdir -p "$WORK"
case "$WORK" in /*) ;; *) WORK="$PWD/$WORK" ;; esac
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
GENERATOR="${LOWERER_FRESH_REG_PROBE_GENERATOR:-}"
if [[ -z "$GENERATOR" ]]; then
  GENERATOR="$WORK/generate"
  "${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror \
    scripts/ci/lowerer_fresh_reg_parity_probe.cpp -o "$GENERATOR"
fi
[[ -x "$GENERATOR" ]] || { echo 'fresh-reg probe generator is not executable' >&2; exit 2; }
"$GENERATOR" "$ROOT" "$WORK/probe.sio"
# Resolver and compiler overrides are inherited from the validation environment.
# Both operations use the repository's public canonical compiler entrypoint.
"$ROOT/bin/souc" check "$WORK/probe.sio" > "$WORK/check.log" 2>&1
"$ROOT/bin/souc" compile "$WORK/probe.sio" -o "$WORK/probe" > "$WORK/build.log" 2>&1
[[ -s "$WORK/probe" ]] || { echo 'fresh-reg probe executable missing' >&2; exit 1; }
chmod +x "$WORK/probe"
timeout 60 "$WORK/probe" > "$WORK/stdout.log" 2> "$WORK/stderr.log"
for n in 1 2 3 4 5 6 7; do
  grep -Fxq "fresh_reg_case $n PASS" "$WORK/stdout.log"
done
[[ "$(grep -Ec '^fresh_reg_case [1-7] PASS$' "$WORK/stdout.log")" == 7 ]]
grep -Fxq 'fresh_reg_production_parity PASS' "$WORK/stdout.log"
# Invalid-function probes run once through each production allocator.
[[ "$(grep -Fxc 'lower_probe: fresh_reg_invalid current_fn=-1 fn_count=1' "$WORK/stdout.log")" == 2 ]]
[[ "$(grep -Fxc 'lower_probe: fresh_reg_invalid current_fn=1 fn_count=1' "$WORK/stdout.log")" == 2 ]]
if grep -q 'FAIL' "$WORK/stdout.log"; then
  echo 'fresh-reg probe contains a failed case' >&2; exit 1
fi
cat "$WORK/stdout.log"
echo "PASS: production fresh_reg allocators preserve diagnostics, first failure, and register bounds (artifacts: $WORK)"
