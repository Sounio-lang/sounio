#!/usr/bin/env bash
# Process-scoped ablation semantics plus a real allocation-read discriminator.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
host=${1:?source-built Madaros}
out_root=${2:?evidence directory}
mkdir -p "$out_root"
out=$(mktemp -d "$out_root/run.XXXXXX")
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_NOCACHE=1
source scripts/lib/souc_invoke.sh
_souc_limits
for mode in 0 1; do
  env SOUNIO_SCO_ABLATION="$mode" SOUNIO_INTO_ACC_ALLOC_TRACE=1 "$host" build \
    tests/run-pass/madaros_flat_arg_dep_reset.sio --target x86_64-linux \
    -o "$out/import-$mode" >"$out/import-$mode.log" 2>&1
  chmod +x "$out/import-$mode"
  "$out/import-$mode" >"$out/import-$mode.stdout"
  grep -Fxq FLAT_ARG_DEP_RESET_OK "$out/import-$mode.stdout"
  grep '^into_acc_alloc: ' "$out/import-$mode.log" >"$out/reads-$mode.log"
  test -s "$out/reads-$mode.log"
  if grep -v ' strategy_env_calls=1 ' "$out/reads-$mode.log"; then
    echo 'FAIL: strategy environment was not read exactly once' >&2; exit 1
  fi
  env SOUNIO_SCO_ABLATION="$mode" SOUNIO_SCO_STRATEGY_TRACE=1 "$host" compile \
    examples/sco_corpus/strategy_probe/probe_nonassoc.sio \
    -o "$out/strategy-$mode" >"$out/strategy-$mode.log" 2>&1
  expected=2; if test "$mode" = 1; then expected=0; fi
  grep '^SCO_STRATEGY function=sco_probe_leaf strategy=' "$out/strategy-$mode.log" >"$out/strategy-$mode.trace"
  test -s "$out/strategy-$mode.trace"
  if grep -vx "SCO_STRATEGY function=sco_probe_leaf strategy=$expected" "$out/strategy-$mode.trace"; then
    echo 'FAIL: ablation strategy changed' >&2; exit 1
  fi
  chmod +x "$out/strategy-$mode"
  "$out/strategy-$mode" >"$out/strategy-$mode.stdout"
  grep -Fxq 1 "$out/strategy-$mode.stdout"
done
echo SCO_ENV_ONCE_OK
