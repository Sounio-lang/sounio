#!/usr/bin/env bash
# Execute the x86 generic/imported aggregate ABI regressions, not just emit ELF.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${1:?pass the current-source Linux x86 bootstrap compiler}"
case "$SEED" in /*) ;; *) SEED="$PWD/$SEED" ;; esac
WORK="${2:-$(mktemp -d /tmp/bootstrap-array-layout.XXXXXX)}"
mkdir -p "$WORK"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s 524288
for fixture in gen_dep_summary_min2 cd_exact_generic_i64 cd_exact_generic_vs_concrete \
    a13_crossmod_mainfirst_ok_ctrlE a13_crossmod_nonfirst_fn_drop_ctrlA \
    a13_crossmod_nonfirst_fn_drop_ctrlC; do
    timeout 90 "$SEED" "tests/run-pass/$fixture.sio" "$WORK/$fixture" > "$WORK/$fixture.build.log" 2>&1
    test -s "$WORK/$fixture"
    chmod +x "$WORK/$fixture"
    timeout 30 "$WORK/$fixture" > "$WORK/$fixture.stdout.log" 2> "$WORK/$fixture.stderr.log"
    case "$fixture" in
        gen_dep_summary_min2) grep -Fxq GEN_DEP_SUMMARY2_OK "$WORK/$fixture.stdout.log" ;;
        cd_exact_generic_i64) grep -Fxq 'SQ PASS' "$WORK/$fixture.stdout.log"; grep -Fxq 'NONZERO PASS' "$WORK/$fixture.stdout.log" ;;
        cd_exact_generic_vs_concrete) grep -Fxq 'BYTECOMPARE PASS' "$WORK/$fixture.stdout.log" ;;
        *) test "$(<"$WORK/$fixture.stdout.log")" = 2 ;;
    esac
    echo "PASS: x86 imported/generic aggregate layout $fixture"
done
