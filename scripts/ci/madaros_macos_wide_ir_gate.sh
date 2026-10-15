#!/usr/bin/env bash
# Exact C++23 oracle -> real IR -> signed ARM64 Mach-O -> execution.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${1:?pass a current-source native bootstrap seed}"
case "$SEED" in /*) ;; *) SEED="$PWD/$SEED" ;; esac
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/madaros-wide-ir.XXXXXX")}"
mkdir -p "$WORK"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
"${CXX:-/usr/bin/clang++}" -std=c++23 -O2 -Wall -Wextra -Werror \
    -I"${BOOST_INCLUDE:-/opt/homebrew/include}" \
    scripts/ci/arm64_macos_wide_oracle.cpp -o "$WORK/oracle"
"$WORK/oracle" > "$WORK/generated.sio" 2> "$WORK/oracle.log"
./bin/souc --science-boundary off check "$WORK/generated.sio" > "$WORK/check.log" 2>&1
"$SEED" "$WORK/generated.sio" "$WORK/driver" --target aarch64-macos > "$WORK/build.log" 2>&1
chmod +x "$WORK/driver"
codesign -s - -f "$WORK/driver"
timeout 60 "$WORK/driver" "$WORK/program" > "$WORK/emit.log" 2>&1
test -s "$WORK/program"
file "$WORK/program" | grep -Eq 'Mach-O 64-bit.*arm64'
chmod +x "$WORK/program"
codesign -s - -f "$WORK/program"
codesign --verify "$WORK/program"
timeout 30 "$WORK/program" > "$WORK/run.log" 2>&1
cat "$WORK/oracle.log"
for mode in zero invalid; do
    "$WORK/oracle" "$mode" > "$WORK/$mode.sio"
    ./bin/souc --science-boundary off check "$WORK/$mode.sio" > "$WORK/$mode.check.log" 2>&1
    "$SEED" "$WORK/$mode.sio" "$WORK/$mode-driver" --target aarch64-macos > "$WORK/$mode.build.log" 2>&1
    chmod +x "$WORK/$mode-driver"
    codesign -s - -f "$WORK/$mode-driver"
    timeout 60 "$WORK/$mode-driver" "$WORK/$mode-program" > "$WORK/$mode.emit.log" 2>&1
    if [[ "$mode" == invalid ]]; then
        test ! -s "$WORK/$mode-program"
        grep -Fq ARM64_REFUSAL "$WORK/$mode.emit.log"
    else
        chmod +x "$WORK/$mode-program"
        codesign -s - -f "$WORK/$mode-program"
        set +e
        timeout 30 "$WORK/$mode-program" > "$WORK/$mode.run.log" 2>&1
        trap_status=$?
        set -e
        test "$trap_status" = 133
    fi
done
echo "PASS: exact wide arithmetic, signed/unsigned comparisons, shifts, division/remainder and destination aliases; artifacts=$WORK"
