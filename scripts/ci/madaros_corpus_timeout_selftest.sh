#!/usr/bin/env bash
# Selftest for the per-test run bound of madaros_corpus_regression_gate.sh.
#
# A run-pass test may declare `//@ timeout: SECONDS`; the corpus gate used to run
# every program under a fixed `timeout 30` and so reported tests/run-pass/serve_prio.sio
# (which declares 120 s and runs about 24 s on one core) as a new failure whenever
# the runner was loaded. This runs the gate's own per-program scripts, extracted from
# the gate, against a stub compiler whose programs sleep 33 s:
#   - with `//@ timeout: 60` the program must pass;
#   - with no directive it must still fail under the 30 s default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/scripts/ci/madaros_corpus_regression_gate.sh"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

awk '/^cat > "\$WORK\/run_one.sh" <<.INNER.$/ { f = 1; next } /^INNER$/ { f = 0 } f' "$GATE" > "$W/run_one.sh"
awk '/^cat > "\$WORK\/timeout_run.sh" <<.TR.$/ { f = 1; next } /^TR$/ { f = 0 } f' "$GATE" > "$W/timeout_run.sh"
[[ -s "$W/run_one.sh" && -s "$W/timeout_run.sh" ]] || { echo "FAIL: could not extract the gate's run scripts" >&2; exit 1; }
chmod +x "$W/run_one.sh" "$W/timeout_run.sh"

cat > "$W/madaros" <<'EOF'
#!/usr/bin/env bash
# stub: `compile SRC -o OUT` writes a program that sleeps 33 s and exits 0
printf '#!/bin/sh\nsleep 33\nexit 0\n' > "$4"
chmod +x "$4"
EOF
chmod +x "$W/madaros"
printf '//@ run-pass\n//@ timeout: 60\n' > "$W/with_directive.sio"
printf '//@ run-pass\n' > "$W/no_directive.sio"

# The two runs are independent; run them side by side (about 33 s in all).
WORK="$W" MADAROS="$W/madaros" "$W/run_one.sh" "$W/with_directive.sio" > "$W/with.out" &
WORK="$W" MADAROS="$W/madaros" "$W/run_one.sh" "$W/no_directive.sio" > "$W/without.out" &
wait
with="$(cat "$W/with.out")"
without="$(cat "$W/without.out")"

[[ -z "$with" ]] || { echo "FAIL: a 33 s program declaring '//@ timeout: 60' was reported: $with" >&2; exit 1; }
[[ "$without" == "no_directive.sio run" ]] || { echo "FAIL: a 33 s program with no directive was not timed out at 30 s (got '$without')" >&2; exit 1; }
echo "MADAROS_CORPUS_TIMEOUT_SELFTEST_OK"
