#!/usr/bin/env bash
# A deliberately invalid compiler must fail and expose its diagnostic in CI.
set -euo pipefail
HARNESS="${1:?pass the compiled corpus harness}"
WORK="${2:?pass a fresh negative-control output directory}"
set +e
"$HARNESS" /bin/sh tests/manifests/madaros_macos_200.txt "$WORK" 1 10 > "$WORK.log" 2>&1
status=$?
set -e
test "$status" = 1
grep -Fq 'FAIL aggregate_local_binding_is_value_copy.sio status=compile_failed compile_rc=' "$WORK.log"
grep -Fq -- '--- aggregate_local_binding_is_value_copy.sio/build.log ---' "$WORK.log"
grep -Eq 'cannot open build|build: No such file|build:.*No such file' "$WORK.log"
echo 'PASS: deliberate compiler failure stays red and exposes its diagnostic'
