#!/bin/bash
# souc-native-wrapper.sh — Drop-in replacement for `souc` CLI using the native compiler
# Usage: souc-native-wrapper.sh check file.sio   → compile to verify (exit 0 = ok)
#        souc-native-wrapper.sh run file.sio      → compile and execute
#        souc-native-wrapper.sh compile file.sio output.elf
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Find the native compiler
NATIVE="${SOUC_NATIVE_BIN:-/tmp/souc-native.elf}"
if [ ! -x "$NATIVE" ]; then
    # Try to build it
    bash "$SCRIPT_DIR/build_native_souc.sh" "$NATIVE" 2>/dev/null || true
fi

if [ ! -x "$NATIVE" ]; then
    echo "error: native compiler not available at $NATIVE" >&2
    exit 1
fi

CMD="${1:-help}"
shift || true

case "$CMD" in
    --version|-v|version)
        echo "souc-native 1.0.0 (self-hosted lean_single.sio)"
        exit 0
        ;;
    check)
        FILE="$1"
        TMP=$(mktemp /tmp/souc-check-XXXXXX.elf)
        trap "rm -f $TMP" EXIT
        if "$NATIVE" "$FILE" "$TMP" >/dev/null 2>&1; then
            echo "ok: $FILE"
            exit 0
        else
            echo "error: $FILE failed compilation" >&2
            exit 1
        fi
        ;;
    run)
        FILE="$1"
        shift || true
        TMP=$(mktemp /tmp/souc-run-XXXXXX.elf)
        trap "rm -f $TMP" EXIT
        "$NATIVE" "$FILE" "$TMP" >/dev/null 2>&1
        chmod +x "$TMP"
        "$TMP" "$@"
        ;;
    compile)
        FILE="$1"
        OUT="${2:-a.elf}"
        "$NATIVE" "$FILE" "$OUT"
        chmod +x "$OUT"
        ;;
    repl)
        echo "error: REPL not supported by native compiler" >&2
        exit 1
        ;;
    *)
        echo "Usage: souc-native-wrapper.sh {check|run|compile} file.sio [output.elf]"
        exit 1
        ;;
esac
