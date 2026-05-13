#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_FILE="${1:-$ROOT/self-hosted/compiler/lean_single.sio}"
STAGE0_COMPILER="${2:-$ROOT/bootstrap/boot4_a1.elf}"
TIMEOUT_SECS="${SELF_HOST_TIMEOUT_SECS:-60}"
OUT_DIR="${SELF_HOST_PROBE_DIR:-/tmp/sounio-self-host-probe-$$}"

mkdir -p "$OUT_DIR"

GEN1="$OUT_DIR/gen1.elf"
GEN2="$OUT_DIR/gen2.elf"
GEN3="$OUT_DIR/gen3.elf"
LOG1="$OUT_DIR/gen1.log"
LOG2="$OUT_DIR/gen2.log"
LOG3="$OUT_DIR/gen3.log"

count_unknown() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        grep -c "warning: unknown function" "$log_file" || true
    else
        echo 0
    fi
}

show_tail() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        tail -n 12 "$log_file"
    fi
}

echo "probe_dir=$OUT_DIR"
echo "source=$SOURCE_FILE"
echo "stage0=$STAGE0_COMPILER"
echo "timeout_secs=$TIMEOUT_SECS"

echo
echo "[1/3] stage0 -> gen1"
"$STAGE0_COMPILER" "$SOURCE_FILE" "$GEN1" >"$LOG1" 2>&1
chmod +x "$GEN1"
echo "unknown_warnings=$(count_unknown "$LOG1")"
show_tail "$LOG1"

echo
echo "[2/3] gen1 -> gen2"
if timeout "${TIMEOUT_SECS}s" "$GEN1" "$SOURCE_FILE" "$GEN2" >"$LOG2" 2>&1; then
    chmod +x "$GEN2"
    echo "unknown_warnings=$(count_unknown "$LOG2")"
    show_tail "$LOG2"
else
    status=$?
    echo "gen1 -> gen2 failed with status=$status"
    show_tail "$LOG2"
    exit "$status"
fi

echo
echo "[3/3] gen2 -> gen3"
if timeout "${TIMEOUT_SECS}s" "$GEN2" "$SOURCE_FILE" "$GEN3" >"$LOG3" 2>&1; then
    chmod +x "$GEN3"
    echo "unknown_warnings=$(count_unknown "$LOG3")"
    show_tail "$LOG3"
    echo
    md5sum "$GEN2" "$GEN3"
else
    status=$?
    echo "gen2 -> gen3 failed with status=$status"
    show_tail "$LOG3"
    exit "$status"
fi
