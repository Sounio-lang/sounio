#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${SOUNIO_G30F_WORKSPACE_GATE_DIR:-$(mktemp -d /tmp/sounio-g30f-workspace.XXXXXX)}"
SOURCE="$ROOT_DIR/tests/native-v2/gri30_full_workspace_uq_one_step.sio"
TRACE="$WORK/trace.log"
ELF="$WORK/uq-one-step.elf"
RAW="${SOUNIO_G30F_WORKSPACE_GATE_BIN:-${MADAROS_RAW_BIN:-}}"

fail() {
    echo "[gri30-full-workspace] FAIL: $*" >&2
    exit 1
}

trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

if [[ -z "$RAW" ]]; then
    RAW="$WORK/madaros-current"
    bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" \
        >"$WORK/build.log" 2>&1 || {
        cat "$WORK/build.log" >&2
        fail "current-source Madaros build failed"
    }
fi
[[ -x "$RAW" ]] || fail "Madaros compiler is missing or not executable: $RAW"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
echo "[gri30-full-workspace] compiler_sha256=$(sha256sum "$RAW" | awk '{print $1}')"

SOUNIO_NV2_IR_TRACE=1 MADAROS_RAW_BIN="$RAW" \
    "$ROOT_DIR/bin/madaros" compile "$SOURCE" -o "$ELF" >"$TRACE" 2>&1 || {
    cat "$TRACE" >&2
    fail "one-step UQ witness did not compile"
}

awk '
BEGIN {
    target["g30f_ws_g_rt"] = 1
    target["g30f_ws_fill_kc"] = 1
    target["g30f_ws_rates"] = 1
    target["g30f_ws_rhs"] = 1
    target["g30f_ws_jacobian"] = 1
    target["g30f_ws_sens_rhs"] = 1
    current = ""
}
/^NV2_IR function / {
    current = ""
    if (match($0, /name=[^ ]+/)) {
        fn_name = substr($0, RSTART + 5, RLENGTH - 5)
        if (fn_name in target) {
            current = fn_name
            seen[fn_name] = 1
        }
    }
    next
}
/^ name=/ {
    split(substr($0, 7), fields, " ")
    fn_name = fields[1]
    if (fn_name in target) {
        current = fn_name
        seen[fn_name] = 1
    }
    next
}
/^ op=alloc / {
    if (current != "") allocs[current]++
}
/^NV2_IR fn=.* op=alloc / {
    if (current != "") allocs[current]++
}
END {
    failed = 0
    for (name in target) {
        count = allocs[name] + 0
        printf("[gri30-full-workspace] helper=%s direct_ir_allocs=%d\n", name, count)
        if (!seen[name] || count != 0) failed = 1
    }
    exit failed
}
' "$TRACE" || fail "hot workspace helpers are missing from the trace or allocate per call"

set +e
timeout 30 "$ELF" >"$WORK/run.stdout" 2>"$WORK/run.stderr"
rc=$?
set -e
if [[ "$rc" -ne 0 ]] || ! grep -Fxq "UQ1 PASS" "$WORK/run.stdout"; then
    cat "$WORK/run.stdout" >&2 || true
    cat "$WORK/run.stderr" >&2 || true
    fail "one-step UQ witness failed: rc=$rc"
fi

echo "[gri30-full-workspace] PASS uq_steps=1 rc=0 marker=UQ1_PASS"
