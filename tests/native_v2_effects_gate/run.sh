#!/usr/bin/env bash
# native_v2_effects_gate/run.sh
#
# Regression gate for algebraic-effect ENFORCEMENT in the self-hosted checker (M2)
# plus the build-mode codegen firewall (SOUNIO_CODEGEN_FIREWALL).
#
# The checker ships at WARN mode (effects_enforcement_mode() == 1). The call-site
# source-tag hooks (println / `/` / `%` / Box::new used without the matching declared
# effect) emit a NON-FATAL `warning[W035]` line and do NOT fail the check. The
# always-on callee-effect-subset check (pure fn calling a `with IO` fn) is NOT routed
# through the toggle and stays a hard E035 REJECT.
#
# The codegen firewall (FIREWALL_ON/FIREWALL_OFF rows) is gated by the
# SOUNIO_CODEGEN_FIREWALL env var, which this runner sets per row: =1 for FIREWALL_ON,
# =0 for every other row. When ON, any function declaring LLM(22)/Prob(7) on the
# codegen path is rejected with a hard error[E035].
#
#   accept/io_io.sio              <-> reject/pure_calls_io.sio        (callee IO subset, hard REJECT)
#   accept/println_with_io.sio    <-> reject/println_no_io.sio        (IO output builtin, W035)
#   accept/div_with_div.sio       <-> reject/div_no_div.sio           (Div on / and %, W035)
#   accept/box_with_alloc.sio     <-> reject/box_no_alloc.sio         (Alloc on Box::new, W035)
#   accept/user_fn_with_llm.sio   <-> reject/codegen_fn_with_llm.sio  (LLM firewall, flag-gated)
#
# Usage:   run.sh <path-to-check-binary>
# The check binary is invoked as:  <bin> --check <file>
# Verdicts are parsed from STDOUT (exit code is unreliable on this toolchain):
#   REJECT_E035   => stdout has NO "check: OK" AND has "error[E035]"
#   ACCEPT_CLEAN  => stdout has "check: OK" AND has NO "warning[W035]"
#   ACCEPT_WARN   => stdout has "check: OK" AND has "warning[W035]"
#
# Expected results live in EXPECTED.txt (single source of truth).
set -uo pipefail

GATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$GATE_DIR/../.." && pwd)"

CHECK_BIN="${1:-${SOUNIO_CHECK_BIN:-}}"
if [[ -z "$CHECK_BIN" ]]; then
    echo "[native-v2-effects] FAIL: no check binary given (arg 1 or SOUNIO_CHECK_BIN)" >&2
    exit 2
fi
if [[ ! -x "$CHECK_BIN" ]]; then
    echo "[native-v2-effects] FAIL: check binary not executable: $CHECK_BIN" >&2
    exit 2
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

EXPECTED="$GATE_DIR/EXPECTED.txt"
if [[ ! -f "$EXPECTED" ]]; then
    echo "[native-v2-effects] FAIL: missing EXPECTED.txt" >&2
    exit 2
fi

PASS=0
TOTAL=0
FAILED=0

while read -r relpath verdict phase _rest; do
    # skip comments / blanks
    [[ -z "${relpath:-}" ]] && continue
    case "$relpath" in \#*) continue ;; esac

    TOTAL=$((TOTAL + 1))
    witness="$GATE_DIR/$relpath"

    if [[ ! -f "$witness" ]]; then
        echo "[native-v2-effects] FAIL[$relpath]: witness file missing" >&2
        FAILED=1
        continue
    fi

    # Build-mode firewall is gated per row: ON only for FIREWALL_ON, else explicitly 0
    # (so the firewall never fires on the non-firewall rows regardless of ambient env).
    if [[ "$phase" == "FIREWALL_ON" ]]; then
        fw=1
    else
        fw=0
    fi

    out="$(SOUNIO_CODEGEN_FIREWALL="$fw" "$CHECK_BIN" --check "$witness" 2>&1)"

    has_ok=0;   grep -q "check: OK"       <<<"$out" && has_ok=1
    has_e035=0; grep -q "error\[E035\]"   <<<"$out" && has_e035=1
    has_w035=0; grep -q "warning\[W035\]" <<<"$out" && has_w035=1

    ok=1
    msg=""
    case "$verdict" in
        REJECT_E035)
            [[ "$has_ok" -eq 1 ]]   && { ok=0; msg="expected REJECT but got 'check: OK'"; }
            [[ "$has_e035" -ne 1 ]] && { ok=0; msg="${msg:+$msg; }rejected without error[E035]"; }
            ;;
        ACCEPT_CLEAN)
            [[ "$has_ok" -ne 1 ]]   && { ok=0; msg="expected ACCEPT ('check: OK') not found"; }
            [[ "$has_w035" -eq 1 ]] && { ok=0; msg="${msg:+$msg; }unexpected warning[W035] on clean accept"; }
            ;;
        ACCEPT_WARN)
            [[ "$has_ok" -ne 1 ]]   && { ok=0; msg="expected ACCEPT ('check: OK') not found"; }
            [[ "$has_w035" -ne 1 ]] && { ok=0; msg="${msg:+$msg; }expected warning[W035] not emitted"; }
            ;;
        *)
            ok=0; msg="unknown verdict '$verdict' in EXPECTED.txt"
            ;;
    esac

    if [[ "$ok" -eq 1 ]]; then
        PASS=$((PASS + 1))
        echo "[native-v2-effects] PASS[$relpath]: $verdict ($phase)"
    else
        FAILED=1
        echo "[native-v2-effects] FAIL[$relpath]: $msg ($phase)" >&2
        echo "----- output[$relpath] -----" >&2
        echo "$out" | tail -4 >&2
        echo "----------------------------" >&2
    fi
done < "$EXPECTED"

echo "[native-v2-effects] PASS $PASS/$TOTAL"
echo "---- $PASS/$TOTAL ----"

if [[ "$FAILED" -ne 0 ]]; then
    echo "[native-v2-effects] FAIL: effect-enforcement gate regressed" >&2
    exit 1
fi
exit 0
