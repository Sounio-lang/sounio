#!/usr/bin/env bash
# tests/parity/run_parity.sh
#
# Parity / golden-regression harness for the Sounio compiler.
#
# Background (see docs/compiler/KNOWN_LIMITATIONS.md, section
# "Single-source build path"): bin/souc is built today from
# self-hosted/compiler/lean_single.sio; the modular tree at
# self-hosted/{lexer,parser,check,ir,native} is a future target
# kept in lock-step by discipline.  This harness exists to catch
# silent divergence between the two builds when the second path
# comes online, and to catch semantic regressions in lean_single.sio
# in the meantime via captured golden outputs.
#
# MODES
#   --capture            Run $SOUC_A on every manifest entry, write
#                        stdout/stderr/rc to golden/<name>.{stdout,stderr,rc}.
#                        Use once to refresh the golden set after an
#                        intentional, reviewed change in semantics.
#   --check              (default) Run $SOUC_A, diff against golden/.
#                        CI-friendly: nonzero exit if any diff or any
#                        behaviour mismatch.
#   --parity PATH        Run $SOUC_A AND $SOUC_B=PATH on every entry,
#                        diff their stdout/stderr/rc against each other
#                        (not against golden).  Nonzero exit on any
#                        divergence.  Used when PATH is the alternate
#                        (modular-built) compiler binary.
#
# ENVIRONMENT
#   SOUC_A   Path to the primary compiler launcher.
#            Default: ./bin/souc
#   SOUC_B   Path to the secondary compiler (only for --parity).
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="check"
SOUC_A="${SOUC_A:-$ROOT/bin/souc}"
SOUC_B=""
MANIFEST="$ROOT/tests/parity/MANIFEST.txt"
GOLDEN_DIR="$ROOT/tests/parity/golden"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --capture) MODE="capture"; shift ;;
        --check)   MODE="check";   shift ;;
        --parity)  MODE="parity"; SOUC_B="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,35p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -x "$SOUC_A" ]]; then
    echo "SOUC_A not executable: $SOUC_A" >&2
    exit 2
fi
if [[ "$MODE" == "parity" && ! -x "$SOUC_B" ]]; then
    echo "SOUC_B not executable: $SOUC_B" >&2
    exit 2
fi

mkdir -p "$GOLDEN_DIR"

# run_one <souc-bin> <mode> <path>  -> writes $TMP/{stdout,stderr,rc}
run_one() {
    local souc="$1" action="$2" src="$3" tmp="$4"
    if [[ "$action" == "run" ]]; then
        "$souc" run "$src" >"$tmp/stdout" 2>"$tmp/stderr" || true
        echo $? >"$tmp/rc"
    else
        "$souc" check "$src" >"$tmp/stdout" 2>"$tmp/stderr" || true
        echo $? >"$tmp/rc"
    fi
}

# name_of <path>  -> slug used for golden files
name_of() {
    basename "$1" .sio
}

pass=0; fail=0; total=0
failures=()

while IFS= read -r raw; do
    line="${raw%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line// }" ]] && continue
    # shellcheck disable=SC2206
    tokens=($line)
    mode="${tokens[0]}"
    src="${tokens[1]}"
    marker="${tokens[2]:-}"
    name="$(name_of "$src")"
    total=$((total+1))
    tmpA="$(mktemp -d)"
    run_one "$SOUC_A" "$mode" "$src" "$tmpA"
    rcA="$(cat "$tmpA/rc")"

    # Per-mode expectation.
    # NOTE: the current bin/souc launcher swallows the compiler's exit
    # code for the 'check' command (always returns 0 even when the
    # self-hosted binary reported "typecheck: failed").  Until that is
    # fixed, for 'fail' mode we accept a program as correctly failing
    # if its output contains "typecheck: failed" OR the expected error
    # marker.  For 'run' mode we still require rc == 0 because 'run'
    # correctly propagates the compiled-and-executed program's rc.
    expected_rc_ok=0
    if [[ "$mode" == "run" && "$rcA" == "0" ]]; then expected_rc_ok=1; fi
    if [[ "$mode" == "fail" ]]; then
        if [[ "$rcA" != "0" ]] \
           || grep -qF "typecheck: failed" "$tmpA/stdout" \
           || grep -qF "typecheck: failed" "$tmpA/stderr"; then
            expected_rc_ok=1
        fi
    fi

    marker_ok=1
    if [[ -n "$marker" ]]; then
        if ! grep -qF "$marker" "$tmpA/stdout" && ! grep -qF "$marker" "$tmpA/stderr"; then
            marker_ok=0
        fi
    fi

    case "$MODE" in
        capture)
            cp "$tmpA/stdout" "$GOLDEN_DIR/$name.stdout"
            cp "$tmpA/stderr" "$GOLDEN_DIR/$name.stderr"
            cp "$tmpA/rc"     "$GOLDEN_DIR/$name.rc"
            if (( expected_rc_ok )) && (( marker_ok )); then
                echo "CAPTURE $name ok"
                pass=$((pass+1))
            else
                echo "CAPTURE $name warn (rc=$rcA marker=$marker_ok)"
                fail=$((fail+1)); failures+=("$name")
            fi
            ;;
        check)
            ok=1
            if (( ! expected_rc_ok )); then ok=0; fi
            if (( ! marker_ok )); then ok=0; fi
            for s in stdout stderr rc; do
                g="$GOLDEN_DIR/$name.$s"
                if [[ -f "$g" ]]; then
                    if ! diff -q "$g" "$tmpA/$s" >/dev/null 2>&1; then
                        ok=0
                    fi
                fi
            done
            if (( ok )); then
                echo "CHECK   $name ok"
                pass=$((pass+1))
            else
                echo "CHECK   $name FAIL (rc=$rcA marker_ok=$marker_ok)"
                fail=$((fail+1)); failures+=("$name")
            fi
            ;;
        parity)
            tmpB="$(mktemp -d)"
            run_one "$SOUC_B" "$mode" "$src" "$tmpB"
            rcB="$(cat "$tmpB/rc")"
            divergent=0
            for s in stdout stderr rc; do
                if ! diff -q "$tmpA/$s" "$tmpB/$s" >/dev/null 2>&1; then
                    divergent=1
                fi
            done
            if (( ! divergent )) && (( expected_rc_ok )) && (( marker_ok )); then
                echo "PARITY  $name ok"
                pass=$((pass+1))
            else
                echo "PARITY  $name FAIL (A rc=$rcA B rc=$rcB divergent=$divergent marker_ok=$marker_ok)"
                fail=$((fail+1)); failures+=("$name")
            fi
            rm -rf "$tmpB"
            ;;
    esac
    rm -rf "$tmpA"
done < "$MANIFEST"

echo "---"
echo "total=$total pass=$pass fail=$fail mode=$MODE"
if (( fail > 0 )); then
    echo "failures:"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
exit 0
