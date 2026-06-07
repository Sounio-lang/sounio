#!/usr/bin/env bash
# diag_bridge.sh — minimal Sounio compiler-diagnostics → LSP bridge.
#
# Runs the Sounio compiler in check mode on a single .sio file, parses the
# diagnostics it prints, and emits an LSP `textDocument/publishDiagnostics`
# params object on stdout. Pure bash: no jq, no python, no awk — manual JSON
# escaping, so it does not regress the "pure-Sounio, no hybrid" posture of the
# in-tree server (`self-hosted/lsp/server.sio`).
#
# WHY THIS EXISTS (see tools/lsp/LSP_EVALUATION_2026-06-07.md):
#   The in-tree LSP server already runs and already publishes diagnostics with
#   the correct error code + message, but lands every one at line 0:0. The sole
#   defect is its line parser: `parse_diagnostics()` in server.sio scans for the
#   literal " at line N", a format NO current compiler emits. The compiler the
#   server actually execs (mini_native, bin/souc-linux-x86_64) emits
#       error[Exxx]: <message> at <func>:<LINE>
#   and the modular compiler (souc main.sio, --check) emits a byte span
#       error[Exxx] at <start>..<end>: <message>
#   This bridge parses BOTH and emits a correctly-located diagnostic, proving
#   the position IS extractable and demonstrating the one-line server.sio fix.
#
# USAGE:
#   tools/lsp/diag_bridge.sh <file.sio> [souc-binary]
# Default souc binary: ./bin/souc-linux-x86_64 (the one the LSP server execs).

set -euo pipefail

FILE="${1:?usage: diag_bridge.sh <file.sio> [souc-binary]}"
SOUC="${2:-./bin/souc-linux-x86_64}"

if [[ ! -f "$FILE" ]]; then
    echo "diag_bridge: no such file: $FILE" >&2
    exit 2
fi
if [[ ! -x "$SOUC" ]]; then
    echo "diag_bridge: souc binary not executable: $SOUC" >&2
    exit 2
fi

# ---- run the compiler in check mode, capture combined output -------------
# Two compilers, two CLIs:
#   mini_native (bin/souc-linux-x86_64, what the LSP server execs) → `<src> <out>`
#   modular     (souc main.sio)                                    → `--check <src>`
# Probe with no args: mini_native prints "Usage: mini_native ...".
# (Capture first, then grep — a `grep -q` short-circuit under `pipefail`
# SIGPIPEs the producer and would mis-read the probe as a failure.)
PROBE="$(timeout 30 "$SOUC" </dev/null 2>&1 || true)"
if printf '%s' "$PROBE" | grep -q '^Usage: mini_native'; then
    RAW="$("$SOUC" "$FILE" /tmp/diag_bridge_out.elf 2>&1 || true)"
else
    # Modular emits one diagnostic across several lines (print_int appends a
    # newline). Re-join physical lines that belong to one diagnostic so the
    # `error[Exxx] at S..E: msg` record lands on a single logical line.
    RAW="$("$SOUC" --check "$FILE" 2>&1 || true)"
    RAW="$(printf '%s' "$RAW" | tr -d '\n' | sed 's/error\[/\nerror[/g')"
fi

# ---- JSON string escaper (pure bash) -------------------------------------
json_escape() {
    local s="$1" out="" c i
    for (( i=0; i<${#s}; i++ )); do
        c="${s:i:1}"
        case "$c" in
            '"')  out+='\"' ;;
            '\')  out+='\\' ;;
            $'\t') out+='\t' ;;
            $'\n') out+='\n' ;;
            $'\r') out+='\r' ;;
            *)    out+="$c" ;;
        esac
    done
    printf '%s' "$out"
}

# ---- byte offset -> 0-based (line,col), using the source file ------------
# Echoes "LINE COL". Used for the modular byte-span format.
offset_to_linecol() {
    local off="$1" line=0 col=0 i=0 ch content
    content="$(cat "$FILE")"
    while [[ $i -lt $off && $i -lt ${#content} ]]; do
        ch="${content:i:1}"
        if [[ "$ch" == $'\n' ]]; then line=$((line+1)); col=0; else col=$((col+1)); fi
        i=$((i+1))
    done
    printf '%s %s' "$line" "$col"
    return 0
}

# ---- parse diagnostics ----------------------------------------------------
# Accumulate JSON diagnostic objects into DIAGS.
DIAGS=""
emit_diag() {
    local line0="$1" col0="$2" code="$3" msg="$4"
    local jmsg; jmsg="$(json_escape "$msg")"
    local end0=$((col0 + 1))
    local obj
    obj="{\"severity\":1,\"source\":\"sounio:check\",\"code\":\"$code\","
    obj+="\"range\":{\"start\":{\"line\":$line0,\"character\":$col0},"
    obj+="\"end\":{\"line\":$line0,\"character\":$end0}},"
    obj+="\"message\":\"$jmsg\"}"
    if [[ -n "$DIAGS" ]]; then DIAGS+=",$obj"; else DIAGS="$obj"; fi
}

while IFS= read -r ln; do
    # Strip a leading "error: " wrapper the modular driver sometimes adds.
    ln="${ln#error: }"
    case "$ln" in
        error\[E*)
            # Extract the code: error[Exxx]
            code="${ln#error[}"; code="${code%%]*}"

            if [[ "$ln" == *" at <"*":"* ]]; then
                # mini_native form: error[Exxx]: <msg> at <func>:<LINE>
                lineno="${ln##*:}"; lineno="${lineno//[^0-9]/}"
                msg="${ln#*]}"; msg="${msg#:}"; msg="${msg# }"
                msg="${msg% at <*}"
                [[ "$lineno" =~ ^[0-9]+$ ]] || lineno=1
                emit_diag $((lineno - 1)) 0 "$code" "error[$code]: $msg"
            elif [[ "$ln" == *" at "*".."* ]]; then
                # modular form: error[Exxx] at <start>..<end>: <msg>
                span="${ln#*] at }"; span="${span%%:*}"
                start="${span%%..*}"; start="${start//[^0-9]/}"
                [[ "$start" =~ ^[0-9]+$ ]] || start=0
                msg="${ln#*: }"
                msg="${msg%%   *}"   # cut at the "   |" help separator
                LC="$(offset_to_linecol "$start")"
                L="${LC%% *}"; C="${LC##* }"
                emit_diag "$L" "$C" "$code" "error[$code]: $msg"
            else
                # code but no position info we recognise — land at file head.
                msg="${ln#*]}"; msg="${msg#:}"; msg="${msg# }"
                emit_diag 0 0 "$code" "error[$code]: $msg"
            fi
            ;;
    esac
done < <(printf '%s\n' "$RAW")

# ---- emit publishDiagnostics params --------------------------------------
URI="file://$(cd "$(dirname "$FILE")" && pwd)/$(basename "$FILE")"
printf '{"uri":"%s","diagnostics":[%s]}\n' "$URI" "$DIAGS"
