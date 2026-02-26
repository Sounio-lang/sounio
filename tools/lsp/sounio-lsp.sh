#!/usr/bin/env bash
# Sounio LSP Server - JSON-RPC over stdio

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOUNIO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIAG_PARSER="$SCRIPT_DIR/parse_diagnostics.sh"

if [[ -x "$SOUNIO_ROOT/.pinned-souc/souc-linux-x86_64" ]]; then
    SOUC_BIN="$SOUNIO_ROOT/.pinned-souc/souc-linux-x86_64"
elif [[ -x "$SOUNIO_ROOT/artifacts/omega/souc-bin/souc-linux-x86_64" ]]; then
    SOUC_BIN="$SOUNIO_ROOT/artifacts/omega/souc-bin/souc-linux-x86_64"
fi

source "$SOUNIO_ROOT/scripts/lib/resolve_souc.sh"

if [[ -z "${SOUC_BIN:-}" ]] || [[ ! -x "${SOUC_BIN:-}" ]]; then
    echo "error: souc binary not found/executable" >&2
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required for sounio-lsp" >&2
    exit 1
fi

log() {
    printf '[sounio-lsp] %s\n' "$*" >&2
}

declare -A OPEN_DOCS
declare -A OPEN_DOC_VERSION

CHECK_PID=""
RUNNING=1
SHUTDOWN_SEEN=0
EXIT_CODE=0

cleanup() {
    if [[ -n "$CHECK_PID" ]] && kill -0 "$CHECK_PID" 2>/dev/null; then
        kill "$CHECK_PID" 2>/dev/null || true
        wait "$CHECK_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT SIGTERM SIGINT SIGPIPE

json_length() {
    LC_ALL=C printf '%s' "$1" | wc -c | tr -d ' '
}

send_wire_message() {
    local payload="$1"
    local length
    length="$(json_length "$payload")"
    printf 'Content-Length: %s\r\n\r\n%s' "$length" "$payload"
}

send_response() {
    local id_json="$1"
    local result_json="$2"
    local payload
    payload="$(jq -cn --argjson id "$id_json" --argjson result "$result_json" \
        '{jsonrpc:"2.0", id:$id, result:$result}')"
    send_wire_message "$payload"
}

send_notification() {
    local method="$1"
    local params_json="$2"
    local payload
    payload="$(jq -cn --arg method "$method" --argjson params "$params_json" \
        '{jsonrpc:"2.0", method:$method, params:$params}')"
    send_wire_message "$payload"
}

send_error() {
    local id_json="$1"
    local code="$2"
    local message="$3"
    local payload
    payload="$(jq -cn --argjson id "$id_json" --argjson code "$code" --arg message "$message" \
        '{jsonrpc:"2.0", id:$id, error:{code:$code, message:$message}}')"
    send_wire_message "$payload"
}

uri_to_path() {
    python3 - "$1" <<'PY'
import sys
from urllib.parse import unquote, urlparse

uri = sys.argv[1]
if uri.startswith("file://"):
    parsed = urlparse(uri)
    print(unquote(parsed.path))
else:
    print(uri)
PY
}

kill_stale_check() {
    if [[ -n "$CHECK_PID" ]] && kill -0 "$CHECK_PID" 2>/dev/null; then
        log "killing stale check pid=$CHECK_PID"
        kill "$CHECK_PID" 2>/dev/null || true
        wait "$CHECK_PID" 2>/dev/null || true
        CHECK_PID=""
    fi
}

read_message() {
    local first_line=""
    if ! IFS= read -r first_line; then
        return 1
    fi
    first_line="${first_line%$'\r'}"

    if [[ -z "$first_line" ]]; then
        read_message
        return $?
    fi

    if [[ "$first_line" == \{* ]]; then
        printf '%s' "$first_line"
        return 0
    fi

    local content_length=""
    local line="$first_line"
    while true; do
        line="${line%$'\r'}"
        if [[ "$line" =~ ^[Cc]ontent-[Ll]ength:[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
            content_length="${BASH_REMATCH[1]}"
        fi
        if [[ -z "$line" ]]; then
            break
        fi
        if ! IFS= read -r line; then
            return 1
        fi
    done

    if [[ -z "$content_length" ]]; then
        return 1
    fi

    dd bs=1 count="$content_length" status=none 2>/dev/null || return 1
}

extract_hover_type() {
    local output="$1"
    local line="$2"
    local col="$3"

    local best=""
    best="$(printf '%s\n' "$output" \
        | awk -v ln="$line" -v col="$col" '
            {
                if ($0 ~ ":" ln ":" col) {
                    print $0
                    exit
                }
            }')"
    if [[ -z "$best" ]]; then
        best="$(printf '%s\n' "$output" \
            | awk -v ln="$line" '
                {
                    if ($0 ~ ":" ln ":") {
                        print $0
                        exit
                    }
                }')"
    fi
    if [[ -z "$best" ]]; then
        return 1
    fi

    if [[ "$best" == *"type:"* ]]; then
        printf '%s' "${best##*type: }"
        return 0
    fi
    if [[ "$best" == *"=>"* ]]; then
        printf '%s' "${best##*=> }"
        return 0
    fi
    printf '%s' "$best"
}

extract_definition_location_json() {
    local output="$1"
    local req_line="$2"
    local req_col="$3"
    local out
    out="$(printf '%s\n' "$output" \
        | awk -v ln="$req_line" -v col="$req_col" '
            $0 ~ ":" ln ":" col {
                if (match($0, /([0-9]+):([0-9]+)[^0-9]*$/ , m)) {
                    print m[1] ":" m[2]
                    exit
                }
            }')"
    if [[ -z "$out" ]]; then
        return 1
    fi

    local dst_line="${out%%:*}"
    local dst_col="${out##*:}"
    jq -cn --argjson ln "$((dst_line - 1))" --argjson col "$((dst_col - 1))" \
        '{uri:null, range:{start:{line:$ln, character:$col}, end:{line:$ln, character:$col}}}'
}

run_check_and_publish() {
    local uri="$1"
    local file_path
    file_path="$(uri_to_path "$uri")"

    if [[ ! -f "$file_path" ]]; then
        log "skip diagnostics (file missing): $file_path"
        return 0
    fi

    kill_stale_check

    local out_file err_file diagnostics rc
    out_file="$(mktemp)"
    err_file="$(mktemp)"
    rc=0

    (
        if command -v timeout >/dev/null 2>&1; then
            timeout 60s "$SOUC_BIN" check "$file_path" >"$out_file" 2>"$err_file"
        else
            "$SOUC_BIN" check "$file_path" >"$out_file" 2>"$err_file"
        fi
    ) &
    CHECK_PID="$!"
    if ! wait "$CHECK_PID"; then
        rc=$?
    fi
    CHECK_PID=""

    if diagnostics="$($DIAG_PARSER <"$err_file" 2>/dev/null)"; then
        :
    else
        diagnostics='[]'
    fi

    local params
    params="$(jq -cn --arg uri "$uri" --argjson diagnostics "$diagnostics" \
        '{uri:$uri, diagnostics:$diagnostics}')"
    send_notification "textDocument/publishDiagnostics" "$params"

    rm -f "$out_file" "$err_file"
    return "$rc"
}

handle_initialize() {
    local id_json="$1"
    local result
    result="$(jq -cn '{
        capabilities: {
            textDocumentSync: 1,
            hoverProvider: true,
            definitionProvider: true
        },
        serverInfo: { name: "sounio-lsp", version: "0.2.0" }
    }')"
    send_response "$id_json" "$result"
}

handle_shutdown() {
    local id_json="$1"
    SHUTDOWN_SEEN=1
    send_response "$id_json" "null"
}

handle_did_open() {
    local params="$1"
    local uri version text
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    version="$(jq -r '.textDocument.version // 0' <<<"$params")"
    text="$(jq -r '.textDocument.text // ""' <<<"$params")"

    OPEN_DOCS["$uri"]="$text"
    OPEN_DOC_VERSION["$uri"]="$version"
    run_check_and_publish "$uri" || true
}

handle_did_change() {
    local params="$1"
    local uri version new_text
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    version="$(jq -r '.textDocument.version // 0' <<<"$params")"
    new_text="$(jq -r '.contentChanges[-1].text // ""' <<<"$params")"
    OPEN_DOCS["$uri"]="$new_text"
    OPEN_DOC_VERSION["$uri"]="$version"
}

handle_did_save() {
    local params="$1"
    local uri
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    run_check_and_publish "$uri" || true
}

handle_did_close() {
    local params="$1"
    local uri
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    unset OPEN_DOCS["$uri"]
    unset OPEN_DOC_VERSION["$uri"]
    local clear
    clear="$(jq -cn --arg uri "$uri" '{uri:$uri, diagnostics:[]}')"
    send_notification "textDocument/publishDiagnostics" "$clear"
}

handle_hover() {
    local id_json="$1"
    local params="$2"
    local uri file_path line col line1 col1 output ty result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    file_path="$(uri_to_path "$uri")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"
    line1="$((line + 1))"
    col1="$((col + 1))"

    if [[ ! -f "$file_path" ]]; then
        send_response "$id_json" "null"
        return
    fi

    output="$($SOUC_BIN check "$file_path" --show-types 2>&1 || true)"
    if ! ty="$(extract_hover_type "$output" "$line1" "$col1")"; then
        send_response "$id_json" "null"
        return
    fi

    result="$(jq -cn --arg ty "$ty" '{
        contents: {
            kind: "markdown",
            value: ("```sounio\\n" + $ty + "\\n```")
        }
    }')"
    send_response "$id_json" "$result"
}

handle_definition() {
    local id_json="$1"
    local params="$2"
    local uri file_path line col line1 col1 output result location

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    file_path="$(uri_to_path "$uri")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"
    line1="$((line + 1))"
    col1="$((col + 1))"

    if [[ ! -f "$file_path" ]]; then
        send_response "$id_json" "null"
        return
    fi

    output="$($SOUC_BIN check "$file_path" --show-defs 2>&1 || true)"
    if [[ "$output" == *"unrecognized option"* ]] || [[ "$output" == *"unknown option"* ]]; then
        output="$($SOUC_BIN check "$file_path" --show-ast 2>&1 || true)"
    fi

    if location="$(extract_definition_location_json "$output" "$line1" "$col1")"; then
        result="$(jq -cn --arg uri "$uri" --argjson loc "$location" '$loc | .uri = $uri')"
        send_response "$id_json" "$result"
        return
    fi

    send_response "$id_json" "null"
}

dispatch_message() {
    local message="$1"
    local method id_json params

    method="$(jq -r '.method // ""' <<<"$message")"
    id_json="$(jq -c '.id // empty' <<<"$message")"
    params="$(jq -c '.params // {}' <<<"$message")"

    if [[ -z "$method" ]]; then
        return 0
    fi

    log "method=$method"
    case "$method" in
        initialize)
            if [[ -n "$id_json" ]]; then
                handle_initialize "$id_json"
            fi
            ;;
        initialized)
            ;;
        shutdown)
            if [[ -n "$id_json" ]]; then
                handle_shutdown "$id_json"
            fi
            ;;
        exit)
            RUNNING=0
            if [[ "$SHUTDOWN_SEEN" -eq 0 ]]; then
                EXIT_CODE=1
            fi
            ;;
        textDocument/didOpen)
            handle_did_open "$params"
            ;;
        textDocument/didChange)
            handle_did_change "$params"
            ;;
        textDocument/didSave)
            handle_did_save "$params"
            ;;
        textDocument/didClose)
            handle_did_close "$params"
            ;;
        textDocument/hover)
            if [[ -n "$id_json" ]]; then
                handle_hover "$id_json" "$params"
            fi
            ;;
        textDocument/definition)
            if [[ -n "$id_json" ]]; then
                handle_definition "$id_json" "$params"
            fi
            ;;
        *)
            if [[ -n "$id_json" ]]; then
                send_error "$id_json" -32601 "Method not found: $method"
            fi
            ;;
    esac
}

main() {
    log "startup souc=$SOUC_BIN"
    while [[ "$RUNNING" -eq 1 ]]; do
        local msg
        if ! msg="$(read_message)"; then
            break
        fi
        if ! jq -e . >/dev/null 2>&1 <<<"$msg"; then
            log "invalid JSON payload; skipping"
            continue
        fi
        dispatch_message "$msg"
    done
    log "shutdown code=$EXIT_CODE"
    exit "$EXIT_CODE"
}

main "$@"
