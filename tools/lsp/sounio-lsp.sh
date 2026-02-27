#!/usr/bin/env bash
# Sounio LSP Server - JSON-RPC over stdio

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOUNIO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIAG_PARSER="$SCRIPT_DIR/parse_diagnostics.sh"

source "$SOUNIO_ROOT/scripts/lib/resolve_souc.sh"

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required for sounio-lsp" >&2
    exit 1
fi

normalize_bool() {
    local raw="$1"
    case "$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) echo "1" ;;
        0|false|no|off) echo "0" ;;
        *)
            echo "error: invalid boolean value '$raw'" >&2
            return 1
            ;;
    esac
}

verify_pinned_souc() {
    local bin="$1"
    local sha_path="${bin}.sha256"
    local sig_path="${bin}.sig"
    if [[ ! -f "$sha_path" ]]; then
        echo "error: strict no-rust mode requires checksum file: $sha_path" >&2
        return 1
    fi
    if [[ ! -f "$sig_path" ]]; then
        echo "error: strict no-rust mode requires signature file: $sig_path" >&2
        return 1
    fi
    local expected
    expected="$(awk '{print $1}' "$sha_path" | tr -d '[:space:]')"
    if [[ -z "$expected" ]]; then
        echo "error: invalid checksum file: $sha_path" >&2
        return 1
    fi
    local actual
    actual="$(sha256sum "$bin" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "error: strict no-rust mode checksum mismatch for $bin" >&2
        return 1
    fi
}

resolve_souc_for_lsp() {
    local strict="$1"
    local explicit="${SOUNIO_LSP_SOUC_BIN:-}"
    local candidate=""

    if [[ -n "$explicit" ]]; then
        candidate="$explicit"
    else
        if [[ -x "$SOUNIO_ROOT/.pinned-souc/souc-linux-x86_64" ]]; then
            candidate="$SOUNIO_ROOT/.pinned-souc/souc-linux-x86_64"
        elif [[ -x "$SOUNIO_ROOT/artifacts/omega/souc-bin/souc-linux-x86_64" ]]; then
            candidate="$SOUNIO_ROOT/artifacts/omega/souc-bin/souc-linux-x86_64"
        elif [[ -n "${SOUC_BIN:-}" && -x "${SOUC_BIN:-}" ]]; then
            candidate="$SOUC_BIN"
        fi
    fi

    if [[ -z "$candidate" || ! -x "$candidate" ]]; then
        echo "error: souc binary not found/executable" >&2
        return 1
    fi

    if [[ "$strict" == "1" ]]; then
        case "$candidate" in
            "$SOUNIO_ROOT"/.pinned-souc/*|"$SOUNIO_ROOT"/artifacts/omega/souc-bin/*) ;;
            *)
                echo "error: strict no-rust mode requires pinned souc under .pinned-souc/ or artifacts/omega/souc-bin/" >&2
                return 1
                ;;
        esac
        verify_pinned_souc "$candidate" || return 1
    fi

    printf '%s\n' "$candidate"
}

SOUNIO_LSP_STRICT_NO_RUST="$(
    normalize_bool "${SOUNIO_LSP_STRICT_NO_RUST:-${SOUNIO_REPO_HARD_NO_RUST:-1}}"
)"
SOUC_BIN="$(resolve_souc_for_lsp "$SOUNIO_LSP_STRICT_NO_RUST")"

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

path_to_uri() {
    python3 - "$1" <<'PY'
import os
import sys
from urllib.parse import quote

path = os.path.abspath(sys.argv[1])
print("file://" + quote(path))
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
    local parsed
    if ! parsed="$(
        LSP_HOVER_OUTPUT="$output" python3 - "$line" "$col" <<'PY'
import math
import os
import re
import sys

req_line = int(sys.argv[1])
req_col = int(sys.argv[2])
text = os.environ.get("LSP_HOVER_OUTPUT", "")

coord_re = re.compile(r"(?P<l1>\d+):(?P<c1>\d+)(?:-(?P<l2>\d+):(?P<c2>\d+))?")

def extract_type(line: str) -> str:
    lowered = line.lower()
    if "type:" in lowered:
        idx = lowered.rfind("type:")
        return line[idx + len("type:"):].strip()
    if "=>" in line:
        return line.rsplit("=>", 1)[1].strip()
    if "->" in line:
        return line.rsplit("->", 1)[1].strip()
    return ""

best = None
best_score = math.inf

for raw_line in text.splitlines():
    candidate_type = extract_type(raw_line)
    if not candidate_type:
        continue

    match = coord_re.search(raw_line)
    if match:
        l1 = int(match.group("l1"))
        c1 = int(match.group("c1"))
        l2 = int(match.group("l2")) if match.group("l2") else l1
        c2 = int(match.group("c2")) if match.group("c2") else c1
        in_range = (l1 <= req_line <= l2)
        if in_range and req_line == l1 == l2:
            in_range = c1 <= req_col <= c2
        if in_range:
            score = 0
        else:
            score = abs(l1 - req_line) * 1000 + abs(c1 - req_col)
    else:
        score = 9_000_000

    if score < best_score:
        best_score = score
        best = candidate_type

if best is None:
    raise SystemExit(1)

print(best)
PY
    )"; then
        return 1
    fi
    printf '%s' "$parsed"
}

extract_definition_location_json() {
    local output="$1"
    local req_line="$2"
    local req_col="$3"
    local current_file_path="$4"
    local match_json
    if ! match_json="$(
        LSP_DEF_OUTPUT="$output" python3 - "$req_line" "$req_col" "$current_file_path" <<'PY'
import json
import math
import os
import re
import sys

req_line = int(sys.argv[1])
req_col = int(sys.argv[2])
current_file = os.path.abspath(sys.argv[3])
text = os.environ.get("LSP_DEF_OUTPUT", "")

coord_re = re.compile(r"(?:(?P<path>[^\s:]+\.sio):)?(?P<line>\d+):(?P<col>\d+)")

def parse_coord(match):
    path = match.group("path")
    line = int(match.group("line"))
    col = int(match.group("col"))
    if path:
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(os.path.dirname(current_file), path))
    return {"path": path or "", "line": line, "col": col}

def score(src):
    return abs(src["line"] - req_line) * 1000 + abs(src["col"] - req_col)

best = None
best_score = math.inf

for raw_line in text.splitlines():
    matches = list(coord_re.finditer(raw_line))
    if not matches:
        continue

    coords = [parse_coord(m) for m in matches]
    lowered = raw_line.lower()

    if len(coords) >= 2:
        src = coords[0]
        dst = coords[-1]
        dst_path = dst["path"] or src["path"]
        candidate = {"path": dst_path, "line": dst["line"], "col": dst["col"]}
        candidate_score = score(src)
    else:
        if not any(token in lowered for token in ("definition", "defined at", "declared at", "declaration", "def", "->")):
            continue
        dst = coords[0]
        candidate = {"path": dst["path"], "line": dst["line"], "col": dst["col"]}
        candidate_score = 2_000_000 + score(dst)

    if candidate_score < best_score:
        best_score = candidate_score
        best = candidate

if best is None:
    raise SystemExit(1)

print(json.dumps(best))
PY
    )"; then
        return 1
    fi

    local dst_line dst_col dst_path dst_uri
    dst_line="$(jq -r '.line' <<<"$match_json")"
    dst_col="$(jq -r '.col' <<<"$match_json")"
    dst_path="$(jq -r '.path // ""' <<<"$match_json")"
    dst_uri="null"
    if [[ -n "$dst_path" ]]; then
        dst_uri="$(path_to_uri "$dst_path" | jq -R '.')"
    fi

    jq -cn \
        --argjson uri "$dst_uri" \
        --argjson ln "$((dst_line - 1))" \
        --argjson col "$((dst_col - 1))" \
        '{uri:$uri, range:{start:{line:$ln, character:$col}, end:{line:$ln, character:$col}}}'
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

    if location="$(extract_definition_location_json "$output" "$line1" "$col1" "$file_path")"; then
        local location_uri
        location_uri="$(jq -r '.uri // empty' <<<"$location")"
        if [[ -z "$location_uri" ]]; then
            location_uri="$uri"
        fi
        result="$(jq -cn --arg uri "$location_uri" --argjson loc "$location" '$loc | .uri = $uri')"
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
