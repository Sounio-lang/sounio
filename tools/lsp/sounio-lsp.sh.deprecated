#!/usr/bin/env bash
# Sounio LSP Server - JSON-RPC over stdio
# BIG LSP Edition - Full IDE features

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

normalize_positive_int() {
    local raw="$1"
    local name="$2"
    if ! [[ "$raw" =~ ^[0-9]+$ ]] || [[ "$raw" == "0" ]]; then
        echo "error: $name must be a positive integer (got '$raw')" >&2
        return 1
    fi
    printf '%s\n' "$raw"
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

is_ci_native_wrapper() {
    local candidate="$1"
    [[ "$candidate" == "$SOUNIO_ROOT/scripts/ci/souc-native-wrapper.sh" ]] \
        && [[ -n "${SOUC_NATIVE_BIN:-}" ]] \
        && [[ -x "${SOUC_NATIVE_BIN:-}" ]]
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
            "$SOUNIO_ROOT"/scripts/ci/souc-native-wrapper.sh)
                if ! is_ci_native_wrapper "$candidate"; then
                    echo "error: strict no-rust mode requires executable SOUC_NATIVE_BIN for native wrapper" >&2
                    return 1
                fi
                ;;
            *)
                echo "error: strict no-rust mode requires pinned souc under .pinned-souc/ or artifacts/omega/souc-bin/" >&2
                return 1
                ;;
        esac
        if ! is_ci_native_wrapper "$candidate"; then
            verify_pinned_souc "$candidate" || return 1
        fi
    fi

    printf '%s\n' "$candidate"
}

SOUNIO_LSP_STRICT_NO_RUST="$(
    normalize_bool "${SOUNIO_LSP_STRICT_NO_RUST:-${SOUNIO_REPO_HARD_NO_RUST:-1}}"
)"
SOUNIO_LSP_CHECK_TIMEOUT_SEC="$(
    normalize_positive_int "${SOUNIO_LSP_CHECK_TIMEOUT_SEC:-60}" "SOUNIO_LSP_CHECK_TIMEOUT_SEC"
)"
SOUC_BIN="$(resolve_souc_for_lsp "$SOUNIO_LSP_STRICT_NO_RUST")"

log() {
    printf '[sounio-lsp] %s\n' "$*" >&2
}

declare -A OPEN_DOCS
declare -A OPEN_DOC_VERSION
declare -A OPEN_DOC_CHECK_TOKEN

CHECK_PID=""
CHECK_PID_URI=""
RUNNING=1
SHUTDOWN_SEEN=0
EXIT_CODE=0
LAST_SOURCE_CLEANUP=""

cleanup() {
    if [[ -n "$CHECK_PID" ]] && kill -0 "$CHECK_PID" 2>/dev/null; then
        kill "$CHECK_PID" 2>/dev/null || true
        wait "$CHECK_PID" 2>/dev/null || true
    fi
    CHECK_PID=""
    CHECK_PID_URI=""
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

run_souc_check() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "${SOUNIO_LSP_CHECK_TIMEOUT_SEC}s" "$SOUC_BIN" check "$@"
    else
        "$SOUC_BIN" check "$@"
    fi
}

run_souc_fmt() {
    "$SOUC_BIN" fmt "$@" 2>&1 || true
}

run_souc_fix() {
    "$SOUC_BIN" fix --dry-run "$@" 2>&1 || true
}

build_failed_check_diagnostics() {
    local rc="$1"
    local err_file="$2"
    local code="SOUNIO_LSP_CHECK_FAILED"
    local message="souc check failed with exit code $rc"
    local first_line
    first_line="$(awk 'NF {print; exit}' "$err_file" | tr -d '\r')"

    if [[ "$rc" -eq 124 ]]; then
        code="SOUNIO_LSP_CHECK_TIMEOUT"
        message="souc check timed out after ${SOUNIO_LSP_CHECK_TIMEOUT_SEC}s"
    fi

    if [[ -n "$first_line" ]]; then
        message="$message: $first_line"
    fi

    jq -cn --arg code "$code" --arg message "$message" '[
        {
            range: {
                start: {line: 0, character: 0},
                end: {line: 0, character: 1}
            },
            severity: 1,
            code: $code,
            source: "sounio-lsp",
            message: $message
        }
    ]'
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

cleanup_temp_source() {
    local path="$1"
    if [[ -n "$path" ]] && [[ -f "$path" ]]; then
        rm -f "$path"
    fi
}

write_open_doc_snapshot() {
    local uri="$1"
    local canonical_path="$2"
    local dir snapshot
    dir="$(dirname "$canonical_path")"
    if [[ -d "$dir" ]] && [[ -w "$dir" ]]; then
        snapshot="$(mktemp "$dir/.sounio-lsp-snapshot.XXXXXX.sio")"
    else
        snapshot="$(mktemp /tmp/.sounio-lsp-snapshot.XXXXXX.sio)"
    fi
    printf '%s' "${OPEN_DOCS[$uri]}" >"$snapshot"
    printf '%s\n' "$snapshot"
}

resolve_source_path_for_uri() {
    local uri="$1"
    local canonical_path
    LAST_SOURCE_CLEANUP=""
    canonical_path="$(uri_to_path "$uri")"
    if [[ -n "${OPEN_DOCS[$uri]+x}" ]]; then
        local snapshot
        snapshot="$(write_open_doc_snapshot "$uri" "$canonical_path")" || return 1
        LAST_SOURCE_CLEANUP="$snapshot"
        printf '%s\n' "$snapshot"
        return 0
    fi

    if [[ -f "$canonical_path" ]]; then
        printf '%s\n' "$canonical_path"
        return 0
    fi

    return 1
}

kill_stale_check() {
    local uri="$1"
    if [[ -n "$CHECK_PID" ]] && kill -0 "$CHECK_PID" 2>/dev/null; then
        if [[ "$CHECK_PID_URI" == "$uri" ]]; then
            log "killing stale check pid=$CHECK_PID uri=$uri"
            kill "$CHECK_PID" 2>/dev/null || true
            wait "$CHECK_PID" 2>/dev/null || true
            CHECK_PID=""
            CHECK_PID_URI=""
        else
            log "active check belongs to another uri; keeping pid=$CHECK_PID uri=$CHECK_PID_URI"
        fi
    fi
}

next_check_token() {
    local uri="$1"
    local current="${OPEN_DOC_CHECK_TOKEN[$uri]:-0}"
    if ! [[ "$current" =~ ^[0-9]+$ ]]; then
        current=0
    fi
    local next=$((current + 1))
    OPEN_DOC_CHECK_TOKEN["$uri"]="$next"
    printf '%s\n' "$next"
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

# =============================================================================
# DOCUMENT SYMBOL EXTRACTION
# =============================================================================

extract_document_symbols() {
    local source_path="$1"
    local uri="$2"
    local content output
    
    if [[ -f "$source_path" ]]; then
        content="$(cat "$source_path")"
    else
        content=""
    fi
    
    # Parse document symbols from AST output
    output="$(run_souc_check "$source_path" --show-ast 2>&1 || true)"
    
    python3 - "$uri" "$output" "$content" <<'PY'
import json
import os
import re
import sys

uri = sys.argv[1]
ast_output = sys.argv[2]
content = sys.argv[3] if len(sys.argv) > 3 else ""

symbols = []

# Extract function definitions
func_pattern = re.compile(r'(fn|func|function|def)\s+(\w+)')
for match in func_pattern.finditer(content):
    name = match.group(2)
    line = content[:match.start()].count('\n')
    col = match.start() - content.rfind('\n', 0, match.start()) - 1
    symbols.append({
        "name": name,
        "kind": 12,  # Function
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        },
        "selectionRange": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        }
    })

# Extract type definitions
type_pattern = re.compile(r'(type|struct|enum|interface|trait)\s+(\w+)')
for match in type_pattern.finditer(content):
    name = match.group(2)
    line = content[:match.start()].count('\n')
    col = match.start() - content.rfind('\n', 0, match.start()) - 1
    kind = 23 if match.group(1) in ('struct',) else 10 if match.group(1) == 'enum' else 11 if match.group(1) == 'interface' else 8 if match.group(1) == 'trait' else 22
    symbols.append({
        "name": name,
        "kind": kind,
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        },
        "selectionRange": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        }
    })

# Extract module declarations
mod_pattern = re.compile(r'mod\s+(\w+)')
for match in mod_pattern.finditer(content):
    name = match.group(1)
    line = content[:match.start()].count('\n')
    col = match.start() - content.rfind('\n', 0, match.start()) - 1
    symbols.append({
        "name": name,
        "kind": 2,  # Module
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        },
        "selectionRange": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        }
    })

# Extract const/let bindings
const_pattern = re.compile(r'(const|let)\s+(\w+)')
for match in const_pattern.finditer(content):
    name = match.group(2)
    line = content[:match.start()].count('\n')
    col = match.start() - content.rfind('\n', 0, match.start()) - 1
    kind = 14 if match.group(1) == 'const' else 13  # Constant / Variable
    symbols.append({
        "name": name,
        "kind": kind,
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        },
        "selectionRange": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(name)}
        }
    })

print(json.dumps(symbols))
PY
}

# =============================================================================
# COMPLETION PROVIDER
# =============================================================================

get_completions() {
    local source_path="$1"
    local line="$2"
    local col="$3"
    local content="${4:-}"
    
    python3 - "$line" "$col" "$content" "$source_path" <<'PY'
import json
import os
import re
import sys

req_line = int(sys.argv[1])
req_col = int(sys.argv[2])
content = sys.argv[3] if len(sys.argv) > 3 else ""
source_path = sys.argv[4] if len(sys.argv) > 4 else ""

completions = []

# Sounio keywords
keywords = [
    "fn", "func", "function", "def",
    "let", "const", "var", "mut",
    "if", "else", "elif", "then",
    "match", "case", "when",
    "for", "while", "loop", "break", "continue",
    "return", "yield",
    "struct", "enum", "type", "trait", "impl", "interface",
    "mod", "use", "import", "export", "pub", "public", "private",
    "async", "await",
    "try", "catch", "throw", "error",
    "epistemic", "uncertain", "confidence", "provenance",
    "ontology", "unit", "dimension",
    "unsafe", "safe",
    "where", "with", "in", "is", "as",
    "true", "false", "nil", "none", "null",
    "Self", "self"
]

for kw in keywords:
    completions.append({
        "label": kw,
        "kind": 14,  # Keyword
        "insertText": kw,
        "detail": "keyword",
        "sortText": "1_" + kw
    })

# Built-in types
types = [
    "Int", "Float", "Bool", "String", "Char",
    "List", "Array", "Map", "Set", "Option", "Result",
    "Vec", "HashMap", "HashSet",
    "Tensor", "Distribution",
    "Unit", "Quantity"
]

for ty in types:
    completions.append({
        "label": ty,
        "kind": 7,  # Type
        "insertText": ty,
        "detail": "built-in type",
        "sortText": "2_" + ty
    })

# Built-in functions
builtins = [
    ("print", "print(${1:value})", "Print a value"),
    ("println", "println(${1:value})", "Print a value with newline"),
    ("assert", "assert(${1:condition})", "Assert condition is true"),
    ("panic", "panic(${1:message})", "Panic with message"),
    ("Some", "Some(${1:value})", "Create Some variant"),
    ("None", "None", "None variant"),
    ("Ok", "Ok(${1:value})", "Create Ok variant"),
    ("Err", "Err(${1:error})", "Create Err variant"),
    ("range", "range(${1:start}, ${2:end})", "Create range"),
    ("len", "len(${1:collection})", "Get length"),
    ("map", "map(${1:fn}, ${2:collection})", "Map function over collection"),
    ("filter", "filter(${1:fn}, ${2:collection})", "Filter collection"),
    ("fold", "fold(${1:fn}, ${2:init}, ${3:collection})", "Fold over collection"),
    ("uncertain", "uncertain(${1:value}, ${2:std})", "Create uncertain value"),
    ("confidence", "confidence(${1:value})", "Get confidence interval"),
    ("provenance", "provenance(${1:value})", "Get provenance info"),
]

for name, insert, detail in builtins:
    completions.append({
        "label": name,
        "kind": 3,  # Function
        "insertText": insert,
        "insertTextFormat": 2,  # Snippet
        "detail": detail,
        "sortText": "3_" + name
    })

# Extract symbols from current document
if content:
    # Functions
    func_pattern = re.compile(r'(?:fn|func|function|def)\s+(\w+)')
    for match in func_pattern.finditer(content):
        name = match.group(1)
        completions.append({
            "label": name,
            "kind": 3,  # Function
            "insertText": name + "()",
            "detail": "function (current file)",
            "sortText": "0_" + name
        })
    
    # Variables
    var_pattern = re.compile(r'(?:let|const|var)\s+(\w+)')
    for match in var_pattern.finditer(content):
        name = match.group(1)
        completions.append({
            "label": name,
            "kind": 6,  # Variable
            "insertText": name,
            "detail": "variable (current file)",
            "sortText": "0_" + name
        })

print(json.dumps(completions))
PY
}

# =============================================================================
# CODE ACTIONS
# =============================================================================

get_code_actions() {
    local source_path="$1"
    local diagnostics="$2"
    
    python3 - "$source_path" "$diagnostics" <<'PY'
import json
import os
import re
import sys

source_path = sys.argv[1]
diagnostics_json = sys.argv[2] if len(sys.argv) > 2 else "[]"

try:
    diagnostics = json.loads(diagnostics_json)
except:
    diagnostics = []

actions = []

if os.path.exists(source_path):
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Suggest removing unused imports
    unused_import = re.compile(r'(?:use|import)\s+(\w+)')
    for match in unused_import.finditer(content):
        line = content[:match.start()].count('\n')
        # Add quick fix for unused import
        actions.append({
            "title": f"Remove unused import",
            "kind": "quickfix",
            "edit": {
                "changes": {
                    source_path: [
                        {
                            "range": {
                                "start": {"line": line, "character": 0},
                                "end": {"line": line + 1, "character": 0}
                            },
                            "newText": ""
                        }
                    ]
                }
            }
        })
    
    # Suggest adding type annotation
    untyped_let = re.compile(r'let\s+(\w+)\s*=\s*([^;]+);')
    for match in untyped_let.finditer(content):
        name = match.group(1)
        value = match.group(2)
        line = content[:match.start()].count('\n')
        start_col = match.start() - content.rfind('\n', 0, match.start()) - 1
        end_col = match.end() - content.rfind('\n', 0, match.end()) - 1
        
        actions.append({
            "title": f"Add explicit type annotation",
            "kind": "quickfix",
            "edit": {
                "changes": {
                    source_path: [
                        {
                            "range": {
                                "start": {"line": line, "character": start_col},
                                "end": {"line": line, "character": end_col}
                            },
                            "newText": f"let {name}: /* type */ = {value};"
                        }
                    ]
                }
            }
        })

# Always add organize imports
actions.append({
    "title": "Organize imports",
    "kind": "source.organizeImports",
    "edit": {
        "changes": {}
    }
})

# Add format document
actions.append({
    "title": "Format document",
    "kind": "source.formatDocument",
    "edit": {
        "changes": {}
    }
})

print(json.dumps(actions))
PY
}

# =============================================================================
# REFERENCE FINDER
# =============================================================================

find_references() {
    local source_path="$1"
    local line="$2"
    local col="$3"
    local include_decl="$4"
    
    python3 - "$source_path" "$line" "$col" "$include_decl" <<'PY'
import json
import os
import re
import sys

source_path = sys.argv[1]
req_line = int(sys.argv[2])
req_col = int(sys.argv[3])
include_decl = sys.argv[4] == "true" if len(sys.argv) > 4 else True

references = []

if os.path.exists(source_path):
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Find the word at position
    lines = content.split('\n')
    if 0 <= req_line < len(lines):
        line = lines[req_line]
        # Extract word at position
        word_pattern = re.compile(r'\w+')
        target_word = None
        for match in word_pattern.finditer(line):
            if match.start() <= req_col <= match.end():
                target_word = match.group()
                break
        
        if target_word:
            # Find all occurrences
            for i, l in enumerate(lines):
                for match in word_pattern.finditer(l):
                    if match.group() == target_word:
                        references.append({
                            "uri": source_path,
                            "range": {
                                "start": {"line": i, "character": match.start()},
                                "end": {"line": i, "character": match.end()}
                            }
                        })

print(json.dumps(references))
PY
}

# =============================================================================
# RENAME PROVIDER
# =============================================================================

get_rename_edits() {
    local source_path="$1"
    local line="$2"
    local col="$3"
    local new_name="$4"
    
    python3 - "$source_path" "$line" "$col" "$new_name" <<'PY'
import json
import os
import re
import sys

source_path = sys.argv[1]
req_line = int(sys.argv[2])
req_col = int(sys.argv[3])
new_name = sys.argv[4]

changes = {}

if os.path.exists(source_path):
    with open(source_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    if 0 <= req_line < len(lines):
        line = lines[req_line]
        # Extract word at position
        word_pattern = re.compile(r'\w+')
        target_word = None
        for match in word_pattern.finditer(line):
            if match.start() <= req_col <= match.end():
                target_word = match.group()
                break
        
        if target_word:
            uri = source_path
            if not uri.startswith("file://"):
                uri = "file://" + os.path.abspath(uri)
            
            edits = []
            for i, l in enumerate(lines):
                for match in word_pattern.finditer(l):
                    if match.group() == target_word:
                        edits.append({
                            "range": {
                                "start": {"line": i, "character": match.start()},
                                "end": {"line": i, "character": match.end()}
                            },
                            "newText": new_name
                        })
            
            changes[uri] = edits

result = {"changes": changes}
print(json.dumps(result))
PY
}

# =============================================================================
# SIGNATURE HELP
# =============================================================================

get_signature_help() {
    local source_path="$1"
    local line="$2"
    local col="$3"
    
    python3 - "$source_path" "$line" "$col" <<'PY'
import json
import os
import re
import sys

source_path = sys.argv[1]
req_line = int(sys.argv[2])
req_col = int(sys.argv[3])

# Built-in function signatures
signatures = {
    "print": ("print(value: T)", "Print a value to stdout"),
    "println": ("println(value: T)", "Print a value with newline"),
    "assert": ("assert(condition: Bool)", "Assert condition is true"),
    "len": ("len(collection: Collection<T>) -> Int", "Get length of collection"),
    "range": ("range(start: Int, end: Int) -> Range<Int>", "Create integer range"),
    "map": ("map(fn: (A) -> B, collection: Collection<A>) -> Collection<B>", "Map function over collection"),
    "filter": ("filter(fn: (T) -> Bool, collection: Collection<T>) -> Collection<T>", "Filter collection"),
    "fold": ("fold(fn: (B, A) -> B, init: B, collection: Collection<A>) -> B", "Fold over collection"),
    "uncertain": ("uncertain(value: T, std: Float) -> Uncertain<T>", "Create uncertain value"),
    "confidence": ("confidence(value: Uncertain<T>) -> Interval<Float>", "Get confidence interval"),
}

# Try to extract the function name being called
result = {"signatures": [], "activeSignature": 0, "activeParameter": 0}

if os.path.exists(source_path):
    with open(source_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    if 0 <= req_line < len(lines):
        line = lines[req_line]
        # Look for function call pattern before cursor
        before_cursor = line[:req_col]
        call_pattern = re.compile(r'(\w+)\s*\(([^()]*)$')
        match = call_pattern.search(before_cursor)
        
        if match:
            func_name = match.group(1)
            args_before = match.group(2)
            
            if func_name in signatures:
                sig, doc = signatures[func_name]
                # Count active parameter by commas
                active_param = args_before.count(',')
                
                result = {
                    "signatures": [{
                        "label": sig,
                        "documentation": doc,
                        "parameters": []
                    }],
                    "activeSignature": 0,
                    "activeParameter": active_param
                }

print(json.dumps(result))
PY
}

# =============================================================================
# DOCUMENT HIGHLIGHTS
# =============================================================================

get_document_highlights() {
    local source_path="$1"
    local line="$2"
    local col="$3"
    
    python3 - "$source_path" "$line" "$col" <<'PY'
import json
import os
import re
import sys

source_path = sys.argv[1]
req_line = int(sys.argv[2])
req_col = int(sys.argv[3])

highlights = []

if os.path.exists(source_path):
    with open(source_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    if 0 <= req_line < len(lines):
        line = lines[req_line]
        # Extract word at position
        word_pattern = re.compile(r'\w+')
        target_word = None
        for match in word_pattern.finditer(line):
            if match.start() <= req_col <= match.end():
                target_word = match.group()
                break
        
        if target_word:
            # Find all occurrences
            for i, l in enumerate(lines):
                for match in word_pattern.finditer(l):
                    if match.group() == target_word:
                        kind = 1 if i == req_line and match.start() <= req_col <= match.end() else 2
                        highlights.append({
                            "range": {
                                "start": {"line": i, "character": match.start()},
                                "end": {"line": i, "character": match.end()}
                            },
                            "kind": kind
                        })

print(json.dumps(highlights))
PY
}

# =============================================================================
# RUN CHECK AND PUBLISH
# =============================================================================

run_check_and_publish() {
    local uri="$1"
    local expected_token="$2"
    local source_path cleanup_path
    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        log "skip diagnostics (source missing): $(uri_to_path "$uri")"
        return 0
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    kill_stale_check "$uri"

    local out_file err_file diagnostics rc
    out_file="$(mktemp)"
    err_file="$(mktemp)"
    rc=0

    (
        run_souc_check "$source_path" >"$out_file" 2>"$err_file"
    ) &
    CHECK_PID="$!"
    CHECK_PID_URI="$uri"
    if wait "$CHECK_PID"; then
        rc=0
    else
        rc=$?
    fi
    CHECK_PID=""
    CHECK_PID_URI=""

    if diagnostics="$($DIAG_PARSER <"$err_file" 2>/dev/null)"; then
        :
    else
        diagnostics='[]'
    fi
    if [[ "$diagnostics" == "[]" ]] && [[ "$rc" -ne 0 ]]; then
        diagnostics="$(build_failed_check_diagnostics "$rc" "$err_file")"
    fi

    local current_token="${OPEN_DOC_CHECK_TOKEN[$uri]:-0}"
    if [[ "$current_token" != "$expected_token" ]]; then
        log "drop stale diagnostics uri=$uri token=$expected_token current_token=$current_token"
        cleanup_temp_source "$cleanup_path"
        rm -f "$out_file" "$err_file"
        return 0
    fi

    local params
    params="$(jq -cn --arg uri "$uri" --argjson diagnostics "$diagnostics" \
        '{uri:$uri, diagnostics:$diagnostics}')"
    send_notification "textDocument/publishDiagnostics" "$params"

    cleanup_temp_source "$cleanup_path"
    rm -f "$out_file" "$err_file"
    return "$rc"
}

# =============================================================================
# LSP MESSAGE HANDLERS
# =============================================================================

handle_initialize() {
    local id_json="$1"
    local result
    result="$(jq -cn '{
        capabilities: {
            textDocumentSync: {
                openClose: true,
                change: 1,
                willSave: false,
                willSaveWaitUntil: false,
                save: { includeText: false }
            },
            hoverProvider: true,
            definitionProvider: true,
            completionProvider: {
                resolveProvider: false,
                triggerCharacters: [".", ":", ">", " ", "(", "["]
            },
            documentFormattingProvider: true,
            documentRangeFormattingProvider: true,
            documentOnTypeFormattingProvider: {
                firstTriggerCharacter: ";",
                moreTriggerCharacter: ["}", ","]
            },
            codeActionProvider: {
                codeActionKinds: ["quickfix", "refactor", "source"],
                resolveProvider: false
            },
            documentSymbolProvider: true,
            workspaceSymbolProvider: true,
            renameProvider: {
                prepareProvider: true
            },
            referencesProvider: true,
            documentHighlightProvider: true,
            signatureHelpProvider: {
                triggerCharacters: ["(", ","],
                retriggerCharacters: [","]
            },
            selectionRangeProvider: true,
            foldingRangeProvider: true,
            executeCommandProvider: {
                commands: ["sounio.organizeImports", "sounio.addImport", "sounio.showType"]
            }
        },
        serverInfo: { name: "sounio-lsp-big", version: "1.0.0" }
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
    if [[ -z "$uri" || "$uri" == "null" ]]; then
        log "didOpen missing uri; ignoring"
        return
    fi
    version="$(jq -r '.textDocument.version // 0' <<<"$params")"
    text="$(jq -r '.textDocument.text // ""' <<<"$params")"

    OPEN_DOCS["$uri"]="$text"
    OPEN_DOC_VERSION["$uri"]="$version"
    local check_token
    next_check_token "$uri" >/dev/null
    check_token="${OPEN_DOC_CHECK_TOKEN[$uri]}"
    run_check_and_publish "$uri" "$check_token" || true
}

handle_did_change() {
    local params="$1"
    local uri version new_text has_change_text current_version
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    if [[ -z "$uri" || "$uri" == "null" ]]; then
        log "didChange missing uri; ignoring"
        return
    fi
    version="$(jq -r '.textDocument.version // 0' <<<"$params")"

    current_version="${OPEN_DOC_VERSION[$uri]:-0}"
    if [[ "$version" =~ ^[0-9]+$ ]] && [[ "$current_version" =~ ^[0-9]+$ ]]; then
        if (( version < current_version )); then
            log "ignore stale didChange uri=$uri incoming_version=$version current_version=$current_version"
            return
        fi
    fi

    has_change_text="$(jq -r '
        (.contentChanges // [])
        | map(select(has("text")))
        | (length > 0)
    ' <<<"$params")"
    if [[ "$has_change_text" != "true" ]]; then
        log "didChange has no text entries; keeping current snapshot uri=$uri"
        OPEN_DOC_VERSION["$uri"]="$version"
        return
    fi

    new_text="$(jq -r '
        (.contentChanges // [])
        | map(select(has("text")))
        | .[-1].text // ""
    ' <<<"$params")"
    OPEN_DOCS["$uri"]="$new_text"
    OPEN_DOC_VERSION["$uri"]="$version"
}

handle_did_save() {
    local params="$1"
    local uri has_inline_text inline_text
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    if [[ -z "$uri" || "$uri" == "null" ]]; then
        log "didSave missing uri; ignoring"
        return
    fi

    has_inline_text="$(jq -r 'has("text")' <<<"$params")"
    if [[ "$has_inline_text" == "true" ]]; then
        inline_text="$(jq -r '.text // ""' <<<"$params")"
        OPEN_DOCS["$uri"]="$inline_text"
    fi

    local check_token
    next_check_token "$uri" >/dev/null
    check_token="${OPEN_DOC_CHECK_TOKEN[$uri]}"
    run_check_and_publish "$uri" "$check_token" || true
}

handle_did_close() {
    local params="$1"
    local uri
    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    if [[ -z "$uri" || "$uri" == "null" ]]; then
        log "didClose missing uri; ignoring"
        return
    fi
    next_check_token "$uri" >/dev/null
    kill_stale_check "$uri"
    unset OPEN_DOCS["$uri"]
    unset OPEN_DOC_VERSION["$uri"]
    local clear
    clear="$(jq -cn --arg uri "$uri" '{uri:$uri, diagnostics:[]}')"
    send_notification "textDocument/publishDiagnostics" "$clear"
}

handle_hover() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col line1 col1 output ty result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"
    line1="$((line + 1))"
    col1="$((col + 1))"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" "null"
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    output="$(run_souc_check "$source_path" --show-types 2>&1 || true)"
    cleanup_temp_source "$cleanup_path"
    if ! ty="$(extract_hover_type "$output" "$line1" "$col1")"; then
        send_response "$id_json" "null"
        return
    fi

    result="$(jq -cn --arg ty "$ty" '{
        contents: {
            kind: "markdown",
            value: ("```sounio\n" + $ty + "\n```")
        }
    }')"
    send_response "$id_json" "$result"
}

handle_definition() {
    local id_json="$1"
    local params="$2"
    local uri file_path source_path cleanup_path line col line1 col1 output result location

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    file_path="$(uri_to_path "$uri")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"
    line1="$((line + 1))"
    col1="$((col + 1))"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" "null"
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    output="$(run_souc_check "$source_path" --show-defs 2>&1 || true)"
    if [[ "$output" == *"unrecognized option"* ]] || [[ "$output" == *"unknown option"* ]]; then
        output="$(run_souc_check "$source_path" --show-ast 2>&1 || true)"
    fi
    cleanup_temp_source "$cleanup_path"

    if location="$(extract_definition_location_json "$output" "$line1" "$col1" "$file_path")"; then
        local location_uri
        location_uri="$(jq -r '.uri // empty' <<<"$location")"
        if [[ -n "$cleanup_path" ]] && [[ -n "$location_uri" ]]; then
            local snapshot_uri
            snapshot_uri="$(path_to_uri "$cleanup_path")"
            if [[ "$location_uri" == "$snapshot_uri" ]]; then
                location_uri="$uri"
            fi
        fi
        if [[ -z "$location_uri" ]]; then
            location_uri="$uri"
        fi
        result="$(jq -cn --arg uri "$location_uri" --argjson loc "$location" '$loc | .uri = $uri')"
        send_response "$id_json" "$result"
        return
    fi

    send_response "$id_json" "null"
}

# =============================================================================
# NEW BIG LSP FEATURES
# =============================================================================

handle_completion() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col content

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '{"items":[]}'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    content="${OPEN_DOCS[$uri]:-}"
    if [[ -z "$content" ]] && [[ -f "$source_path" ]]; then
        content="$(cat "$source_path")"
    fi

    local completions
    completions="$(get_completions "$source_path" "$line" "$col" "$content")"
    
    cleanup_temp_source "$cleanup_path"

    local result
    result="$(jq -cn --argjson items "$completions" '{items:$items}')"
    send_response "$id_json" "$result"
}

handle_formatting() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path formatted result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" "null"
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    # Use souc fmt to format the document
    formatted="$(run_souc_fmt "$source_path" 2>/dev/null || cat "$source_path")"
    
    cleanup_temp_source "$cleanup_path"

    # Create text edit for entire document
    local original_lines new_lines
    original_lines="$(wc -l < "$source_path" | tr -d ' ')"
    new_lines="$(printf '%s' "$formatted" | wc -l | tr -d ' ')"
    
    result="$(jq -cn --arg text "$formatted" '[{
        range: {
            start: {line: 0, character: 0},
            end: {line: 999999, character: 999999}
        },
        newText: $text
    }]')"
    
    send_response "$id_json" "$result"
}

handle_code_action() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path context actions result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    context="$(jq -c '.context // {}' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '[]'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    local diagnostics
    diagnostics="$(jq -c '.diagnostics // []' <<<"$context")"
    
    actions="$(get_code_actions "$source_path" "$diagnostics")"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$actions"
}

handle_document_symbol() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path symbols result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '[]'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    symbols="$(extract_document_symbols "$source_path" "$uri")"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$symbols"
}

handle_workspace_symbol() {
    local id_json="$1"
    local params="$2"
    local query symbols result

    query="$(jq -r '.query // ""' <<<"$params")"
    
    # For now, return empty or search in open documents
    # In a full implementation, this would search the entire workspace
    symbols="[]"
    
    send_response "$id_json" "$symbols"
}

handle_references() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col include_decl references result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"
    include_decl="$(jq -r '.context.includeDeclaration // true' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '[]'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    references="$(find_references "$source_path" "$line" "$col" "$include_decl")"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$references"
}

handle_rename() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col new_name result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"
    new_name="$(jq -r '.newName // ""' <<<"$params")"

    if [[ -z "$new_name" ]]; then
        send_error "$id_json" -32600 "newName is required"
        return
    fi

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '{"changes":{}}'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    result="$(get_rename_edits "$source_path" "$line" "$col" "$new_name")"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$result"
}

handle_prepare_rename() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" "null"
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    # Return the range of the word at position
    result="$(python3 - "$source_path" "$line" "$col" <<'PY'
import json
import os
import re
import sys

source_path = sys.argv[1]
req_line = int(sys.argv[2])
req_col = int(sys.argv[3])

if os.path.exists(source_path):
    with open(source_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    if 0 <= req_line < len(lines):
        line = lines[req_line]
        word_pattern = re.compile(r'\w+')
        for match in word_pattern.finditer(line):
            if match.start() <= req_col <= match.end():
                print(json.dumps({
                    "start": {"line": req_line, "character": match.start()},
                    "end": {"line": req_line, "character": match.end()}
                }))
                exit(0)

print("null")
PY
    )"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$result"
}

handle_signature_help() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '{"signatures":[]}'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    result="$(get_signature_help "$source_path" "$line" "$col")"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$result"
}

handle_document_highlight() {
    local id_json="$1"
    local params="$2"
    local uri source_path cleanup_path line col highlights result

    uri="$(jq -r '.textDocument.uri' <<<"$params")"
    line="$(jq -r '.position.line' <<<"$params")"
    col="$(jq -r '.position.character' <<<"$params")"

    if ! source_path="$(resolve_source_path_for_uri "$uri")"; then
        send_response "$id_json" '[]'
        return
    fi
    cleanup_path="$LAST_SOURCE_CLEANUP"

    highlights="$(get_document_highlights "$source_path" "$line" "$col")"
    
    cleanup_temp_source "$cleanup_path"

    send_response "$id_json" "$highlights"
}

# =============================================================================
# MESSAGE DISPATCH
# =============================================================================

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
        textDocument/completion)
            if [[ -n "$id_json" ]]; then
                handle_completion "$id_json" "$params"
            fi
            ;;
        textDocument/formatting)
            if [[ -n "$id_json" ]]; then
                handle_formatting "$id_json" "$params"
            fi
            ;;
        textDocument/rangeFormatting)
            if [[ -n "$id_json" ]]; then
                # For now, just format entire document
                handle_formatting "$id_json" "$params"
            fi
            ;;
        textDocument/codeAction)
            if [[ -n "$id_json" ]]; then
                handle_code_action "$id_json" "$params"
            fi
            ;;
        textDocument/documentSymbol)
            if [[ -n "$id_json" ]]; then
                handle_document_symbol "$id_json" "$params"
            fi
            ;;
        workspace/symbol)
            if [[ -n "$id_json" ]]; then
                handle_workspace_symbol "$id_json" "$params"
            fi
            ;;
        textDocument/references)
            if [[ -n "$id_json" ]]; then
                handle_references "$id_json" "$params"
            fi
            ;;
        textDocument/rename)
            if [[ -n "$id_json" ]]; then
                handle_rename "$id_json" "$params"
            fi
            ;;
        textDocument/prepareRename)
            if [[ -n "$id_json" ]]; then
                handle_prepare_rename "$id_json" "$params"
            fi
            ;;
        textDocument/signatureHelp)
            if [[ -n "$id_json" ]]; then
                handle_signature_help "$id_json" "$params"
            fi
            ;;
        textDocument/documentHighlight)
            if [[ -n "$id_json" ]]; then
                handle_document_highlight "$id_json" "$params"
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
    log "startup souc=$SOUC_BIN BIG LSP v1.0.0"
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
