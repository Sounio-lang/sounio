#!/usr/bin/env bash
# gen_stdlib_docs.sh — Extract stdlib API documentation from .sio source files
#
# Walks stdlib/**/*.sio, extracts triple-slash doc comments (/// ...) that
# immediately precede fn, struct, or type declarations, and writes
# website/src/data/stdlib-api.json.
#
# Output shape:
#   [{"module": "stdlib/epistemic/lib", "name": "propagate", "kind": "fn",
#     "doc": "Propagate uncertainty through a function.", "signature": "fn propagate(...)"}]
#
# Requirements: bash, awk — no external deps.
#
# Usage:
#   bash scripts/build/gen_stdlib_docs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STDLIB_DIR="${REPO_ROOT}/stdlib"
OUT_FILE="${REPO_ROOT}/website/src/data/stdlib-api.json"

mkdir -p "$(dirname "${OUT_FILE}")"

# awk script: reads one .sio file, emits JSON objects (one per declaration)
# Called once per file; filename passed via -v module=<module>
read -r -d '' AWK_SCRIPT << 'AWKEOF'
BEGIN {
    in_doc = 0
    doc = ""
    record_count = 0
}

# Strip trailing \r (Windows line endings)
{ sub(/\r$/, "") }

# Triple-slash doc comment
/^[[:space:]]*\/\/\/[[:space:]]?/ {
    # Accumulate doc lines
    line = $0
    sub(/^[[:space:]]*\/\/\/[[:space:]]?/, "", line)
    if (in_doc) {
        doc = doc "\\n" line
    } else {
        doc = line
        in_doc = 1
    }
    next
}

# fn declaration
in_doc && /^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]/ {
    sig = $0
    sub(/^[[:space:]]+/, "", sig)
    # Extract name: first identifier after `fn `
    name = sig
    sub(/^(pub[[:space:]]+)?fn[[:space:]]+/, "", name)
    sub(/[[:space:](].*/, "", name)
    gsub(/\\n/, "\\n", doc)
    gsub(/"/, "\\\"", doc)
    gsub(/"/, "\\\"", sig)
    if (record_count > 0) printf(",\n")
    printf("{\"module\":\"%s\",\"name\":\"%s\",\"kind\":\"fn\",\"doc\":\"%s\",\"signature\":\"%s\"}", module, name, doc, sig)
    record_count++
    in_doc = 0; doc = ""
    next
}

# struct declaration
in_doc && /^[[:space:]]*(pub[[:space:]]+)?(linear[[:space:]]+)?struct[[:space:]]/ {
    sig = $0
    sub(/^[[:space:]]+/, "", sig)
    name = sig
    sub(/^(pub[[:space:]]+)?(linear[[:space:]]+)?struct[[:space:]]+/, "", name)
    sub(/[[:space:]{].*/, "", name)
    gsub(/"/, "\\\"", doc)
    gsub(/"/, "\\\"", sig)
    if (record_count > 0) printf(",\n")
    printf("{\"module\":\"%s\",\"name\":\"%s\",\"kind\":\"struct\",\"doc\":\"%s\",\"signature\":\"%s\"}", module, name, doc, sig)
    record_count++
    in_doc = 0; doc = ""
    next
}

# type alias declaration
in_doc && /^[[:space:]]*(pub[[:space:]]+)?type[[:space:]]/ {
    sig = $0
    sub(/^[[:space:]]+/, "", sig)
    name = sig
    sub(/^(pub[[:space:]]+)?type[[:space:]]+/, "", name)
    sub(/[[:space:]=].*/, "", name)
    gsub(/"/, "\\\"", doc)
    gsub(/"/, "\\\"", sig)
    if (record_count > 0) printf(",\n")
    printf("{\"module\":\"%s\",\"name\":\"%s\",\"kind\":\"type\",\"doc\":\"%s\",\"signature\":\"%s\"}", module, name, doc, sig)
    record_count++
    in_doc = 0; doc = ""
    next
}

# Any other non-blank, non-comment line resets the doc accumulator
/^[[:space:]]*[^[:space:]\/]/ {
    in_doc = 0; doc = ""
}
AWKEOF

echo "[gen_stdlib_docs] Scanning ${STDLIB_DIR}/**/*.sio ..."

# Build JSON array
{
    echo "["
    first=1
    while IFS= read -r -d '' sio_file; do
        # Derive module path relative to repo root, strip .sio extension
        rel="${sio_file#${REPO_ROOT}/}"
        module="${rel%.sio}"

        # Run awk; if it emits anything, prepend comma separator between files
        chunk="$(awk -v module="${module}" "${AWK_SCRIPT}" "${sio_file}" 2>/dev/null || true)"
        if [[ -n "${chunk}" ]]; then
            if [[ "${first}" == "0" ]]; then
                echo ","
            fi
            printf '%s' "${chunk}"
            first=0
        fi
    done < <(find "${STDLIB_DIR}" -name '*.sio' -print0 | sort -z)
    echo ""
    echo "]"
} > "${OUT_FILE}"

count="$(grep -c '"module"' "${OUT_FILE}" 2>/dev/null || echo 0)"
echo "[gen_stdlib_docs] Wrote ${count} entries → ${OUT_FILE}"
