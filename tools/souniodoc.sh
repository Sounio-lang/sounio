#!/usr/bin/env bash
# souniodoc — lightweight documentation generator for Sounio source files
#
# Extracts doc comments (/// and //!) and pairs them with definitions.
# Outputs Markdown documentation.
#
# Usage:
#   bash tools/souniodoc.sh <file.sio>            # single file
#   bash tools/souniodoc.sh <directory>            # all .sio files recursively
#   bash tools/souniodoc.sh <file.sio> --html      # wrap in HTML
#
# Doc comment syntax:
#   /// Outer doc comment (documents the following item)
#   //! Inner doc comment (documents the enclosing module)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: souniodoc <file.sio|directory> [--html]" >&2
    exit 1
fi

TARGET="$1"
FORMAT="${2:-md}"

extract_docs() {
    local file="$1"
    local module_name
    module_name="$(basename "$file" .sio)"
    local in_doc=false
    local doc_buf=""
    local module_doc=""

    echo "## Module: \`$module_name\`"
    echo ""
    echo "**Source:** \`$file\`"
    echo ""

    while IFS= read -r line; do
        # Trim leading whitespace
        local trimmed="${line#"${line%%[![:space:]]*}"}"

        # Inner doc comment (//!)
        if [[ "$trimmed" =~ ^//!\  ]]; then
            local text="${trimmed#//! }"
            module_doc="${module_doc}${text}\n"
            continue
        elif [[ "$trimmed" == "//!" ]]; then
            module_doc="${module_doc}\n"
            continue
        fi

        # Outer doc comment (///)
        if [[ "$trimmed" =~ ^///\  ]]; then
            local text="${trimmed#/// }"
            doc_buf="${doc_buf}${text}\n"
            in_doc=true
            continue
        elif [[ "$trimmed" == "///" ]]; then
            doc_buf="${doc_buf}\n"
            in_doc=true
            continue
        fi

        # If we had doc comments and hit a definition, emit
        if $in_doc; then
            if [[ "$trimmed" =~ ^fn\  ]]; then
                local sig="${trimmed%%\{*}"
                sig="${sig%%\=*}"
                sig="$(echo "$sig" | sed 's/[[:space:]]*$//')"
                echo "### \`$sig\`"
                echo ""
                echo -e "$doc_buf"
                echo ""
                doc_buf=""
                in_doc=false
            elif [[ "$trimmed" =~ ^struct\  ]]; then
                local name="${trimmed#struct }"
                name="${name%%\{*}"
                name="${name%%\(*}"
                name="$(echo "$name" | sed 's/[[:space:]]*$//')"
                echo "### struct \`$name\`"
                echo ""
                echo -e "$doc_buf"
                echo ""
                doc_buf=""
                in_doc=false
            elif [[ "$trimmed" =~ ^enum\  ]]; then
                local name="${trimmed#enum }"
                name="${name%%\{*}"
                name="$(echo "$name" | sed 's/[[:space:]]*$//')"
                echo "### enum \`$name\`"
                echo ""
                echo -e "$doc_buf"
                echo ""
                doc_buf=""
                in_doc=false
            elif [[ "$trimmed" =~ ^type\  ]]; then
                local name="${trimmed#type }"
                name="${name%%\=*}"
                name="$(echo "$name" | sed 's/[[:space:]]*$//')"
                echo "### type \`$name\`"
                echo ""
                echo -e "$doc_buf"
                echo ""
                doc_buf=""
                in_doc=false
            elif [[ "$trimmed" =~ ^linear\ struct\  ]]; then
                local name="${trimmed#linear struct }"
                name="${name%%\{*}"
                name="$(echo "$name" | sed 's/[[:space:]]*$//')"
                echo "### linear struct \`$name\`"
                echo ""
                echo -e "$doc_buf"
                echo ""
                doc_buf=""
                in_doc=false
            elif [[ -n "$trimmed" && ! "$trimmed" =~ ^// ]]; then
                # Non-definition line after doc comment — discard orphan docs
                doc_buf=""
                in_doc=false
            fi
        fi
    done < "$file"

    # Emit module-level doc at the top (after header)
    if [[ -n "$module_doc" ]]; then
        echo "**Module documentation:**"
        echo ""
        echo -e "$module_doc"
        echo ""
    fi

    echo "---"
    echo ""
}

wrap_html() {
    echo "<!DOCTYPE html>"
    echo "<html><head><meta charset=\"utf-8\"><title>Sounio Documentation</title>"
    echo "<style>"
    echo "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 0 auto; padding: 2em; }"
    echo "h1 { border-bottom: 2px solid #333; }"
    echo "h2 { border-bottom: 1px solid #ccc; margin-top: 2em; }"
    echo "h3 { color: #1a73e8; }"
    echo "code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }"
    echo "pre { background: #f8f8f8; padding: 1em; border-radius: 5px; overflow-x: auto; }"
    echo "</style></head><body>"
    echo "<h1>Sounio Documentation</h1>"
    # Convert markdown to basic HTML (simplified)
    sed 's/^### /\n<h3>/; s/^## /\n<h2>/; s/$/\n/; s/`\([^`]*\)`/<code>\1<\/code>/g'
    echo "</body></html>"
}

echo "# Sounio Documentation"
echo ""
echo "Generated by souniodoc"
echo ""

if [[ -d "$TARGET" ]]; then
    while IFS= read -r -d '' file; do
        extract_docs "$file"
    done < <(find "$TARGET" -name '*.sio' -print0 | sort -z)
elif [[ -f "$TARGET" ]]; then
    extract_docs "$TARGET"
else
    echo "error: $TARGET is not a file or directory" >&2
    exit 1
fi

if [[ "$FORMAT" == "--html" ]]; then
    # Re-run and pipe through HTML wrapper (in practice, pipe output)
    echo "(HTML mode: pipe output through 'tools/souniodoc.sh ... | tools/souniodoc.sh --html')" >&2
fi
