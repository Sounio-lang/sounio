#!/usr/bin/env bash
# Parse Sounio compiler stderr output and convert to LSP Diagnostic JSON array.
# Reads from stdin and prints a JSON array to stdout.

set -euo pipefail

get_severity() {
    local level="$1"
    case "${level,,}" in
        error) echo 1 ;;
        warning|warn) echo 2 ;;
        info|information) echo 3 ;;
        note|hint) echo 4 ;;
        *) echo 1 ;;
    esac
}

trim_left() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    printf '%s' "$s"
}

append_diag() {
    local level="$1"
    local code="$2"
    local message="$3"
    local line="$4"
    local col="$5"
    local end_col="$6"

    if [[ -z "$message" ]]; then
        return 0
    fi

    if ! [[ "$line" =~ ^[0-9]+$ ]]; then
        line=1
    fi
    if ! [[ "$col" =~ ^[0-9]+$ ]]; then
        col=1
    fi
    if ! [[ "$end_col" =~ ^[0-9]+$ ]]; then
        end_col=$((col + 1))
    fi
    if (( end_col <= col )); then
        end_col=$((col + 1))
    fi

    local severity
    severity="$(get_severity "$level")"

    local diag
    diag="$(jq -cn \
        --arg message "$message" \
        --argjson line "$((line - 1))" \
        --argjson col "$((col - 1))" \
        --argjson end_col "$((end_col - 1))" \
        --argjson severity "$severity" \
        --arg code "$code" \
        '{
            range: {
                start: {line: $line, character: $col},
                end: {line: $line, character: $end_col}
            },
            severity: $severity,
            code: (if $code != "" then $code else null end),
            source: "sounio",
            message: $message
        }')"

    if [[ -n "$DIAGNOSTICS" ]]; then
        DIAGNOSTICS+=","
    fi
    DIAGNOSTICS+="$diag"
}

main() {
    local input=""
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        input+="$line"$'\n'
    done

    DIAGNOSTICS=""
    local current_level=""
    local current_code=""
    local current_message=""
    local current_line=""
    local current_col=""
    local current_end_col=""

    while IFS= read -r line; do
        if [[ "$line" =~ ^(error|warning|warn|info|note)[[:space:]]*(\[([EW][0-9]+)\])?[[:space:]]*:[[:space:]]*(.+)$ ]]; then
            append_diag "$current_level" "$current_code" "$current_message" "$current_line" "$current_col" "$current_end_col"

            current_level="${BASH_REMATCH[1]}"
            current_code="${BASH_REMATCH[3]:-}"
            current_message="${BASH_REMATCH[4]}"
            current_line=""
            current_col=""
            current_end_col=""
            continue
        fi

        if [[ "$line" =~ ^[[:space:]]*--\>[[:space:]]+(.+):([0-9]+):([0-9]+)[[:space:]]*$ ]]; then
            current_line="${BASH_REMATCH[2]}"
            current_col="${BASH_REMATCH[3]}"
            current_end_col=$((current_col + 1))
            continue
        fi

        if [[ "$line" =~ ^[[:space:]]*\| ]]; then
            local rhs
            rhs="${line#*|}"

            local caret_start=-1
            local caret_end=-1
            local i
            for ((i = 0; i < ${#rhs}; i++)); do
                local c="${rhs:$i:1}"
                if [[ "$c" == "^" ]]; then
                    if (( caret_start < 0 )); then
                        caret_start=$i
                    fi
                    caret_end=$i
                fi
            done

            if (( caret_start >= 0 )) && [[ -n "$current_col" ]]; then
                current_end_col=$((current_col + caret_end - caret_start + 1))
                local extra_msg
                extra_msg="$(trim_left "${rhs##*^}")"
                if [[ -n "$extra_msg" ]] && [[ "$extra_msg" != "$rhs" ]]; then
                    current_message+=": $extra_msg"
                fi
                continue
            fi

            local continuation
            continuation="$(trim_left "$rhs")"
            if [[ "$continuation" == "Type errors:" ]] || [[ "$continuation" == "Parse errors:" ]] || [[ "$continuation" == "× Type errors:" ]] || [[ "$continuation" == "× Parse errors:" ]]; then
                continue
            fi
            if [[ -n "$continuation" ]] && [[ "$continuation" != "|" ]]; then
                current_message+=" $continuation"
            fi
            continue
        fi

        if [[ "$line" =~ ^Error:[[:space:]]*(.+)$ ]]; then
            local detail="${BASH_REMATCH[1]}"
            if [[ "$detail" != "Type errors:" ]] && [[ "$detail" != "Parse errors:" ]] && [[ "$detail" != "× Type errors:" ]] && [[ "$detail" != "× Parse errors:" ]]; then
                current_message+=": $detail"
            fi
            continue
        fi
    done <<< "$input"

    append_diag "$current_level" "$current_code" "$current_message" "$current_line" "$current_col" "$current_end_col"
    printf '[%s]\n' "$DIAGNOSTICS"
}

main "$@"
