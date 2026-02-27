#!/usr/bin/env bash
#
# TOML parser for Sounio package manager
# Parses basic TOML format: [sections] and key=value pairs
#

set -euo pipefail

# Parse a TOML file and output as KEY=VALUE pairs
# Format: section.key=value
# Usage: toml_parse "file.toml"
toml_parse() {
    local file="$1"
    local current_section=""
    local in_array=false
    local array_name=""
    local array_values=()

    [[ ! -f "$file" ]] && return 1

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

        # Trim leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        # Handle array of tables: [[section]]
        if [[ "$line" =~ ^\[\[([^]]+)\]\]$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            printf '%s\n' "[$current_section]"
            continue
        fi

        # Handle section headers: [section] or [section.subsection]
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            continue
        fi

        # Handle inline tables: key = { ... }
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*\{(.*)\}[[:space:]]*$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local inline="${BASH_REMATCH[2]}"
            local full_key="${current_section:+${current_section}.}${key}"
            printf '%s=%s\n' "$full_key" "{${inline}}"
            continue
        fi

        # Handle array values
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*\[ ]]; then
            in_array=true
            array_name="${BASH_REMATCH[1]}"
            array_values=()
            local rest="${line#*[}"
            if [[ "$line" =~ \] ]]; then
                in_array=false
                rest="${rest%\]*}"
                IFS=',' read -ra items <<< "$rest"
                for item in "${items[@]}"; do
                    item="${item#"${item%%[![:space:]]*}"}"
                    item="${item%"${item##*[![:space:]]}"}"
                    item="${item#\"}"
                    item="${item%\"}"
                    item="${item#\'}"
                    item="${item%\'}"
                    [[ -n "$item" ]] && array_values+=("$item")
                done
                local full_key="${current_section:+${current_section}.}${array_name}"
                printf '%s=%s\n' "$full_key" "[${array_values[*]}]"
            fi
            continue
        fi

        # Continue reading array values
        if $in_array; then
            if [[ "$line" =~ \] ]]; then
                in_array=false
                local value="${line%\]*}"
                value="${value%,}"
                value="${value#"${value%%[![:space:]]*}"}"
                value="${value%"${value##*[![:space:]]}"}"
                value="${value#\"}"
                value="${value%\"}"
                value="${value#\'}"
                value="${value%\'}"
                [[ -n "$value" ]] && array_values+=("$value")
                local full_key="${current_section:+${current_section}.}${array_name}"
                printf '%s=%s\n' "$full_key" "[${array_values[*]}]"
            else
                local value="${line%,}"
                value="${value#"${value%%[![:space:]]*}"}"
                value="${value%"${value##*[![:space:]]}"}"
                value="${value#\"}"
                value="${value%\"}"
                value="${value#\'}"
                value="${value%\'}"
                [[ -n "$value" ]] && array_values+=("$value")
            fi
            continue
        fi

        # Handle key-value pairs
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Remove inline comments
            value="${value%%#*}"

            # Trim whitespace
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"

            # Handle quoted strings
            if [[ "$value" =~ ^\"(.*)\"$ ]]; then
                value="${BASH_REMATCH[1]}"
            elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
                value="${BASH_REMATCH[1]}"
            fi

            local full_key="${current_section:+${current_section}.}${key}"
            printf '%s=%s\n' "$full_key" "$value"
        fi
    done < "$file"
}

# Parse an inline table into key=value pairs
# Usage: toml_parse_inline_table "{key=value, key2=value2}"
toml_parse_inline_table() {
    local inline="$1"

    # Remove braces
    inline="${inline#\{}"
    inline="${inline%\}}"

    # Parse comma-separated pairs
    local IFS=','
    read -ra pairs <<< "$inline"

    for pair in "${pairs[@]}"; do
        pair="${pair#"${pair%%[![:space:]]*}"}"
        pair="${pair%"${pair##*[![:space:]]}"}"

        if [[ "$pair" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            local k="${BASH_REMATCH[1]}"
            local v="${BASH_REMATCH[2]}"

            # Trim and unquote value
            v="${v#"${v%%[![:space:]]*}"}"
            v="${v%"${v##*[![:space:]]}"}"
            v="${v#\"}"
            v="${v%\"}"
            v="${v#\'}"
            v="${v%\'}"

            printf '%s=%s\n' "$k" "$v"
        fi
    done
}

# Get a value from parsed TOML output
# Usage: toml_get "parsed_output" "key.subkey"
toml_get() {
    local parsed="$1"
    local key="$2"

    while IFS='=' read -r k v; do
        [[ "$k" == "$key" ]] || continue
        printf '%s\n' "$v"
        return 0
    done <<< "$parsed"

    return 1
}

# Get all keys matching a prefix
# Usage: toml_get_keys "parsed_output" "section."
toml_get_keys() {
    local parsed="$1"
    local prefix="${2:-}"

    while IFS='=' read -r k v; do
        [[ "$k" == "$prefix"* ]] || continue
        printf '%s\n' "$k"
    done <<< "$parsed"
}

# Check if a key exists
# Usage: toml_has "parsed_output" "key"
toml_has() {
    local parsed="$1"
    local key="$2"

    toml_get "$parsed" "$key" >/dev/null 2>&1
}

# Get array elements (returns space-separated values)
# Usage: toml_get_array "parsed_output" "section.array"
toml_get_array() {
    local parsed="$1"
    local key="$2"
    local value

    value=$(toml_get "$parsed" "$key") || return 1

    # Check if it's an array format [item1 item2 ...]
    if [[ "$value" =~ ^\[(.*)\]$ ]]; then
        local content="${BASH_REMATCH[1]}"
        printf '%s\n' "$content"
    else
        printf '%s\n' "$value"
    fi
}

# Example usage:
#   parsed=$(toml_parse "Sounio.toml")
#   name=$(toml_get "$parsed" "package.name")
#   version=$(toml_get "$parsed" "package.version")
#
#   # Parse inline table
#   dep_info=$(toml_get "$parsed" "dependencies.somelib")
#   toml_parse_inline_table "$dep_info" | while IFS='=' read -r k v; do
#       echo "Key: $k, Value: $v"
#   done
