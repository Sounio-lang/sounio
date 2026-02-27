#!/usr/bin/env bash
# Sounio TOML Parser Library
# Parses TOML files to key=value pairs

set -euo pipefail

# Parse a TOML file and output key=value pairs
# Keys are formatted as "section.key" or "section.table.key"
# Usage: toml_parse "file.toml"
toml_parse() {
    local file="$1"
    local section=""
    local line_num=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        line_num=$((line_num + 1))
        
        # Skip empty lines and comments
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Trim leading/trailing whitespace
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Handle section headers [section] or [[section]]
        if [[ "$line" =~ ^\[\[([^\]]+)\]\]$ ]]; then
            # Array of tables - treat as regular section for now
            section="${BASH_REMATCH[1]}"
            continue
        elif [[ "$line" =~ ^\[([^\]]+)\]$ ]]; then
            section="${BASH_REMATCH[1]}"
            continue
        fi
        
        # Handle inline tables: key = { ... }
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*\{(.+)\} ]]; then
            local key="${BASH_REMATCH[1]}"
            local table_content="${BASH_REMATCH[2]}"
            
            if [[ -n "$section" ]]; then
                toml_parse_inline_table "$table_content" | while IFS='=' read -r subkey value; do
                    echo "${section}.${key}.${subkey}=${value}"
                done
            else
                toml_parse_inline_table "$table_content" | while IFS='=' read -r subkey value; do
                    echo "${key}.${subkey}=${value}"
                done
            fi
            continue
        fi
        
        # Handle key = value pairs
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            
            # Remove quotes from values
            value=$(echo "$value" | sed -E 's/^"(.*)"$/\1/;s/^'\''(.*)'\''$/\1/')
            
            if [[ -n "$section" ]]; then
                echo "${section}.${key}=${value}"
            else
                echo "${key}=${value}"
            fi
        fi
    done < "$file"
}

# Parse inline table content: key1 = value1, key2 = value2
# Usage: toml_parse_inline_table "key1 = \"val1\", key2 = 123"
toml_parse_inline_table() {
    local content="$1"
    local IFS=','
    
    # Split by comma and process each pair
    echo "$content" | tr ',' '\n' | while IFS='=' read -r key value; do
        # Trim whitespace
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Remove quotes
        value=$(echo "$value" | sed -E 's/^"(.*)"$/\1/;s/^'\''(.*)'\''$/\1/')
        
        if [[ -n "$key" ]]; then
            echo "${key}=${value}"
        fi
    done
}

# Get value from parsed TOML output
# Usage: toml_get "parsed_output" "section.key"
toml_get() {
    local parsed="$1"
    local key="$2"
    
    echo "$parsed" | grep "^${key}=" | cut -d'=' -f2- | head -1
}

# Get all keys under a section
# Usage: toml_get_section_keys "parsed_output" "dependencies"
toml_get_section_keys() {
    local parsed="$1"
    local section="$2"
    
    echo "$parsed" | grep "^${section}\." | cut -d'=' -f1 | sed "s/^${section}\.//"
}

# Check if a key exists in parsed TOML
# Usage: toml_has_key "parsed_output" "section.key"
toml_has_key() {
    local parsed="$1"
    local key="$2"
    
    echo "$parsed" | grep -q "^${key}="
}

# Parse dependencies section specially
# Returns: dep_name|version|git_url|branch|path
toml_parse_dependencies() {
    local file="$1"
    local in_deps=false
    local dep_name=""
    local dep_version=""
    local dep_git=""
    local dep_branch=""
    local dep_path=""
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Check for [dependencies] section
        if [[ "$line" =~ ^\[dependencies\] ]]; then
            in_deps=true
            continue
        fi
        
        # Exit if we hit another section
        if [[ "$line" =~ ^\[ && ! "$line" =~ ^\[dependencies\] ]]; then
            if [[ "$in_deps" == true ]]; then
                in_deps=false
            fi
            continue
        fi
        
        [[ "$in_deps" != true ]] && continue
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Handle inline table dependency
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*\{(.+)\} ]]; then
            dep_name="${BASH_REMATCH[1]}"
            local table_content="${BASH_REMATCH[2]}"
            
            dep_version=""
            dep_git=""
            dep_branch=""
            dep_path=""
            
            # Parse inline table
            echo "$table_content" | tr ',' '\n' | while IFS='=' read -r key value; do
                key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                value=$(echo "$value" | sed -E 's/^"(.*)"$/\1/;s/^'\''(.*)'\''$/\1/')
                
                case "$key" in
                    version) echo "VERSION:$value" ;;
                    git) echo "GIT:$value" ;;
                    branch) echo "BRANCH:$value" ;;
                    path) echo "PATH:$value" ;;
                esac
            done | {
                while IFS=: read -r k v; do
                    case "$k" in
                        VERSION) dep_version="$v" ;;
                        GIT) dep_git="$v" ;;
                        BRANCH) dep_branch="$v" ;;
                        PATH) dep_path="$v" ;;
                    esac
                done
                echo "${dep_name}|${dep_version}|${dep_git}|${dep_branch}|${dep_path}"
            }
        fi
    done < "$file"
}
