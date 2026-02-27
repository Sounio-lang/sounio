#!/usr/bin/env bash
# Semantic Versioning Library for Sounio Package Manager
# Supports parsing, comparison, and constraint satisfaction

set -euo pipefail

# Parse a semantic version string
# Usage: semver_parse "1.2.3" -> outputs: major=1 minor=2 patch=3 prerelease="" build=""
semver_parse() {
    local version="$1"
    
    # Remove leading 'v' if present
    version="${version#v}"
    
    # Extract build metadata (after +)
    local build=""
    if [[ "$version" =~ ^([^+]+)\+(.+)$ ]]; then
        version="${BASH_REMATCH[1]}"
        build="${BASH_REMATCH[2]}"
    fi
    
    # Extract prerelease (after -)
    local prerelease=""
    if [[ "$version" =~ ^([^-]+)-(.+)$ ]]; then
        version="${BASH_REMATCH[1]}"
        prerelease="${BASH_REMATCH[2]}"
    fi
    
    # Parse major.minor.patch
    local major=0 minor=0 patch=0
    IFS='.' read -r major minor patch <<< "$version"
    
    # Ensure numeric values
    [[ "$major" =~ ^[0-9]+$ ]] || major=0
    [[ "$minor" =~ ^[0-9]+$ ]] || minor=0
    [[ "$patch" =~ ^[0-9]+$ ]] || patch=0
    
    echo "major=${major}"
    echo "minor=${minor}"
    echo "patch=${patch}"
    echo "prerelease=${prerelease}"
    echo "build=${build}"
}

# Get individual component from parsed version
semver_get_major() { semver_parse "$1" | grep "^major=" | cut -d'=' -f2; }
semver_get_minor() { semver_parse "$1" | grep "^minor=" | cut -d'=' -f2; }
semver_get_patch() { semver_parse "$1" | grep "^patch=" | cut -d'=' -f2; }

# Compare two prerelease identifiers
# Returns: -1 if a < b, 0 if equal, 1 if a > b
_prerelease_compare() {
    local a="$1"
    local b="$2"
    
    # Empty prerelease has higher precedence than any prerelease
    [[ -z "$a" && -z "$b" ]] && echo 0 && return
    [[ -z "$a" && -n "$b" ]] && echo 1 && return
    [[ -n "$a" && -z "$b" ]] && echo -1 && return
    
    # Split by dots and compare each identifier
    local IFS='.'
    local a_parts=($a)
    local b_parts=($b)
    local max_len=${#a_parts[@]}
    [[ ${#b_parts[@]} -gt $max_len ]] && max_len=${#b_parts[@]}
    
    for ((i=0; i<max_len; i++)); do
        local a_id="${a_parts[$i]:-}"
        local b_id="${b_parts[$i]:-}"
        
        # If one has more parts, it has higher precedence
        [[ -z "$a_id" && -n "$b_id" ]] && echo -1 && return
        [[ -n "$a_id" && -z "$b_id" ]] && echo 1 && return
        
        # Compare numeric vs alphanumeric
        local a_isnum=0 b_isnum=0
        [[ "$a_id" =~ ^[0-9]+$ ]] && a_isnum=1
        [[ "$b_id" =~ ^[0-9]+$ ]] && b_isnum=1
        
        if [[ $a_isnum -eq 1 && $b_isnum -eq 1 ]]; then
            # Numeric comparison
            [[ $a_id -lt $b_id ]] && echo -1 && return
            [[ $a_id -gt $b_id ]] && echo 1 && return
        elif [[ $a_isnum -eq 1 ]]; then
            # Numeric identifiers have lower precedence
            echo -1 && return
        elif [[ $b_isnum -eq 1 ]]; then
            echo 1 && return
        else
            # ASCII comparison
            [[ "$a_id" < "$b_id" ]] && echo -1 && return
            [[ "$a_id" > "$b_id" ]] && echo 1 && return
        fi
    done
    
    echo 0
}

# Compare two semantic versions
# Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2
semver_compare() {
    local v1="$1"
    local v2="$2"
    
    local v1_info v2_info
    v1_info=$(semver_parse "$v1")
    v2_info=$(semver_parse "$v2")
    
    local major1=$(echo "$v1_info" | grep "^major=" | cut -d'=' -f2)
    local minor1=$(echo "$v1_info" | grep "^minor=" | cut -d'=' -f2)
    local patch1=$(echo "$v1_info" | grep "^patch=" | cut -d'=' -f2)
    local pre1=$(echo "$v1_info" | grep "^prerelease=" | cut -d'=' -f2)
    
    local major2=$(echo "$v2_info" | grep "^major=" | cut -d'=' -f2)
    local minor2=$(echo "$v2_info" | grep "^minor=" | cut -d'=' -f2)
    local patch2=$(echo "$v2_info" | grep "^patch=" | cut -d'=' -f2)
    local pre2=$(echo "$v2_info" | grep "^prerelease=" | cut -d'=' -f2)
    
    # Compare major
    [[ $major1 -lt $major2 ]] && echo -1 && return
    [[ $major1 -gt $major2 ]] && echo 1 && return
    
    # Compare minor
    [[ $minor1 -lt $minor2 ]] && echo -1 && return
    [[ $minor1 -gt $minor2 ]] && echo 1 && return
    
    # Compare patch
    [[ $patch1 -lt $patch2 ]] && echo -1 && return
    [[ $patch1 -gt $patch2 ]] && echo 1 && return
    
    # Compare prerelease
    _prerelease_compare "$pre1" "$pre2"
}

# Check if version satisfies a constraint
# Supports: ^ (caret), ~ (tilde), = (exact), >=, <=, >, <
semver_satisfies() {
    local version="$1"
    local constraint="$2"
    
    # Remove leading 'v'
    version="${version#v}"
    constraint="${constraint#v}"
    
    # Handle wildcard
    [[ "$constraint" == "*" ]] && return 0
    
    # Handle caret (^) - compatible with version
    if [[ "$constraint" =~ ^\^(.+)$ ]]; then
        local caret_ver="${BASH_REMATCH[1]}"
        local caret_info=$(semver_parse "$caret_ver")
        local ver_info=$(semver_parse "$version")
        
        local c_major=$(echo "$caret_info" | grep "^major=" | cut -d'=' -f2)
        local c_minor=$(echo "$caret_info" | grep "^minor=" | cut -d'=' -f2)
        local v_major=$(echo "$ver_info" | grep "^major=" | cut -d'=' -f2)
        local v_minor=$(echo "$ver_info" | grep "^minor=" | cut -d'=' -f2)
        
        # ^0.0.x matches exact version only
        if [[ $c_major -eq 0 && $c_minor -eq 0 ]]; then
            [[ $(semver_compare "$version" "$caret_ver") -eq 0 ]]
            return
        fi
        
        # ^0.x matches compatible with 0.x (minor must match)
        if [[ $c_major -eq 0 ]]; then
            [[ $v_major -eq 0 && $v_minor -eq $c_minor && $(semver_compare "$version" "$caret_ver") -ge 0 ]]
            return
        fi
        
        # ^x matches compatible with major version x
        [[ $v_major -eq $c_major && $(semver_compare "$version" "$caret_ver") -ge 0 ]]
        return
    fi
    
    # Handle tilde (~) - approximately equivalent
    if [[ "$constraint" =~ ^~(.+)$ ]]; then
        local tilde_ver="${BASH_REMATCH[1]}"
        local tilde_info=$(semver_parse "$tilde_ver")
        local ver_info=$(semver_parse "$version")
        
        local t_major=$(echo "$tilde_info" | grep "^major=" | cut -d'=' -f2)
        local t_minor=$(echo "$tilde_info" | grep "^minor=" | cut -d'=' -f2)
        local v_major=$(echo "$ver_info" | grep "^major=" | cut -d'=' -f2)
        local v_minor=$(echo "$ver_info" | grep "^minor=" | cut -d'=' -f2)
        
        # ~1.2.3 allows >= 1.2.3, < 1.3.0
        # ~1.2 allows >= 1.2.0, < 1.3.0
        # ~1 allows >= 1.0.0, < 2.0.0
        [[ $v_major -eq $t_major && $v_minor -eq $t_minor && $(semver_compare "$version" "$tilde_ver") -ge 0 ]]
        return
    fi
    
    # Handle comparison operators (using string prefix checks)
    if [[ "$constraint" == ">="* ]]; then
        local cmp_ver="${constraint#>=}"
        [[ $(semver_compare "$version" "$cmp_ver") -ge 0 ]]
        return
    fi
    
    if [[ "$constraint" == ">"* ]]; then
        local cmp_ver="${constraint#>}"
        [[ $(semver_compare "$version" "$cmp_ver") -gt 0 ]]
        return
    fi
    
    if [[ "$constraint" == "<="* ]]; then
        local cmp_ver="${constraint#<=}"
        [[ $(semver_compare "$version" "$cmp_ver") -le 0 ]]
        return
    fi
    
    if [[ "$constraint" == "<"* ]]; then
        local cmp_ver="${constraint#<}"
        [[ $(semver_compare "$version" "$cmp_ver") -lt 0 ]]
        return
    fi
    
    # Handle exact version (= is optional)
    local exact_ver="${constraint#=}"
    [[ $(semver_compare "$version" "$exact_ver") -eq 0 ]]
}

# Find the highest version that satisfies a constraint
# Usage: semver_resolve "^1.0.0" "1.0.0" "1.0.1" "1.1.0" "2.0.0"
semver_resolve() {
    local constraint="$1"
    shift
    
    local best_version=""
    
    for version in "$@"; do
        if semver_satisfies "$version" "$constraint"; then
            if [[ -z "$best_version" ]] || [[ $(semver_compare "$version" "$best_version") -gt 0 ]]; then
                best_version="$version"
            fi
        fi
    done
    
    echo "$best_version"
}

# Validate if a string is a valid semantic version
semver_is_valid() {
    local version="$1"
    version="${version#v}"
    
    # Basic semver pattern: major.minor.patch[-prerelease][+build]
    [[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$ ]]
}

# Bump version components
semver_bump_major() {
    local info=$(semver_parse "$1")
    local major=$(echo "$info" | grep "^major=" | cut -d'=' -f2)
    echo "$((major + 1)).0.0"
}

semver_bump_minor() {
    local info=$(semver_parse "$1")
    local major=$(echo "$info" | grep "^major=" | cut -d'=' -f2)
    local minor=$(echo "$info" | grep "^minor=" | cut -d'=' -f2)
    echo "${major}.$((minor + 1)).0"
}

semver_bump_patch() {
    local info=$(semver_parse "$1")
    local major=$(echo "$info" | grep "^major=" | cut -d'=' -f2)
    local minor=$(echo "$info" | grep "^minor=" | cut -d'=' -f2)
    local patch=$(echo "$info" | grep "^patch=" | cut -d'=' -f2)
    echo "${major}.${minor}.$((patch + 1))"
}
