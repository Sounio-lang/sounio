#!/usr/bin/env bash
#
# Semantic versioning utilities for Sounio package manager
# Supports semver 2.0.0 with constraint operators: ^, ~, =, >=, <=, >, <
#

set -euo pipefail

# Parse a semantic version string
# Usage: semver_parse "1.2.3"
# Returns: major=1 minor=2 patch=3 prerelease="" build=""
semver_parse() {
    local version="$1"
    local major=0 minor=0 patch=0
    local prerelease="" build=""

    # Remove leading 'v' if present
    version="${version#v}"

    # Extract build metadata (after +)
    if [[ "$version" =~ ^([^+]+)\+(.+)$ ]]; then
        version="${BASH_REMATCH[1]}"
        build="${BASH_REMATCH[2]}"
    fi

    # Extract prerelease (after -)
    if [[ "$version" =~ ^([^-]+)-(.+)$ ]]; then
        version="${BASH_REMATCH[1]}"
        prerelease="${BASH_REMATCH[2]}"
    fi

    # Parse major.minor.patch
    IFS='.' read -r major minor patch <<< "$version"

    # Handle short versions
    [[ -z "$minor" ]] && minor=0
    [[ -z "$patch" ]] && patch=0

    printf 'major=%s\nminor=%s\npatch=%s\nprerelease=%s\nbuild=%s\n' \
        "$major" "$minor" "$patch" "$prerelease" "$build"
}

# Get a component from parsed version
semver_get() {
    local parsed="$1"
    local component="$2"

    grep "^${component}=" <<< "$parsed" | cut -d'=' -f2-
}

# Compare two semantic versions
# Usage: semver_compare "1.2.3" "1.2.4"
# Returns: -1 (v1 < v2), 0 (v1 = v2), 1 (v1 > v2)
semver_compare() {
    local v1="$1" v2="$2"
    local p1 p2 major1 major2 minor1 minor2 patch1 patch2 pre1 pre2

    p1=$(semver_parse "$v1")
    p2=$(semver_parse "$v2")

    major1=$(semver_get "$p1" "major")
    major2=$(semver_get "$p2" "major")
    minor1=$(semver_get "$p1" "minor")
    minor2=$(semver_get "$p2" "minor")
    patch1=$(semver_get "$p1" "patch")
    patch2=$(semver_get "$p2" "patch")
    pre1=$(semver_get "$p1" "prerelease")
    pre2=$(semver_get "$p2" "prerelease")

    # Compare major
    if (( major1 < major2 )); then echo -1; return; fi
    if (( major1 > major2 )); then echo 1; return; fi

    # Compare minor
    if (( minor1 < minor2 )); then echo -1; return; fi
    if (( minor1 > minor2 )); then echo 1; return; fi

    # Compare patch
    if (( patch1 < patch2 )); then echo -1; return; fi
    if (( patch1 > patch2 )); then echo 1; return; fi

    # Compare prerelease
    # A version without prerelease is greater than one with prerelease
    if [[ -z "$pre1" && -n "$pre2" ]]; then echo 1; return; fi
    if [[ -n "$pre1" && -z "$pre2" ]]; then echo -1; return; fi
    if [[ -n "$pre1" && -n "$pre2" ]]; then
        local cmp
        cmp=$(semver_compare_prerelease "$pre1" "$pre2")
        [[ "$cmp" != "0" ]] && { echo "$cmp"; return; }
    fi

    echo 0
}

# Compare prerelease identifiers
semver_compare_prerelease() {
    local pre1="$1" pre2="$2"
    local id1 id2

    local IFS='.'
    read -ra ids1 <<< "$pre1"
    read -ra ids2 <<< "$pre2"

    local len1=${#ids1[@]} len2=${#ids2[@]}
    local max=$(( len1 > len2 ? len1 : len2 ))

    for (( i=0; i<max; i++ )); do
        id1="${ids1[$i]:-}"
        id2="${ids2[$i]:-}"

        # Missing identifier is less than any identifier
        [[ -z "$id1" ]] && { echo -1; return; }
        [[ -z "$id2" ]] && { echo 1; return; }

        # Numeric identifiers are compared as integers
        if [[ "$id1" =~ ^[0-9]+$ && "$id2" =~ ^[0-9]+$ ]]; then
            if (( id1 < id2 )); then echo -1; return; fi
            if (( id1 > id2 )); then echo 1; return; fi
        else
            # Alphanumeric identifiers are compared lexically
            if [[ "$id1" < "$id2" ]]; then echo -1; return; fi
            if [[ "$id1" > "$id2" ]]; then echo 1; return; fi
        fi
    done

    echo 0
}

# Check if a version satisfies a constraint
# Usage: semver_satisfies "1.2.3" "^1.0.0"
# Supported constraints:
#   ^1.2.3 - caret, compatible with major version
#   ~1.2.3 - tilde, compatible with minor version  
#   =1.2.3 - exact
#   >=1.2.3, <=1.2.3, >1.2.3, <1.2.3 - ranges
#   1.2.3 || 2.0.0 - OR constraints
semver_satisfies() {
    local version="$1"
    local constraint="$2"

    # Handle OR constraints
    if [[ "$constraint" =~ \|\| ]]; then
        local IFS='||'
        read -ra parts <<< "$constraint"
        for part in "${parts[@]}"; do
            part="${part#"${part%%[![:space:]]*}"}"
            part="${part%"${part##*[![:space:]]}"}"
            if semver_satisfies "$version" "$part"; then
                return 0
            fi
        done
        return 1
    fi

    # Handle hyphen ranges: 1.2.3 - 2.3.4
    if [[ "$constraint" =~ ^([0-9.]+)[[:space:]]*-[[:space:]]*([0-9.]+)$ ]]; then
        local low="${BASH_REMATCH[1]}"
        local high="${BASH_REMATCH[2]}"
        if (( $(semver_compare "$version" "$low") >= 0 )) && \
           (( $(semver_compare "$version" "$high") <= 0 )); then
            return 0
        fi
        return 1
    fi

    # Parse operator and version
    local op=""
    local ver=""

    # Handle multi-char operators first (>=, <=, >, <)
    if [[ "$constraint" =~ ^(>=|<=|>|<)([0-9].*)$ ]]; then
        op="${BASH_REMATCH[1]}"
        ver="${BASH_REMATCH[2]}"
    # Then single-char operators (start with ^ or ~)
    elif [[ "$constraint" =~ ^(\^)([0-9].*)$ ]]; then
        op="^"
        ver="${BASH_REMATCH[2]}"
    elif [[ "$constraint" =~ ^(~)([0-9].*)$ ]]; then
        op="~"
        ver="${BASH_REMATCH[2]}"
    elif [[ "$constraint" =~ ^([0-9]+\.[0-9]+\.[0-9]+.*)$ ]]; then
        # Bare version = exact match
        op="="
        ver="$constraint"
    else
        return 1
    fi

    # Remove trailing whitespace from version
    ver="${ver%"${ver##*[![:space:]]}"}"

    case "$op" in
        "=")
            [[ $(semver_compare "$version" "$ver") == "0" ]]
            ;;
        "^")
            semver_satisfies_caret "$version" "$ver"
            ;;
        "~")
            semver_satisfies_tilde "$version" "$ver"
            ;;
        ">=")
            (( $(semver_compare "$version" "$ver") >= 0 ))
            ;;
        "<=")
            (( $(semver_compare "$version" "$ver") <= 0 ))
            ;;
        ">")
            (( $(semver_compare "$version" "$ver") > 0 ))
            ;;
        "<")
            (( $(semver_compare "$version" "$ver") < 0 ))
            ;;
        *)
            return 1
            ;;
    esac
}

# Check if version satisfies caret constraint (^x.y.z)
# Allows changes that do not modify the left-most non-zero digit
semver_satisfies_caret() {
    local version="$1" target="$2"
    local pv tv major minor patch

    pv=$(semver_parse "$version")
    tv=$(semver_parse "$target")

    local major_t minor_t patch_t
    major_t=$(semver_get "$tv" "major")
    minor_t=$(semver_get "$tv" "minor")
    patch_t=$(semver_get "$tv" "patch")

    local major_v minor_v patch_v
    major_v=$(semver_get "$pv" "major")
    minor_v=$(semver_get "$pv" "minor")
    patch_v=$(semver_get "$pv" "patch")

    # If major > 0, compatible with same major
    if (( major_t > 0 )); then
        (( major_v == major_t )) || return 1
        (( minor_v > minor_t )) || { (( minor_v == minor_t )) && (( patch_v >= patch_t )); } || return 1
        return 0
    fi

    # If major = 0, minor > 0: compatible with same minor
    if (( minor_t > 0 )); then
        (( minor_v == minor_t )) || return 1
        (( patch_v >= patch_t )) || return 1
        return 0
    fi

    # If major = 0, minor = 0: only same patch
    (( patch_v == patch_t ))
}

# Check if version satisfies tilde constraint (~x.y.z)
# Allows patch-level changes if minor version is specified
semver_satisfies_tilde() {
    local version="$1" target="$2"
    local pv tv

    pv=$(semver_parse "$version")
    tv=$(semver_parse "$target")

    local major_t minor_t patch_t
    major_t=$(semver_get "$tv" "major")
    minor_t=$(semver_get "$tv" "minor")
    patch_t=$(semver_get "$tv" "patch")

    local major_v minor_v patch_v
    major_v=$(semver_get "$pv" "major")
    minor_v=$(semver_get "$pv" "minor")
    patch_v=$(semver_get "$pv" "patch")

    (( major_v == major_t )) || return 1
    (( minor_v == minor_t )) || return 1
    (( patch_v >= patch_t )) || return 1
}

# Find the highest version that satisfies a constraint
# Usage: semver_resolve "^1.0.0" "1.0.0" "1.0.1" "1.1.0" "2.0.0"
# Returns: the highest matching version or empty if none
semver_resolve() {
    local constraint="$1"
    shift

    local best=""
    local version

    for version in "$@"; do
        if semver_satisfies "$version" "$constraint"; then
            if [[ -z "$best" ]] || (( $(semver_compare "$version" "$best") > 0 )); then
                best="$version"
            fi
        fi
    done

    printf '%s\n' "$best"
}

# Get the minimum version from a constraint
# Usage: semver_min_version "^1.2.3"
semver_min_version() {
    local constraint="$1"

    if [[ "$constraint" =~ ^[=\^~]+([0-9]+\.[0-9]+\.[0-9]+.*)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    return 1
}

# Validate that a string is a valid semantic version
semver_validate() {
    local version="$1"

    # Remove leading 'v'
    version="${version#v}"

    # Basic semver regex
    [[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-([0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*))?(\+([0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*))?$ ]]
}

# Example usage:
#   semver_compare "1.2.3" "1.2.4"  # Outputs: -1
#   semver_satisfies "1.2.5" "^1.0.0" && echo "satisfies"
#   best=$(semver_resolve "~1.2.0" "1.2.0" "1.2.5" "1.3.0")  # Returns: 1.2.5
