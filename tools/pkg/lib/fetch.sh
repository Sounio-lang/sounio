#!/usr/bin/env bash
#
# Package fetching utilities for Sounio package manager
# Handles registry, git, and path sources
#

set -euo pipefail

# Configuration
PKG_CACHE_DIR="${PKG_CACHE_DIR:-$HOME/.cache/sounio/pkg}"
PKG_REGISTRY_URL="${PKG_REGISTRY_URL:-https://packages.souniolang.org}"
PKG_TIMEOUT="${PKG_TIMEOUT:-60}"

# Initialize cache directory
# Usage: fetch_init
fetch_init() {
    if [[ ! -d "$PKG_CACHE_DIR" ]]; then
        mkdir -p "$PKG_CACHE_DIR"
    fi

    # Create subdirectories
    mkdir -p "$PKG_CACHE_DIR/registry"
    mkdir -p "$PKG_CACHE_DIR/git"
    mkdir -p "$PKG_CACHE_DIR/tmp"
}

# Fetch a package from the registry
# Usage: fetch_from_registry "name" "version" "destination"
# Returns: 0 on success, 1 on failure
fetch_from_registry() {
    local name="$1"
    local version="$2"
    local dest="$3"

    local cache_path="$PKG_CACHE_DIR/registry/${name}-${version}.tar.gz"
    local url="${PKG_REGISTRY_URL}/${name}/${version}.tar.gz"

    # Check cache first
    if [[ -f "$cache_path" ]]; then
        echo "Using cached: $cache_path" >&2
        cp "$cache_path" "$dest.tar.gz"
        tar -xzf "$dest.tar.gz" -C "$dest" --strip-components=1
        rm "$dest.tar.gz"
        return 0
    fi

    # Download to cache
    echo "Fetching: $url" >&2
    if ! curl -fsSL --connect-timeout "$PKG_TIMEOUT" \
              --max-time "$((PKG_TIMEOUT * 2))" \
              -o "$cache_path.tmp" "$url" 2>/dev/null; then
        rm -f "$cache_path.tmp"
        echo "Failed to download: $url" >&2
        return 1
    fi

    mv "$cache_path.tmp" "$cache_path"

    # Extract to destination
    mkdir -p "$dest"
    tar -xzf "$cache_path" -C "$dest" --strip-components=1

    return 0
}

# Fetch a package from a git repository
# Usage: fetch_from_git "url" "branch/tag" "destination"
fetch_from_git() {
    local url="$1"
    local ref="${2:-main}"
    local dest="$3"

    local cache_name
    cache_name=$(echo "$url" | sed 's/[^a-zA-Z0-9]/_/g')
    local cache_path="$PKG_CACHE_DIR/git/${cache_name}"

    # Clone or update cached repository
    if [[ ! -d "$cache_path/.git" ]]; then
        echo "Cloning: $url" >&2
        rm -rf "$cache_path"
        if ! git clone --quiet --bare "$url" "$cache_path" 2>/dev/null; then
            echo "Failed to clone: $url" >&2
            return 1
        fi
    else
        # Update cached repository
        echo "Updating cached repository" >&2
        git --git-dir="$cache_path" fetch --quiet origin 2>/dev/null || true
    fi

    # Checkout to destination
    rm -rf "$dest"
    mkdir -p "$dest"

    if ! git --git-dir="$cache_path" --work-tree="$dest" checkout -f "$ref" 2>/dev/null; then
        echo "Failed to checkout ref: $ref" >&2
        return 1
    fi

    # Remove .git directory from destination
    rm -rf "$dest/.git"

    return 0
}

# Fetch a package from a local path
# Usage: fetch_from_path "source_path" "destination"
fetch_from_path() {
    local source="$1"
    local dest="$2"

    # Resolve relative paths
    if [[ ! "$source" = /* ]]; then
        source="$(pwd)/$source"
    fi

    if [[ ! -d "$source" ]]; then
        echo "Source path does not exist: $source" >&2
        return 1
    fi

    echo "Copying from: $source" >&2

    # Remove destination if it exists
    rm -rf "$dest"
    mkdir -p "$dest"

    # Copy files (excluding common non-package files)
    rsync -a --exclude='.git' --exclude='.gitignore' \
             --exclude='*.md' --exclude='LICENSE' \
             --exclude='.github' --exclude='tests' \
             "$source/" "$dest/" 2>/dev/null || \
        cp -r "$source/"* "$dest/" 2>/dev/null || {
            echo "Failed to copy from: $source" >&2
            return 1
        }

    return 0
}

# Verify package integrity using hash
# Usage: fetch_verify "directory" "expected_hash"
# Hash format: sha256:abc123 or just abc123 (assumes sha256)
fetch_verify() {
    local dir="$1"
    local expected="$2"
    local hash_type="sha256"

    # Parse hash type if specified
    if [[ "$expected" =~ ^([a-z0-9]+):(.*)$ ]]; then
        hash_type="${BASH_REMATCH[1]}"
        expected="${BASH_REMATCH[2]}"
    fi

    # Compute actual hash
    local actual=""
    case "$hash_type" in
        sha256)
            if command -v sha256sum &>/dev/null; then
                actual=$(find "$dir" -type f -exec sha256sum {} \; | \
                         sort | sha256sum | cut -d' ' -f1)
            elif command -v shasum &>/dev/null; then
                actual=$(find "$dir" -type f -exec shasum -a 256 {} \; | \
                         sort | shasum -a 256 | cut -d' ' -f1)
            else
                echo "No SHA-256 tool available" >&2
                return 1
            fi
            ;;
        sha1)
            if command -v sha1sum &>/dev/null; then
                actual=$(find "$dir" -type f -exec sha1sum {} \; | \
                         sort | sha1sum | cut -d' ' -f1)
            elif command -v shasum &>/dev/null; then
                actual=$(find "$dir" -type f -exec shasum -a 1 {} \; | \
                         sort | shasum -a 1 | cut -d' ' -f1)
            else
                echo "No SHA-1 tool available" >&2
                return 1
            fi
            ;;
        md5)
            if command -v md5sum &>/dev/null; then
                actual=$(find "$dir" -type f -exec md5sum {} \; | \
                         sort | md5sum | cut -d' ' -f1)
            elif command -v md5 &>/dev/null; then
                actual=$(find "$dir" -type f -exec md5 {} \; | \
                         sort | md5 | cut -d' ' -f1)
            else
                echo "No MD5 tool available" >&2
                return 1
            fi
            ;;
        *)
            echo "Unsupported hash type: $hash_type" >&2
            return 1
            ;;
    esac

    if [[ "$actual" != "$expected" ]]; then
        echo "Hash mismatch!" >&2
        echo "  Expected: $expected" >&2
        echo "  Actual:   $actual" >&2
        return 1
    fi

    return 0
}

# Compute hash for a directory
# Usage: fetch_compute_hash "directory" [hash_type]
fetch_compute_hash() {
    local dir="$1"
    local hash_type="${2:-sha256}"

    local hash=""
    case "$hash_type" in
        sha256)
            if command -v sha256sum &>/dev/null; then
                hash=$(find "$dir" -type f -exec sha256sum {} \; | \
                       sort | sha256sum | cut -d' ' -f1)
            elif command -v shasum &>/dev/null; then
                hash=$(find "$dir" -type f -exec shasum -a 256 {} \; | \
                       sort | shasum -a 256 | cut -d' ' -f1)
            fi
            ;;
        *)
            echo "Unsupported hash type: $hash_type" >&2
            return 1
            ;;
    esac

    printf '%s:%s\n' "$hash_type" "$hash"
}

# Clean cache
# Usage: fetch_clean_cache [days]
fetch_clean_cache() {
    local days="${1:-30}"

    if [[ -d "$PKG_CACHE_DIR" ]]; then
        echo "Cleaning cache older than $days days..." >&2
        find "$PKG_CACHE_DIR" -type f -mtime +$days -delete
        find "$PKG_CACHE_DIR" -type d -empty -delete
    fi
}

# Get cache size
# Usage: fetch_cache_size
fetch_cache_size() {
    if [[ -d "$PKG_CACHE_DIR" ]]; then
        du -sh "$PKG_CACHE_DIR" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Example usage:
#   fetch_init
#   fetch_from_registry "stdlib" "1.0.0" "./vendor/stdlib"
#   fetch_from_git "https://github.com/sounio/stdlib.git" "v1.0.0" "./vendor/stdlib"
#   fetch_verify "./vendor/stdlib" "sha256:abc123"
