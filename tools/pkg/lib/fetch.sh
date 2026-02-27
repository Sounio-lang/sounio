#!/usr/bin/env bash
# Package fetching library for Sounio Package Manager

set -euo pipefail

# Cache directory for downloaded packages
PKG_CACHE_DIR="${HOME}/.cache/sounio/pkg"

# Initialize the package cache directory
# Usage: fetch_init
fetch_init() {
    if [[ ! -d "$PKG_CACHE_DIR" ]]; then
        mkdir -p "$PKG_CACHE_DIR"
        echo "Initialized package cache: $PKG_CACHE_DIR"
    fi
}

# Get cache directory (create if needed)
# Usage: _get_cache_dir
_get_cache_dir() {
    if [[ ! -d "$PKG_CACHE_DIR" ]]; then
        mkdir -p "$PKG_CACHE_DIR"
    fi
    echo "$PKG_CACHE_DIR"
}

# Calculate SHA256 hash of a directory or file
# Usage: fetch_calculate_hash "/path/to/pkg"
fetch_calculate_hash() {
    local target="$1"
    
    if [[ -d "$target" ]]; then
        # Hash all files in directory, sorted
        find "$target" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
    elif [[ -f "$target" ]]; then
        sha256sum "$target" | cut -d' ' -f1
    else
        echo ""
    fi
}

# Fetch a package from the registry
# Usage: fetch_from_registry "package-name" "1.0.0" "/dest/path"
fetch_from_registry() {
    local name="$1"
    local version="$2"
    local dest="$3"
    
    local cache_dir=$(_get_cache_dir)
    local cache_key="${name}-${version}"
    local cached_pkg="${cache_dir}/${cache_key}"
    
    echo "Fetching ${name}@${version} from registry..."
    
    # Check if already cached
    if [[ -d "$cached_pkg" ]]; then
        echo "  Using cached package"
        cp -r "$cached_pkg" "$dest"
        return 0
    fi
    
    # MOCK: Registry query - in production, this would query a real registry
    # For now, return error since we don't have a real registry
    echo "  Warning: Registry not configured, attempting mock fetch..."
    
    # Try to find in local stdlib
    local stdlib_path="stdlib/${name}"
    if [[ -d "$stdlib_path" ]]; then
        echo "  Found in local stdlib"
        mkdir -p "$cached_pkg"
        cp -r "${stdlib_path}/"* "$cached_pkg/" 2>/dev/null || true
        cp -r "$cached_pkg" "$dest"
        return 0
    fi
    
    echo "  Error: Package ${name}@${version} not found in registry"
    return 1
}

# Fetch a package from a Git repository
# Usage: fetch_from_git "https://github.com/user/repo.git" "main" "/dest/path"
fetch_from_git() {
    local url="$1"
    local branch="${2:-main}"
    local dest="$3"
    
    local cache_dir=$(_get_cache_dir)
    local cache_key
    cache_key=$(echo -n "${url}" | sha256sum | cut -d' ' -f1 | head -c 16)
    local cached_repo="${cache_dir}/git-${cache_key}"
    
    echo "Fetching from git: ${url}@${branch}..."
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        echo "  Error: git is not installed"
        return 1
    fi
    
    # Clone or update cached repository
    if [[ -d "${cached_repo}/.git" ]]; then
        echo "  Updating cached repository..."
        (cd "$cached_repo" && git fetch origin "$branch" && git checkout "$branch" && git pull origin "$branch")
    else
        echo "  Cloning repository..."
        git clone --depth 1 --branch "$branch" "$url" "$cached_repo"
    fi
    
    # Copy to destination
    rm -rf "$dest"
    cp -r "$cached_repo" "$dest"
    
    # Remove .git directory from destination
    rm -rf "${dest}/.git"
    
    echo "  Fetched to ${dest}"
    return 0
}

# Fetch a package from a local path
# Usage: fetch_from_path "/path/to/package" "/dest/path"
fetch_from_path() {
    local source_path="$1"
    local dest="$2"
    
    echo "Fetching from path: ${source_path}..."
    
    # Resolve relative paths
    if [[ ! "$source_path" = /* ]]; then
        source_path="$(pwd)/${source_path}"
    fi
    
    if [[ ! -d "$source_path" ]]; then
        echo "  Error: Path does not exist: ${source_path}"
        return 1
    fi
    
    # Use rsync if available, otherwise cp
    rm -rf "$dest"
    if command -v rsync &> /dev/null; then
        rsync -a --exclude='.git' --exclude='vendor' "$source_path/" "$dest/"
    else
        mkdir -p "$dest"
        cp -r "${source_path}/"* "$dest/" 2>/dev/null || true
        cp -r "${source_path}/."* "$dest/" 2>/dev/null || true
        rm -rf "${dest}/.git" 2>/dev/null || true
    fi
    
    echo "  Copied to ${dest}"
    return 0
}

# Verify package hash
# Usage: fetch_verify "/path/to/package" "expected_hash"
fetch_verify() {
    local pkg_dir="$1"
    local expected_hash="$2"
    
    if [[ -z "$expected_hash" ]]; then
        echo "  Warning: No hash specified, skipping verification"
        return 0
    fi
    
    echo "Verifying package hash..."
    
    local actual_hash
    actual_hash=$(fetch_calculate_hash "$pkg_dir")
    
    if [[ "$actual_hash" == "$expected_hash" ]]; then
        echo "  Hash verified: ${actual_hash:0:16}..."
        return 0
    else
        echo "  Error: Hash mismatch!"
        echo "    Expected: ${expected_hash:0:16}..."
        echo "    Actual:   ${actual_hash:0:16}..."
        return 1
    fi
}

# Extract a tar.gz archive
# Usage: fetch_extract "/path/to/archive.tar.gz" "/dest/dir"
fetch_extract() {
    local archive="$1"
    local dest="$2"
    
    echo "Extracting ${archive}..."
    
    if [[ ! -f "$archive" ]]; then
        echo "  Error: Archive not found: ${archive}"
        return 1
    fi
    
    mkdir -p "$dest"
    tar -xzf "$archive" -C "$dest" --strip-components=1
    
    echo "  Extracted to ${dest}"
    return 0
}

# Download a file via HTTP
# Usage: fetch_download "https://example.com/pkg.tar.gz" "/dest/path.tar.gz"
fetch_download() {
    local url="$1"
    local dest="$2"
    
    echo "Downloading ${url}..."
    
    if ! command -v curl &> /dev/null; then
        echo "  Error: curl is not installed"
        return 1
    fi
    
    mkdir -p "$(dirname "$dest")"
    
    if curl -fsSL "$url" -o "$dest" 2>/dev/null; then
        echo "  Downloaded to ${dest}"
        return 0
    else
        echo "  Error: Failed to download ${url}"
        return 1
    fi
}

# Clean the package cache
# Usage: fetch_clean_cache
fetch_clean_cache() {
    if [[ -d "$PKG_CACHE_DIR" ]]; then
        echo "Cleaning package cache: ${PKG_CACHE_DIR}"
        rm -rf "${PKG_CACHE_DIR:?}/"*
        echo "  Cache cleaned"
    else
        echo "Cache directory does not exist"
    fi
}

# Get cache size
# Usage: fetch_cache_size
fetch_cache_size() {
    if [[ -d "$PKG_CACHE_DIR" ]]; then
        du -sh "$PKG_CACHE_DIR" | cut -f1
    else
        echo "0B"
    fi
}
