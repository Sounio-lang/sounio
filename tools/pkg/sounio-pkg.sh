#!/usr/bin/env bash
#
# Sounio Package Manager
# Main CLI for dependency management
#

set -euo pipefail

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source libraries
source "${SCRIPT_DIR}/lib/parse_toml.sh"
source "${SCRIPT_DIR}/lib/semver.sh"
source "${SCRIPT_DIR}/lib/lockfile.sh"
source "${SCRIPT_DIR}/lib/fetch.sh"

# Configuration
readonly PKG_VERSION="0.1.0"
readonly VENDOR_DIR="vendor"
readonly MANIFEST_FILE="Sounio.toml"
readonly LOCKFILE="Sounio.lock"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}info:${NC} $*"; }
log_success() { echo -e "${GREEN}success:${NC} $*"; }
log_warn() { echo -e "${YELLOW}warning:${NC} $*"; }
log_error() { echo -e "${RED}error:${NC} $*"; }

# Show usage
usage() {
    cat <<EOF
Sounio Package Manager v${PKG_VERSION}

Usage: sounio-pkg <command> [options]

Commands:
    install          Install dependencies from lockfile
    update           Update dependencies and generate lockfile
    add <pkg>        Add a package dependency (stub)
    remove <pkg>     Remove a package dependency (stub)
    verify           Verify package hashes
    clean            Remove vendor/ directory and cache
    help             Show this help message

Options:
    -v, --verbose    Verbose output
    -h, --help       Show help

Examples:
    sounio-pkg install
    sounio-pkg update
    sounio-pkg verify
    sounio-pkg clean
EOF
}

# Parse manifest and resolve dependencies
parse_manifest() {
    local file="$1"

    if [[ ! -f "$file" ]]; then
        log_error "Manifest file not found: $file"
        return 1
    fi

    toml_parse "$file"
}

# Install dependencies from lockfile
cmd_install() {
    log_info "Installing dependencies..."

    if [[ ! -f "$LOCKFILE" ]]; then
        log_warn "Lockfile not found, running update..."
        cmd_update
        return 0
    fi

    if ! lockfile_is_fresh "$MANIFEST_FILE" "$LOCKFILE"; then
        log_warn "Lockfile is outdated, run 'sounio-pkg update'"
    fi

    # Initialize cache
    fetch_init

    # Read lockfile and install each package
    local in_package=false
    local pkg_name="" pkg_version="" pkg_source="" pkg_hash=""
    local installed=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^version ]] && continue

        # Start of package
        if [[ "$line" == "[[package]]" ]]; then
            # Install previous package
            if [[ -n "$pkg_name" ]]; then
                install_package "$pkg_name" "$pkg_version" "$pkg_source" "$pkg_hash" && \
                    ((installed++))
            fi
            in_package=true
            pkg_name=""
            pkg_version=""
            pkg_source=""
            pkg_hash=""
            continue
        fi

        # Parse fields
        if [[ "$line" =~ ^name[[:space:]]*=[[:space:]]*\"(.+)\"$ ]]; then
            pkg_name="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^version[[:space:]]*=[[:space:]]*\"(.+)\"$ ]]; then
            pkg_version="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^source[[:space:]]*=[[:space:]]*\"(.+)\"$ ]]; then
            pkg_source="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^hash[[:space:]]*=[[:space:]]*\"(.+)\"$ ]]; then
            pkg_hash="${BASH_REMATCH[1]}"
        fi
    done < "$LOCKFILE"

    # Install last package
    if [[ -n "$pkg_name" ]]; then
        install_package "$pkg_name" "$pkg_version" "$pkg_source" "$pkg_hash" && \
            ((installed++))
    fi

    log_success "Installed $installed packages to $VENDOR_DIR/"
}

# Install a single package
install_package() {
    local name="$1"
    local version="$2"
    local source="$3"
    local hash="$4"
    local dest="$VENDOR_DIR/$name"

    log_info "Installing $name@$version"

    # Determine source type and fetch
    if [[ "$source" == registry://* ]]; then
        local pkg_name="${source#registry://}"
        fetch_from_registry "$pkg_name" "$version" "$dest" || {
            log_error "Failed to fetch $name from registry"
            return 1
        }
    elif [[ "$source" == git+* ]]; then
        local url="${source#git+}"
        local branch="$version"
        fetch_from_git "$url" "$branch" "$dest" || {
            log_error "Failed to clone $name"
            return 1
        }
    elif [[ "$source" == path://* ]]; then
        local path="${source#path://}"
        fetch_from_path "$path" "$dest" || {
            log_error "Failed to copy $name"
            return 1
        }
    else
        log_error "Unknown source type: $source"
        return 1
    fi

    # Verify hash if provided
    if [[ -n "$hash" ]]; then
        if ! fetch_verify "$dest" "$hash"; then
            log_error "Hash verification failed for $name"
            rm -rf "$dest"
            return 1
        fi
    fi

    return 0
}

# Update dependencies and generate lockfile
cmd_update() {
    log_info "Updating dependencies..."

    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_error "Manifest file not found: $MANIFEST_FILE"
        return 1
    fi

    # Parse manifest
    local parsed
    parsed=$(parse_manifest "$MANIFEST_FILE")

    # Initialize cache
    fetch_init

    # Build lockfile entries from dependencies
    local entries=()
    local dep_name dep_version dep_source dep_hash

    # Read dependencies from manifest
    while IFS='=' read -r key value; do
        [[ "$key" == dependencies.* ]] || continue
        local dep="${key#dependencies.}"

        # Check if inline table or simple version
        if [[ "$value" == \{*\} ]]; then
            # Parse inline table
            while IFS='=' read -r k v; do
                case "$k" in
                    version) dep_version="$v" ;;
                    git) dep_source="git+$v" ;;
                    path) dep_source="path://$v" ;;
                    *) dep_version="$v" ;;
                esac
            done < <(toml_parse_inline_table "$value")
        else
            dep_version="$value"
            dep_source="registry://$dep"
        fi

        dep_name="$dep"
        dep_hash=""  # Will be computed after fetch

        # Resolve version if needed
        if [[ "$dep_version" == *"^"* ]] || [[ "$dep_version" == *"~"* ]] || [[ "$dep_version" == *">"* ]] || [[ "$dep_version" == *"<"* ]] || [[ "$dep_version" == *"="* ]]; then
            log_info "Resolving $dep_name $dep_version"
            # In real implementation, query registry for available versions
            # For now, use constraint as version
            dep_version=$(semver_min_version "$dep_version" || echo "$dep_version")
        fi

        entries+=("$dep_name $dep_version $dep_source $dep_hash")
        log_info "Resolved $dep_name@$dep_version"
    done <<< "$parsed"

    # Write lockfile
    lockfile_write "$LOCKFILE" "${entries[@]}"

    log_success "Generated $LOCKFILE with ${#entries[@]} packages"
    log_info "Run 'sounio-pkg install' to download packages"
}

# Add a package (stub)
cmd_add() {
    local pkg="${1:-}"

    if [[ -z "$pkg" ]]; then
        log_error "Package name required"
        echo "Usage: sounio-pkg add <package>[@version]"
        return 1
    fi

    log_warn "Add command is a stub - not yet implemented"
    log_info "Would add: $pkg"

    # Parse package[@version]
    local name version
    if [[ "$pkg" =~ @ ]]; then
        name="${pkg%%@*}"
        version="${pkg#*@}"
    else
        name="$pkg"
        version="*"
    fi

    log_info "Package: $name, Version: $version"
}

# Remove a package (stub)
cmd_remove() {
    local pkg="${1:-}"

    if [[ -z "$pkg" ]]; then
        log_error "Package name required"
        echo "Usage: sounio-pkg remove <package>"
        return 1
    fi

    log_warn "Remove command is a stub - not yet implemented"
    log_info "Would remove: $pkg"
}

# Verify package hashes
cmd_verify() {
    log_info "Verifying package integrity..."

    if [[ ! -d "$VENDOR_DIR" ]]; then
        log_error "Vendor directory not found: $VENDOR_DIR"
        return 1
    fi

    if [[ ! -f "$LOCKFILE" ]]; then
        log_error "Lockfile not found: $LOCKFILE"
        return 1
    fi

    local verified=0
    local failed=0

    for pkg_dir in "$VENDOR_DIR"/*; do
        [[ -d "$pkg_dir" ]] || continue

        local name
        name=$(basename "$pkg_dir")

        local expected_hash
        expected_hash=$(lockfile_get_source "$LOCKFILE" "$name" 2>/dev/null || echo "")
        expected_hash=$(grep -A5 "name = \"$name\"" "$LOCKFILE" 2>/dev/null | \
                        grep "^hash" | sed 's/.*= "\(.*\)".*/\1/')

        if [[ -z "$expected_hash" ]]; then
            log_warn "No hash for $name"
            continue
        fi

        if fetch_verify "$pkg_dir" "$expected_hash" 2>/dev/null; then
            log_success "$name verified"
            ((verified++))
        else
            log_error "$name verification failed"
            ((failed++))
        fi
    done

    echo ""
    log_info "Verified: $verified, Failed: $failed"

    (( failed == 0 )) || return 1
}

# Clean vendor directory and cache
cmd_clean() {
    log_info "Cleaning up..."

    if [[ -d "$VENDOR_DIR" ]]; then
        rm -rf "$VENDOR_DIR"
        log_success "Removed $VENDOR_DIR/"
    fi

    if [[ -n "${1:-}" && "$1" == "--all" ]]; then
        if [[ -d "$PKG_CACHE_DIR" ]]; then
            rm -rf "$PKG_CACHE_DIR"
            log_success "Removed cache directory"
        fi
        if [[ -f "$LOCKFILE" ]]; then
            rm -f "$LOCKFILE"
            log_success "Removed $LOCKFILE"
        fi
    fi

    log_success "Clean complete"
}

# Main entry point
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        install)
            cmd_install
            ;;
        update)
            cmd_update
            ;;
        add)
            cmd_add "$@"
            ;;
        remove|rm)
            cmd_remove "$@"
            ;;
        verify)
            cmd_verify
            ;;
        clean)
            cmd_clean "$@"
            ;;
        help|--help|-h)
            usage
            ;;
        version|--version|-v)
            echo "sounio-pkg v${PKG_VERSION}"
            ;;
        *)
            log_error "Unknown command: $cmd"
            usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
