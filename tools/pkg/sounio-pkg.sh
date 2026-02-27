#!/usr/bin/env bash
# Sounio Package Manager
# Main CLI for managing Sounio dependencies

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source libraries
source "${SCRIPT_DIR}/lib/parse_toml.sh"
source "${SCRIPT_DIR}/lib/semver.sh"
source "${SCRIPT_DIR}/lib/lockfile.sh"
source "${SCRIPT_DIR}/lib/fetch.sh"

# Default configuration
MANIFEST_FILE="Sounio.toml"
LOCKFILE="Sounio.lock"
VENDOR_DIR="vendor"

# Colors for output (disable if not TTY)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

# Print usage information
usage() {
    cat << EOF
Sounio Package Manager

Usage: sounio-pkg [command] [options]

Commands:
    install          Install dependencies from lockfile (or update if stale)
    update           Generate new lockfile from manifest
    add <pkg>        Add a dependency (stub: edit Sounio.toml manually)
    remove <pkg>     Remove a dependency (stub: edit Sounio.toml manually)
    verify           Verify package hashes
    clean            Remove vendor and cache directories
    help             Show this help message

Examples:
    sounio-pkg install
    sounio-pkg update
    sounio-pkg verify
    sounio-pkg clean

EOF
}

# Print error message and exit
error() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

# Print info message
info() {
    echo -e "${BLUE}Info:${NC} $1"
}

# Print success message
success() {
    echo -e "${GREEN}Success:${NC} $1"
}

# Print warning message
warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Check if we're in a Sounio project
check_project() {
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        error "No ${MANIFEST_FILE} found. Are you in a Sounio project directory?"
    fi
}

# Install command - install from lockfile
# If lockfile is stale or missing, run update first
cmd_install() {
    check_project
    
    info "Installing dependencies..."
    
    # Check if lockfile is fresh
    if ! lockfile_is_fresh "$MANIFEST_FILE" "$LOCKFILE"; then
        warn "Lockfile is missing or outdated, running update..."
        cmd_update
        return 0
    fi
    
    # Create vendor directory
    mkdir -p "$VENDOR_DIR"
    
    # Read lockfile and install each package
    local installed=0
    while IFS='|' read -r name version source hash; do
        info "Installing ${name}@${version}..."
        
        local dest="${VENDOR_DIR}/${name}"
        
        # Determine source type and fetch
        if [[ -z "$source" || "$source" == "registry" ]]; then
            if ! fetch_from_registry "$name" "$version" "$dest"; then
                error "Failed to fetch ${name} from registry"
            fi
        elif [[ "$source" == git:* ]]; then
            local git_url="${source#git:}"
            local branch="main"
            if [[ "$git_url" == *#* ]]; then
                branch="${git_url#*#}"
                git_url="${git_url%#*}"
            fi
            if ! fetch_from_git "$git_url" "$branch" "$dest"; then
                error "Failed to fetch ${name} from git"
            fi
        elif [[ "$source" == path:* ]]; then
            local path="${source#path:}"
            if ! fetch_from_path "$path" "$dest"; then
                error "Failed to fetch ${name} from path"
            fi
        fi
        
        # Verify hash if available
        if [[ -n "$hash" ]]; then
            if ! fetch_verify "$dest" "$hash"; then
                error "Hash verification failed for ${name}"
            fi
        fi
        
        ((installed++))
        success "Installed ${name}@${version}"
    done < <(lockfile_read "$LOCKFILE")
    
    success "Installed ${installed} package(s) to ${VENDOR_DIR}/"
}

# Update command - generate lockfile from manifest
cmd_update() {
    check_project
    
    info "Updating dependencies from ${MANIFEST_FILE}..."
    
    # Initialize fetch cache
    fetch_init
    
    local entries=()
    
    # Parse dependencies from manifest
    while IFS='|' read -r name version git_url branch path; do
        info "Resolving ${name}..."
        
        local source=""
        local resolved_version="${version}"
        local hash=""
        
        if [[ -n "$path" ]]; then
            # Path dependency
            source="path:${path}"
            resolved_version="local"
        elif [[ -n "$git_url" ]]; then
            # Git dependency
            local branch_suffix=""
            [[ -n "$branch" ]] && branch_suffix="#${branch}"
            source="git:${git_url}${branch_suffix}"
            resolved_version="git-${branch:-main}"
        elif [[ -n "$version" ]]; then
            # Registry dependency
            source="registry"
            # MOCK: In production, query registry for available versions
            # For now, use version constraint as-is
            resolved_version="${version#^}"
            resolved_version="${resolved_version#~}"
            resolved_version="${resolved_version#=}"
        fi
        
        # MOCK: Calculate hash (in production, get from registry)
        hash="mock-hash-$(echo -n "${name}${resolved_version}" | sha256sum | cut -d' ' -f1 | head -c 16)"
        
        entries+=("${name}|${resolved_version}|${source}|${hash}")
        info "  Resolved to ${name}@${resolved_version}"
    done < <(toml_parse_dependencies "$MANIFEST_FILE")
    
    # Write lockfile
    lockfile_write "$LOCKFILE" "${entries[@]}"
    
    success "Updated ${#entries[@]} package(s) in ${LOCKFILE}"
}

# Add command (stub)
cmd_add() {
    local pkg="${1:-}"
    
    if [[ -z "$pkg" ]]; then
        error "Package name required. Usage: sounio-pkg add <package>"
    fi
    
    warn "The 'add' command is a stub. Please edit ${MANIFEST_FILE} manually."
    echo ""
    echo "To add a dependency, add to [dependencies] section:"
    echo "  ${pkg} = { version = \"^1.0.0\" }"
    echo ""
    echo "Then run: sounio-pkg update"
}

# Remove command (stub)
cmd_remove() {
    local pkg="${1:-}"
    
    if [[ -z "$pkg" ]]; then
        error "Package name required. Usage: sounio-pkg remove <package>"
    fi
    
    warn "The 'remove' command is a stub. Please edit ${MANIFEST_FILE} manually."
    echo ""
    echo "To remove a dependency, delete the line from [dependencies] section."
    echo ""
    echo "Then run: sounio-pkg update"
}

# Verify command - verify package hashes
cmd_verify() {
    check_project
    
    if [[ ! -f "$LOCKFILE" ]]; then
        error "No ${LOCKFILE} found. Run 'sounio-pkg update' first."
    fi
    
    info "Verifying package hashes..."
    
    local verified=0
    local failed=0
    
    while IFS='|' read -r name version source hash; do
        local pkg_dir="${VENDOR_DIR}/${name}"
        
        if [[ ! -d "$pkg_dir" ]]; then
            warn "${name} not installed, skipping"
            continue
        fi
        
        if [[ -z "$hash" ]]; then
            warn "${name} has no hash, skipping"
            continue
        fi
        
        # MOCK: Skip actual verification for mock hashes
        if [[ "$hash" == mock-hash:* ]]; then
            info "${name}: skipping mock hash"
            ((verified++))
            continue
        fi
        
        if fetch_verify "$pkg_dir" "$hash" 2>/dev/null; then
            success "${name}: verified"
            ((verified++))
        else
            error "${name}: verification failed"
            ((failed++))
        fi
    done < <(lockfile_read "$LOCKFILE")
    
    if [[ $failed -eq 0 ]]; then
        success "Verified ${verified} package(s)"
    else
        error "${failed} package(s) failed verification"
    fi
}

# Clean command - remove vendor and cache
cmd_clean() {
    info "Cleaning package directories..."
    
    if [[ -d "$VENDOR_DIR" ]]; then
        rm -rf "$VENDOR_DIR"
        info "Removed ${VENDOR_DIR}/"
    fi
    
    if [[ -f "$LOCKFILE" ]]; then
        rm -f "$LOCKFILE"
        info "Removed ${LOCKFILE}"
    fi
    
    fetch_clean_cache
    
    success "Cleaned all package files"
}

# Main entry point
main() {
    local cmd="${1:-help}"
    shift || true
    
    case "$cmd" in
        install)
            cmd_install "$@"
            ;;
        update)
            cmd_update "$@"
            ;;
        add)
            cmd_add "$@"
            ;;
        remove)
            cmd_remove "$@"
            ;;
        verify)
            cmd_verify "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            error "Unknown command: $cmd. Run 'sounio-pkg help' for usage."
            ;;
    esac
}

# Run main
main "$@"
