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
readonly PKG_VERSION="0.2.0"
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
    add <pkg>        Add a package dependency
    remove <pkg>     Remove a package dependency
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

Add Examples:
    sounio-pkg add stdlib@^1.2.0
    sounio-pkg add parser --git https://github.com/sounio/parser.git
    sounio-pkg add utils --path ../utils
    sounio-pkg add test-helpers --dev

Remove Examples:
    sounio-pkg remove stdlib
    sounio-pkg remove stdlib --clean
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

# Query registry for available versions
query_registry() {
    local name="$1"
    
    # Try to fetch from packages.souniolang.org
    # Fallback to hardcoded for now
    local versions
    versions=$(curl -s "https://packages.souniolang.org/v1/packages/$name/versions" 2>/dev/null) || true
    
    if [[ -n "$versions" ]]; then
        echo "$versions" | head -1
    else
        # Fallback - assume 1.0.0 exists
        echo "1.0.0"
    fi
}

# Add registry dependency to Sounio.toml
add_registry_dep() {
    local name="$1"
    local version="$2"
    local is_dev="$3"
    local toml_file="${4:-$MANIFEST_FILE}"
    
    local section="dependencies"
    $is_dev && section="dev-dependencies"
    
    # Check if already exists
    local parsed
    parsed=$(toml_parse "$toml_file" 2>/dev/null || true)
    
    # Ensure [dependencies] or [dev-dependencies] section exists
    if ! grep -q "^\[$section\]" "$toml_file" 2>/dev/null; then
        echo "" >> "$toml_file"
        echo "[$section]" >> "$toml_file"
    fi
    
    # Check if dependency already exists
    if grep -q "^${name}\s*=" "$toml_file" 2>/dev/null; then
        # Update existing dependency
        sed -i "s/^${name}\s*=.*/${name} = \"${version}\"/" "$toml_file"
    else
        # Add new dependency to section
        # Find the section and add after it
        local tmp_file="${toml_file}.tmp"
        local in_section=false
        local section_found=false
        
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" == "[$section]" ]]; then
                in_section=true
                section_found=true
            elif [[ "$line" =~ ^\[.*\]$ ]]; then
                in_section=false
            fi
            
            echo "$line" >> "$tmp_file"
            
            # Add dependency at end of section (before next section or EOF)
            if $in_section && $section_found; then
                if [[ "$line" =~ ^\[.*\]$ ]]; then
                    : # Just entered section, don't add yet
                elif [[ -z "$line" ]] || [[ "$line" =~ ^\[.*\]$ ]]; then
                    : # Empty line or next section
                fi
            fi
        done < "$toml_file"
        
        # Use sed to add after section header
        sed -i "/^\[$section\]/a ${name} = \"${version}\"" "$toml_file"
    fi
    
    log_info "Added $name = \"$version\" to [$section]"
}

# Add git dependency to Sounio.toml
add_git_dep() {
    local name="$1"
    local git_url="$2"
    local version="$3"
    local is_dev="$4"
    local toml_file="${5:-$MANIFEST_FILE}"
    
    local section="dependencies"
    $is_dev && section="dev-dependencies"
    
    # Ensure section exists
    if ! grep -q "^\[$section\]" "$toml_file" 2>/dev/null; then
        echo "" >> "$toml_file"
        echo "[$section]" >> "$toml_file"
    fi
    
    # Build inline table
    local dep_line
    if [[ -n "$version" ]]; then
        dep_line="${name} = { git = \"${git_url}\", version = \"${version}\" }"
    else
        dep_line="${name} = { git = \"${git_url}\" }"
    fi
    
    # Add or update
    if grep -q "^${name}\s*=" "$toml_file" 2>/dev/null; then
        # Remove old entry
        sed -i "/^${name}\s*=/d" "$toml_file"
    fi
    
    # Add after section header
    sed -i "/^\[$section\]/a ${dep_line}" "$toml_file"
    
    log_info "Added $name (git: $git_url) to [$section]"
}

# Add path dependency to Sounio.toml
add_path_dep() {
    local name="$1"
    local path="$2"
    local is_dev="$3"
    local toml_file="${4:-$MANIFEST_FILE}"
    
    local section="dependencies"
    $is_dev && section="dev-dependencies"
    
    # Ensure section exists
    if ! grep -q "^\[$section\]" "$toml_file" 2>/dev/null; then
        echo "" >> "$toml_file"
        echo "[$section]" >> "$toml_file"
    fi
    
    # Build inline table
    local dep_line="${name} = { path = \"${path}\" }"
    
    # Add or update
    if grep -q "^${name}\s*=" "$toml_file" 2>/dev/null; then
        # Remove old entry
        sed -i "/^${name}\s*=/d" "$toml_file"
    fi
    
    # Add after section header
    sed -i "/^\[$section\]/a ${dep_line}" "$toml_file"
    
    log_info "Added $name (path: $path) to [$section]"
}

# Remove dependency from Sounio.toml manifest
remove_from_manifest() {
    local name="$1"
    local toml_file="${2:-$MANIFEST_FILE}"
    
    if [[ ! -f "$toml_file" ]]; then
        log_error "Manifest file not found: $toml_file"
        return 1
    fi
    
    # Check if dependency exists
    if ! grep -q "^${name}\s*=" "$toml_file" 2>/dev/null; then
        log_warn "Dependency '$name' not found in $toml_file"
        return 1
    fi
    
    # Remove the line
    sed -i "/^${name}\s*=/d" "$toml_file"
    
    log_info "Removed $name from $toml_file"
    return 0
}

# Remove package from lockfile
remove_from_lockfile() {
    local name="$1"
    local lock_file="${2:-$LOCKFILE}"
    
    if [[ ! -f "$lock_file" ]]; then
        return 0
    fi
    
    lockfile_remove "$lock_file" "$name"
    log_info "Removed $name from $lock_file"
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
        
        # Initialize variables
        dep_name=""
        dep_version=""
        dep_source=""
        dep_hash=""

        # Check if inline table or simple version
        if [[ "$value" == \{*\} ]]; then
            # Parse inline table
            while IFS='=' read -r k v; do
                case "$k" in
                    version) dep_version="$v" ;;
                    git) dep_source="git+$v" ;;
                    path) dep_source="path://$v" ;;
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

# Add a package dependency
cmd_add() {
    local dep_spec="${1:-}"
    
    # Handle help flag before checking for package name
    if [[ "$dep_spec" == "--help" ]] || [[ "$dep_spec" == "-h" ]]; then
        echo "Usage: sounio-pkg add <package>[@version] [options]"
        echo ""
        echo "Add a package dependency to Sounio.toml"
        echo ""
        echo "Options:"
        echo "  --git <url>       Add from git repository"
        echo "  --path <path>     Add from local path"
        echo "  --version <ver>   Specify version constraint"
        echo "  --dev             Add as dev dependency"
        echo ""
        echo "Examples:"
        echo "  sounio-pkg add stdlib@^1.2.0"
        echo "  sounio-pkg add parser --git https://github.com/sounio/parser.git"
        echo "  sounio-pkg add utils --path ../utils"
        echo "  sounio-pkg add test-helpers --dev"
        return 0
    fi
    
    if [[ -z "$dep_spec" ]]; then
        log_error "Package name required"
        echo "Usage: sounio-pkg add <package>[@version] [options]"
        echo "Options:"
        echo "  --git <url>       Add from git repository"
        echo "  --path <path>     Add from local path"
        echo "  --version <ver>   Specify version"
        echo "  --dev             Add as dev dependency"
        return 1
    fi
    shift
    
    # Parse name[@version]
    local name="${dep_spec%%@*}"
    local version="${dep_spec#*@}"
    [[ "$version" == "$dep_spec" ]] && version=""
    
    # Parse options
    local git_url=""
    local path=""
    local is_dev=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --git) git_url="$2"; shift 2 ;;
            --path) path="$2"; shift 2 ;;
            --version) version="$2"; shift 2 ;;
            --dev) is_dev=true; shift ;;
            -h|--help) 
                echo "Usage: sounio-pkg add <package>[@version] [options]"
                echo "Options:"
                echo "  --git <url>       Add from git repository"
                echo "  --path <path>     Add from local path"
                echo "  --version <ver>   Specify version"
                echo "  --dev             Add as dev dependency"
                return 0
                ;;
            *) 
                log_error "Unknown option: $1"
                echo "Run 'sounio-pkg add --help' for usage"
                return 1
                ;;
        esac
    done
    
    # Verify manifest exists
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_error "Manifest file not found: $MANIFEST_FILE"
        return 1
    fi
    
    # Update Sounio.toml
    if [[ -n "$git_url" ]]; then
        add_git_dep "$name" "$git_url" "$version" "$is_dev"
    elif [[ -n "$path" ]]; then
        add_path_dep "$name" "$path" "$is_dev"
    else
        # Registry dependency - query for latest if no version specified
        if [[ -z "$version" ]]; then
            log_info "Querying registry for $name..."
            version=$(query_registry "$name")
            version="^${version}"
        fi
        add_registry_dep "$name" "${version:-^1.0.0}" "$is_dev"
    fi
    
    # Regenerate lockfile
    cmd_update
    
    log_success "Added $name to dependencies"
}

# Remove a package dependency
cmd_remove() {
    local name="${1:-}"
    local clean=false
    
    if [[ -z "$name" ]]; then
        log_error "Package name required"
        echo "Usage: sounio-pkg remove <package> [--clean]"
        echo "Options:"
        echo "  --clean    Also remove from vendor directory"
        return 1
    fi
    
    # Check for --clean flag
    if [[ "${2:-}" == "--clean" ]]; then
        clean=true
    fi
    
    # Verify manifest exists
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_error "Manifest file not found: $MANIFEST_FILE"
        return 1
    fi
    
    # Remove from Sounio.toml
    if ! remove_from_manifest "$name"; then
        # Dependency not found, but continue to clean lockfile if exists
        true
    fi
    
    # Remove from lockfile
    if [[ -f "$LOCKFILE" ]]; then
        remove_from_lockfile "$name"
    fi
    
    # Clean vendor if requested
    if $clean && [[ -d "vendor/$name" ]]; then
        rm -rf "vendor/$name"
        log_success "Cleaned vendor/$name"
    fi
    
    log_success "Removed $name"
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
