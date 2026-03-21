#!/bin/bash
# Sounio Installation Script
#
# Usage: curl -fsSL https://sounio-lang.org/install.sh | bash
#    or: curl -fsSL https://raw.githubusercontent.com/sounio-lang/sounio/main/dist/install.sh | bash
#
# Environment variables:
#   SOUNIO_VERSION  - Version to install (default: latest)
#   SOUNIO_INSTALL  - Installation directory (default: ~/.sounio)

set -euo pipefail

# Configuration
GITHUB_REPO="sounio-lang/sounio"
INSTALL_DIR="${SOUNIO_INSTALL:-$HOME/.sounio}"
BIN_DIR="$INSTALL_DIR/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}info:${NC} $1"
}

log_success() {
    echo -e "${GREEN}success:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}warning:${NC} $1"
}

log_error() {
    echo -e "${RED}error:${NC} $1" >&2
}

# Detect OS and architecture
detect_platform() {
    local os arch

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$os" in
        linux)
            os="unknown-linux-gnu"
            ;;
        darwin)
            os="apple-darwin"
            ;;
        mingw*|msys*|cygwin*)
            os="pc-windows-msvc"
            ;;
        *)
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac

    case "$arch" in
        x86_64|amd64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="aarch64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac

    echo "${arch}-${os}"
}

# Get latest version from GitHub
get_latest_version() {
    local version
    version=$(curl -fsSL "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    if [ -z "$version" ]; then
        log_error "Failed to get latest version"
        exit 1
    fi
    echo "$version"
}

# Download and install
install_sounio() {
    local version="${SOUNIO_VERSION:-}"
    local platform
    local download_url
    local archive_name
    local ext

    platform=$(detect_platform)
    log_info "Detected platform: $platform"

    if [ -z "$version" ]; then
        log_info "Fetching latest version..."
        version=$(get_latest_version)
    fi
    log_info "Installing Sounio v$version"

    # Determine archive extension
    if [[ "$platform" == *"windows"* ]]; then
        ext="zip"
    else
        ext="tar.gz"
    fi

    archive_name="souc-${platform}.${ext}"
    download_url="https://github.com/${GITHUB_REPO}/releases/download/v${version}/${archive_name}"

    # Create installation directory
    mkdir -p "$BIN_DIR"

    # Download
    log_info "Downloading from $download_url"
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT

    if ! curl -fsSL "$download_url" -o "$temp_dir/$archive_name"; then
        log_error "Download failed. Please check your internet connection and try again."
        exit 1
    fi

    # Extract
    log_info "Extracting..."
    cd "$temp_dir"
    if [[ "$ext" == "zip" ]]; then
        unzip -q "$archive_name"
    else
        tar -xzf "$archive_name"
    fi

    # Install binary
    if [[ "$platform" == *"windows"* ]]; then
        cp souc.exe "$BIN_DIR/"
    else
        cp souc "$BIN_DIR/"
        chmod +x "$BIN_DIR/souc"
    fi

    log_success "Sounio v$version installed to $BIN_DIR/souc"

    # Check if bin directory is in PATH
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        log_warn "Add $BIN_DIR to your PATH:"
        echo ""
        echo "  For bash/zsh, add to ~/.bashrc or ~/.zshrc:"
        echo "    export PATH=\"$BIN_DIR:\$PATH\""
        echo ""
        echo "  For fish, run:"
        echo "    fish_add_path $BIN_DIR"
        echo ""
    fi

    # Verify installation
    if [ -x "$BIN_DIR/souc" ]; then
        log_success "Installation complete!"
        echo ""
        "$BIN_DIR/souc" --version 2>/dev/null || true
    fi
}

# Run installation
install_sounio
