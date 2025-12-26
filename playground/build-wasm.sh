#!/bin/bash
# Build the Sounio compiler as a WASM module for the playground

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILER_DIR="$(dirname "$SCRIPT_DIR")/compiler"
PLAYGROUND_DIR="$SCRIPT_DIR"

echo "Building Sounio compiler for WASM target..."

# Check for wasm-pack or use cargo + wasm-bindgen-cli
if command -v wasm-pack &> /dev/null; then
    echo "Using wasm-pack..."
    cd "$COMPILER_DIR"
    wasm-pack build --target web --out-dir "$PLAYGROUND_DIR/dist" --features wasm
else
    echo "Using cargo + wasm-bindgen-cli..."

    # Check for wasm-bindgen CLI
    if ! command -v wasm-bindgen &> /dev/null; then
        echo "Installing wasm-bindgen-cli..."
        cargo install wasm-bindgen-cli
    fi

    # Check for wasm32 target
    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        echo "Installing wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi

    # Build
    cd "$COMPILER_DIR"
    cargo build --target wasm32-unknown-unknown --release --features wasm

    # Generate JS bindings
    mkdir -p "$PLAYGROUND_DIR/dist"
    wasm-bindgen \
        --target web \
        --out-dir "$PLAYGROUND_DIR/dist" \
        --out-name sounio_compiler \
        "$COMPILER_DIR/../target/wasm32-unknown-unknown/release/sounio.wasm"
fi

echo ""
echo "WASM build complete!"
echo "Output files in: $PLAYGROUND_DIR/dist/"
ls -la "$PLAYGROUND_DIR/dist/"*.wasm "$PLAYGROUND_DIR/dist/"*.js 2>/dev/null || true
