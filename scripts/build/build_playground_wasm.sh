#!/usr/bin/env bash
# Reproduce the current browser preview shim. This is not a Sounio compiler.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_js="$repo_root/scripts/build/sounio_compiler.js"
target_dir="$repo_root/website/public/wasm"

test -s "$source_js"
mkdir -p "$target_dir"
cp "$source_js" "$target_dir/sounio_compiler.js"

# Valid, empty WebAssembly module retained for the current browser contract.
# The JavaScript preview does not execute it.
printf '\x00\x61\x73\x6d\x01\x00\x00\x00' > "$target_dir/sounio_compiler_bg.wasm"

echo 'Browser preview shim regenerated; no Sounio compilation occurred.'
