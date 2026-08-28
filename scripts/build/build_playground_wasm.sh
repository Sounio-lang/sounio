#!/usr/bin/env bash
# build_playground_wasm.sh — Build playground WASM assets
#
# Phase 1: emits a pure-JS shim that makes the playground functional
# immediately. The shim implements compile/run/format/version without
# requiring a real WASM binary.
#
# Phase 2 (future): when wasm-pack + a WASM Sounio target exist, replace
# the shim with the real binary by running:
#   wasm-pack build compiler --target web --out-dir ../website/public/wasm
#
# Usage:
#   bash scripts/build_playground_wasm.sh
#
# The script is also wired as `npm run build:wasm` in website/package.json.
set -euo pipefail

# Script lives at scripts/build/ — repo root is two levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/website/public/wasm"

mkdir -p "${OUT_DIR}"

echo "[build_playground_wasm] Writing JS shim → ${OUT_DIR}/sounio_compiler.js"

cat > "${OUT_DIR}/sounio_compiler.js" << 'JSEOF'
// sounio_compiler.js — Phase 1 shim
// Implements the wasm-pack module contract:
//   default export  init(wasmUrl?) → Promise<void>
//   compile(src)    → JSON string
//   run(src)        → JSON string
//   format(src)     → string
//   version()       → string
//
// This shim does lightweight syntactic analysis so the playground is fully
// interactive before a real WASM binary is available.

const SHIM_VERSION = "sounio-shim-0.1.0 (phase-1)";

// ---------------------------------------------------------------------------
// Tiny Sounio syntax checker
// ---------------------------------------------------------------------------

function checkSource(src) {
  const lines = src.split('\n');
  const diagnostics = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNo = i + 1;

    // Rust-style &mut not allowed
    if (/&mut\b/.test(line)) {
      diagnostics.push({
        severity: 'error',
        message: 'Use `&!` instead of `&mut` (Sounio syntax)',
        line: lineNo,
        column: line.indexOf('&mut') + 1,
      });
    }

    // Rust macros not allowed
    const macroMatch = line.match(/\b(assert|println|eprintln|vec|panic|todo|unimplemented)!/);
    if (macroMatch) {
      diagnostics.push({
        severity: 'error',
        message: `Rust macro \`${macroMatch[0]}\` is not valid Sounio — use Sounio builtins`,
        line: lineNo,
        column: line.indexOf(macroMatch[0]) + 1,
      });
    }

    // #[attribute] not allowed
    if (/^\s*#\[/.test(line)) {
      diagnostics.push({
        severity: 'error',
        message: 'Rust attributes (#[...]) are not valid in Sounio',
        line: lineNo,
        column: 1,
      });
    }

    // `pub` keyword not implemented
    if (/\bpub\b/.test(line)) {
      diagnostics.push({
        severity: 'warning',
        message: '`pub` visibility modifier is not yet implemented in Sounio',
        line: lineNo,
        column: line.indexOf('pub') + 1,
      });
    }

    // IO effect without `with IO`
    if (/\bprint\s*\(/.test(line)) {
      const fnDefAbove = src.slice(0, src.split('\n').slice(0, i).join('\n').length);
      if (!/\bwith\b[^{]*\bIO\b/.test(fnDefAbove + line)) {
        // Only warn, not error — shim can't do full scope analysis
        diagnostics.push({
          severity: 'warning',
          message: 'Call to `print` may require `with IO` on the enclosing function',
          line: lineNo,
          column: line.indexOf('print(') + 1,
        });
      }
    }

    // let mut should be `var`
    if (/\blet\s+mut\b/.test(line)) {
      diagnostics.push({
        severity: 'error',
        message: 'Use `var` instead of `let mut` for mutable bindings',
        line: lineNo,
        column: line.indexOf('let mut') + 1,
      });
    }
  }

  return diagnostics;
}

// ---------------------------------------------------------------------------
// Simulate program output for well-known patterns
// ---------------------------------------------------------------------------

function simulateRun(src) {
  const output = [];

  // Extract string literals from print calls
  const printRe = /print\s*\(\s*"([^"]*)"/g;
  let m;
  while ((m = printRe.exec(src)) !== null) {
    output.push(m[1]);
  }

  // Simple arithmetic: detect `let x = <number>` and return value
  const mainReturnMatch = src.match(/fn\s+main\s*\([^)]*\)\s*->\s*i32[^{]*\{[\s\S]*?\n\s*(\d+)\s*\n?\s*\}/);
  const returnValue = mainReturnMatch ? mainReturnMatch[1] : null;

  return { output: output.join('\n'), returnValue };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function compile(source) {
  const diagnostics = checkSource(source);
  const hasErrors = diagnostics.some(d => d.severity === 'error');
  return JSON.stringify({
    schemaVersion: 1,
    success: !hasErrors,
    diagnostics,
  });
}

export function run(source) {
  const diagnostics = checkSource(source);
  const hasErrors = diagnostics.some(d => d.severity === 'error');

  if (hasErrors) {
    return JSON.stringify({
      schemaVersion: 1,
      success: false,
      diagnostics,
      output: '',
      returnValue: null,
    });
  }

  const { output, returnValue } = simulateRun(source);
  return JSON.stringify({
    schemaVersion: 1,
    success: true,
    diagnostics,
    output,
    returnValue,
  });
}

export function format(source) {
  // Minimal formatter: normalise indentation to 4 spaces
  return source
    .split('\n')
    .map(line => line.replace(/^\t+/, tabs => '    '.repeat(tabs.length)))
    .join('\n');
}

export function version() {
  return SHIM_VERSION;
}

// Default export matches wasm-pack init(wasmUrl?) → Promise<void>
export default async function init(_wasmUrl) {
  // No real WASM to initialise in Phase 1 — resolve immediately.
  return;
}
JSEOF

echo "[build_playground_wasm] Writing minimal WASM stub → ${OUT_DIR}/sounio_compiler_bg.wasm"

# Minimal valid WASM module: magic + version + empty (no sections).
# 8 bytes: \0asm (magic) + \1\0\0\0 (version 1).
# The JS shim does not call any WASM exports, so an empty module is fine.
printf '\x00\x61\x73\x6d\x01\x00\x00\x00' > "${OUT_DIR}/sounio_compiler_bg.wasm"

echo "[build_playground_wasm] Done."
echo ""
echo "  ${OUT_DIR}/sounio_compiler.js"
echo "  ${OUT_DIR}/sounio_compiler_bg.wasm"
echo ""
echo "Phase 2 (real WASM): when a WASM Sounio target is available, run:"
echo "  wasm-pack build compiler --target web --out-dir ../website/public/wasm"
echo "and commit the generated files."
