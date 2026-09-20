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

    // let mut should be `var`
    if (/\blet\s+mut\b/.test(line)) {
      diagnostics.push({
        severity: 'error',
        message: 'Use `var` instead of `let mut` for mutable bindings',
        line: lineNo,
        column: line.indexOf('let mut') + 1,
      });
    }

    // IO effect check: print without `with IO` in function signature
    if (/\bprint\s*\(/.test(line)) {
      // Walk back to find the nearest fn declaration
      let hasWith = false;
      for (let j = i; j >= 0; j--) {
        if (/\bfn\b/.test(lines[j])) {
          hasWith = /\bwith\b[^{]*\bIO\b/.test(lines[j]);
          break;
        }
      }
      if (!hasWith) {
        diagnostics.push({
          severity: 'warning',
          message: 'Call to `print` requires `with IO` on the enclosing function',
          line: lineNo,
          column: line.indexOf('print(') + 1,
        });
      }
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

  // Simple: detect the final expression in main as the return value
  const mainBody = src.match(/fn\s+main[^{]*\{([\s\S]*)\}/);
  let returnValue = null;
  if (mainBody) {
    const body = mainBody[1].trim();
    const lastLine = body.split('\n').filter(l => l.trim()).pop() || '';
    const numMatch = lastLine.trim().match(/^(-?\d+)$/);
    if (numMatch) returnValue = numMatch[1];
  }

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
  // Minimal formatter: normalise tabs to 4 spaces
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
