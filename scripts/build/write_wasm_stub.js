#!/usr/bin/env node
// write_wasm_stub.js — Write the minimal 8-byte WASM module stub.
// Used by build_playground_wasm.sh when printf is not available.
// Can also be run directly: node scripts/write_wasm_stub.js
'use strict';
const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, '..', 'website', 'public', 'wasm');
fs.mkdirSync(outDir, { recursive: true });

const outFile = path.join(outDir, 'sounio_compiler_bg.wasm');

// Minimal valid WASM module: magic number + version 1, no sections.
// \0asm (0x00 0x61 0x73 0x6d) + version (0x01 0x00 0x00 0x00)
const buf = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
fs.writeFileSync(outFile, buf);

console.log('[write_wasm_stub] Wrote', outFile, '(', buf.length, 'bytes)');
