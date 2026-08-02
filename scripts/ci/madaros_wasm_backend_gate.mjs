import {readFile} from 'node:fs/promises';

const artifactPath = process.argv[2];
if (!artifactPath) {
  throw new Error('usage: node scripts/ci/madaros_wasm_backend_gate.mjs <artifact.wasm>');
}

const bytes = await readFile(artifactPath);
if (bytes.length < 8) {
  throw new Error(`WASM artifact is too small: ${bytes.length}`);
}
if (!bytes.subarray(0, 4).equals(Buffer.from([0x00, 0x61, 0x73, 0x6d]))) {
  throw new Error('WASM artifact has an invalid magic header');
}
if (!WebAssembly.validate(bytes)) {
  throw new Error('WebAssembly.validate rejected the artifact');
}

const module = await WebAssembly.compile(bytes);
const imports = WebAssembly.Module.imports(module);
if (imports.length !== 0) {
  throw new Error(`fixture unexpectedly requires ${imports.length} host import(s)`);
}

const instance = await WebAssembly.instantiate(module, {});
const exportedMain = instance.exports.main;
if (typeof exportedMain !== 'function') {
  throw new Error('WASM artifact does not export main');
}
const result = exportedMain();
if (result !== 3n) {
  throw new Error(`WASM main returned ${String(result)} instead of 3`);
}

const exports = WebAssembly.Module.exports(module).map((entry) => entry.name).sort();
process.stdout.write(`${JSON.stringify({
  schema: 'sounio.madaros-wasm-backend-gate.v1',
  artifactBytes: bytes.length,
  imports: [],
  exports,
  mainResult: String(result),
  validated: true,
}, null, 2)}\n`);
