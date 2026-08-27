#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { accessSync, constants } from "node:fs";

const compiler = process.argv[2];
if (!compiler) {
  console.error("usage: madaros_version_json_gate.mjs <compiler>");
  process.exit(2);
}

try {
  accessSync(compiler, constants.X_OK);
} catch {
  console.error(`compiler is not executable: ${compiler}`);
  process.exit(2);
}

const run = spawnSync(compiler, ["--version-json"], {
  encoding: "utf8",
  timeout: 30_000,
});

if (run.error || run.status !== 0) {
  console.error(run.error?.message ?? run.stderr.trim());
  process.exit(run.status ?? 1);
}

const raw = run.stdout.trim();
let parsed;
try {
  parsed = JSON.parse(raw);
} catch (error) {
  console.error(`invalid version JSON: ${error.message}`);
  console.error(raw);
  process.exit(1);
}

const expected = {
  abi_version: 1,
  runtime_version: "1.0.0",
  ir_max_funcs: 1024,
  ir_max_instrs: 128,
  supports_ffi: true,
  supports_gpu: false,
};

const mismatches = Object.entries(expected)
  .filter(([key, value]) => parsed[key] !== value)
  .map(([key, value]) => ({ key, expected: value, actual: parsed[key] }));

const report = {
  schema: "sounio.madaros-version-json-gate.v1",
  compiler,
  parsed,
  mismatches,
  verified: mismatches.length === 0,
};

console.log(JSON.stringify(report));
process.exit(report.verified ? 0 : 1);
