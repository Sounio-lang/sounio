#!/usr/bin/env node
// Wrapper project metadata only; uses the same boundary TOML parser.
import fs from 'node:fs';
import { load_toml, read_utf8, is_record, py_str, py_truth } from './core.mjs';
const [file, key] = process.argv.slice(2);
try {
  if (!file || !key || process.argv.length !== 4) throw new Error('expected manifest path and key');
  const data = load_toml(file);
  const table = (k) => {
    const v = data[k];
    if (v === undefined) return {};
    if (!is_record(v)) throw new Error(k + ' must be a table');
    return v;
  };
  if (key === 'package.name') console.log(py_str(table('package').name ?? 'sounio-project'));
  else if (key === 'entry') {
    let value = '';
    const bins = data.bin;
    if (Array.isArray(bins) && bins.length) {
      if (!is_record(bins[0])) throw new Error('bin item is not a table');
      value = bins[0].path ?? '';
    }
    if (!py_truth(value)) value = table('project').entry ?? '';
    if (!py_truth(value)) {
      const lib = table('lib').path ?? '';
      if (py_truth(lib) && !read_utf8(file).includes('[[bin]]')) value = lib;
    }
    console.log(py_truth(value) ? py_str(value) : 'src/main.sio');
  } else {
    console.error('unsupported manifest key');
    process.exitCode = 2;
  }
} catch (e) {
  console.error(`manifest-value: ${e.message}`);
  process.exitCode = 3;
}
