#!/usr/bin/env node
// Line-local parameter-position function-type census. Preserve legacy hit text,
// but inspect balanced return types before deciding whether the type owns effects.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function bareFunctionTypes(source) {
  const hits = [];
  for (const raw of source.split('\n')) {
    // Deliberately retain the previous instrument's noise-stripping scope.
    const line = raw.replace(/\/\/.*$/, '').replace(/"[^"]*"/g, '');
    const anchors = /:\s*fn\([^)]*\)\s*->/g;
    let anchor;
    while ((anchor = anchors.exec(line)) !== null) {
      const afterArrow = anchors.lastIndex;
      const stack = [];
      let ownsEffects = false;
      for (let i = afterArrow; i < line.length; i++) {
        const c = line[i];
        if (stack.length === 0) {
          // Do not borrow an enclosing declaration's clause after its ')' or
          // the next parameter's annotation after a top-level comma.
          if (c === ')' || c === ',' || c === '=' || c === '{') break;
          const tail = line.slice(i);
          if (/^with\s+[A-Za-z]/.test(tail) &&
              (i === 0 || !/[A-Za-z0-9_]/.test(line[i - 1]))) {
            ownsEffects = true;
            break;
          }
        }
        if (c === '(') stack.push(')');
        else if (c === '[') stack.push(']');
        else if (c === '<') stack.push('>');
        else if (stack.length !== 0 && c === stack.at(-1)) stack.pop();
      }
      if (!ownsEffects) {
        // Keeping this display format avoids rewriting frozen baseline lists.
        let end = afterArrow;
        while (end < line.length && line[end] !== ',' && line[end] !== ')') end++;
        hits.push(line.slice(anchor.index, end));
      }
    }
  }
  return hits;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  try {
    const args = process.argv.slice(2);
    const paths = args[0] === '--files0'
      ? fs.readFileSync(0, 'utf8').split('\0').filter(Boolean)
      : args;
    for (const path of paths) {
      if (/^(archive|bootstrap)\//.test(path) || /\.sio\.old$/.test(path)) continue;
      if (!fs.existsSync(path) || !fs.statSync(path).isFile()) continue;
      for (const hit of bareFunctionTypes(fs.readFileSync(path, 'utf8'))) {
        process.stdout.write(path + '\t' + hit + '\n');
      }
    }
  } catch (error) {
    process.stderr.write('fn_type_effect_scan: ' + error.message + '\n');
    process.exitCode = 2;
  }
}
