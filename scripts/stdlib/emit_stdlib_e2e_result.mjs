#!/usr/bin/env node
// Operational metadata only. Preserve the v1 E2E JSON and TSV producer contract.
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
export function decodeDetail(raw) {
  // Match non-strict binascii.a2b_base64 semantics rather than Node's different
  // URL-safe/permissive decoder. Reviewed against CPython3.12.3 Modules/binascii.c.
  if (/[^\x00-\x7f]/.test(raw)) throw new Error('base64 detail must be ASCII');
  const bytes = [];
  let position = 0, left = 0, pads = 0, padded = false;
  for (const character of raw) {
    if (character === '=') {
      if (position >= 2 && position + ++pads >= 4) { padded = true; break; }
      continue;
    }
    const digit = alphabet.indexOf(character);
    if (digit < 0) continue;
    pads = 0;
    if (position === 0) { left = digit; position = 1; }
    else if (position === 1) { bytes.push((left << 2) | (digit >> 4)); left = digit & 15; position = 2; }
    else if (position === 2) { bytes.push((left << 4) | (digit >> 2)); left = digit & 3; position = 3; }
    else { bytes.push((left << 6) | digit); left = 0; position = 0; }
  }
  if (!padded && position !== 0) throw new Error(position === 1 ? 'invalid base64 data length' : 'incorrect base64 padding');
  return Buffer.from(bytes).toString('utf8'); // UTF-8 errors are replaced; BOM is retained.
}
function integer(raw) {
  // Bash emits decimal counters and process status. Keep all signed 64-bit
  // producer integers exact instead of rounding values above Number.MAX_SAFE_INTEGER.
  const token = String(raw).trim();
  if (!/^[+-]?[0-9](?:_?[0-9])*$/.test(token)) throw new Error('invalid decimal integer: ' + token);
  return BigInt(token.replaceAll('_', ''));
}
export function timestampUtc(date = new Date()) {
  // datetime.isoformat uses six fractional digits when the fraction is nonzero.
  return date.toISOString().replace(/\.(\d{3})Z$/, (_, digits) => digits === '000' ? 'Z' : '.' + digits + '000Z');
}
export function makeDocument(text, context, counts, generatedAt = timestampUtc()) {
  const results = [], ignored = [];
  for (const line of text.split(/\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]/)) {
    if (!line) continue;
    const fields = line.split('\t');
    if (fields.length !== 8) continue; // Preserve the existing malformed-row policy.
    const [status, test_path, mode, reason, owner, unblock_condition, exit_code, detail_b64] = fields;
    const row = { status, test_path, mode, reason, owner, unblock_condition,
      exit_code: integer(exit_code), detail_excerpt: decodeDetail(detail_b64) };
    results.push(row);
    if (mode === 'ignore') ignored.push({ test_path, reason, owner, unblock_condition });
  }
  const pass = integer(counts.pass), fail = integer(counts.fail), skip = integer(counts.skip);
  return { schema: 'sounio.stdlib.e2e.result.v1', generated_at_utc: generatedAt,
    command: 'bash scripts/stdlib/run_stdlib_e2e.sh', context,
    totals: { pass, fail, skip, total: pass + fail + skip },
    results, failures: results.filter(row => row.status === 'fail'), ignored };
}
export function stringifyDocument(value, depth = 0) {
  // Match indent=2/ensure_ascii output while retaining exact integer JSON tokens.
  if (typeof value === 'bigint') return value.toString();
  if (typeof value !== 'object' || value === null) {
    return JSON.stringify(value).replace(/[\u007f-\uffff]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
  }
  const array = Array.isArray(value), entries = array ? value : Object.entries(value);
  if (!entries.length) return array ? '[]' : '{}';
  const indent = '  '.repeat(depth + 1), tail = '  '.repeat(depth);
  const rows = entries.map(entry => indent + (array ? stringifyDocument(entry, depth + 1)
    : stringifyDocument(entry[0]) + ': ' + stringifyDocument(entry[1], depth + 1)));
  return (array ? '[\n' : '{\n') + rows.join(',\n') + '\n' + tail + (array ? ']' : '}');
}
export function emit(args) {
  if (args.length !== 9) throw new Error('usage: RESULTS_TSV OUTPUT_JSON PASS FAIL SKIP ROOT SOUC STDLIB FILTER');
  const [resultsPath, outPath, pass, fail, skip, root_dir, souc_bin, stdlib_path, filter] = args;
  const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(readFileSync(resultsPath));
  const document = makeDocument(text, { root_dir, souc_bin, stdlib_path, filter }, { pass, fail, skip });
  writeFileSync(outPath, stringifyDocument(document) + '\n', 'utf8');
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { emit(process.argv.slice(2)); }
  catch (error) { console.error('STDLIB_E2E_JSON_ERROR: ' + error.message); process.exitCode = 1; }
}
