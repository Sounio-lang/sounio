// Operational science-boundary metadata; no scientific numerical calculation.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash, randomBytes } from 'node:crypto';
import { parse as parseToml, TomlDate } from './vendor/smol-toml/dist/index.js';
export const RECEIPT_SCHEMA = 'sounio.package-boundary-receipt.v1';
export const CLAIM_SCHEMA = 'sounio.claim-contract.v1';
export const SCIENCE_SCHEMA = 'sounio.science-manifest.v1';
export const TSV_COLUMNS = [
  'path',
  'ring',
  'evidence_status',
  'context_of_use',
  'visibility',
  'enforcement',
  'next_gate',
  'allowed_claim_classes',
  'evidence_refs',
  'declared_by',
  'declared_at',
  'review_state',
];
export const CONCLUSIVE_RINGS = new Set(['pl-core', 'scientific-package', 'research']);
export const AUDITABLE_RINGS = new Set([
  'scientific-package-candidate',
  'mixed-unresolved',
  'unclassified',
]);
export const VISIBILITIES = new Set(['public', 'protected', 'embargoed']);
export const ENFORCEMENTS = new Set(['off', 'advisory', 'strict']);
export const ALLOWED_DEPENDENCIES = {
  'pl-core': new Set(['pl-core']),
  'scientific-package': new Set(['scientific-package', 'pl-core']),
  research: new Set(['research', 'scientific-package', 'pl-core']),
};
export const EMPIRICAL_CLASSES = new Set(['empirical', 'clinical', 'clinical-validation']);
export const EMPIRICAL_EVIDENCE = new Set(['dataset', 'split', 'diagnostics', 'gate', 'review']);
export const GUM_CLASSES = new Set(['gum', 'gum-uncertainty', 'gum-evaluation']);
export const GUM_EVIDENCE = new Set(['method', 'witness']);
export const LEGACY_EVIDENCE_TYPES = new Set([
  'epistemic-score',
  'score',
  'regulatory-ready',
  'provenance-level',
  'gum-compliant',
  'validation-coverage',
]);
export const EXIT_REJECT = 20,
  EXIT_UNKNOWN = 21;
export class Diagnostic {
  constructor(code, severity, message, p = '', dependency = '') {
    Object.assign(this, { code, severity, message, path: p, dependency });
  }
  to_dict() {
    return {
      code: this.code,
      severity: this.severity,
      message: this.message,
      ...(this.path ? { path: this.path } : {}),
      ...(this.dependency ? { dependency: this.dependency } : {}),
    };
  }
}
export class PolicyRow {
  constructor(fields) {
    Object.assign(this, fields);
  }
  public_dict() {
    const result = {};
    for (const key of TSV_COLUMNS.slice(1)) result[key] = this[key];
    return result;
  }
}
export class Policy {
  constructor({ root, source = null, kind, rows = [], diagnostics = [], legacy_present = false }) {
    Object.assign(this, { root, source, kind, rows, diagnostics, legacy_present });
  }
  get valid() {
    return !this.diagnostics.some((d) => d.code === 'E-SRB-000');
  }
  get declared() {
    return this.rows.length > 0;
  }
}
export class ImportSpec {
  constructor(segments, line) {
    Object.assign(this, { segments, line });
  }
  get display() {
    return this.segments.join('::');
  }
}
export class Closure {
  constructor({ root, source }) {
    Object.assign(this, {
      root,
      source,
      nodes: [],
      edges: [],
      unresolved: [],
      parse_failures: [],
      saturated: false,
      capacity: 256,
      collector: 'sounio-host-syntax-v1',
      report_sha256: '',
    });
  }
}
export function py_strip(v) {
  return String(v).replace(
    /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g,
    ''
  );
}
export function is_record(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v) && !(v instanceof Date);
}
export function py_truth(v) {
  if (v === null || v === undefined || v === false || v === 0 || v === 0n || v === '') return false;
  if (Array.isArray(v)) return v.length > 0;
  if (is_record(v)) return Object.keys(v).length > 0;
  return true;
}
export function python_compare(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const c = python_compare(a[i], b[i]);
      if (c) return c;
    }
    return a.length - b.length;
  }
  if (
    (typeof a === 'number' || typeof a === 'bigint') &&
    (typeof b === 'number' || typeof b === 'bigint')
  )
    return a < b ? -1 : a > b ? 1 : 0;
  const x = Array.from(String(a)),
    y = Array.from(String(b));
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const c = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (c) return c;
  }
  return x.length - y.length;
}
export function python_sort(values, key = (x) => x) {
  return [...values].sort((a, b) => python_compare(key(a), key(b)));
}
function temporalString(v) {
  const raw = v.boundaryTomlLexeme;
  if (typeof raw !== 'string') throw new Error('TOML temporal has no lossless source lexeme');
  const m = raw.match(
    /^(?:(\d{4}-\d{2}-\d{2})(?:[Tt ]|$))?(?:(\d{2}:\d{2})(?::(\d{2}))?(?:\.(\d+))?([Zz]|[+-]\d{2}:\d{2})?)?$/
  );
  if (!m) throw new Error('unsupported TOML temporal representation');
  if (!m[2]) return m[1];
  let clock = m[2] + ':' + (m[3] || '00');
  const micros = (m[4] || '').slice(0, 6).padEnd(6, '0');
  if (/[^0]/.test(micros)) clock += '.' + micros;
  let offset = m[5] || '';
  if (offset.toUpperCase() === 'Z' || offset === '-00:00') offset = '+00:00';
  return (m[1] ? m[1] + ' ' : '') + clock + offset;
}
export function py_str(v) {
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  if (v instanceof TomlDate) return temporalString(v);
  if (typeof v === 'bigint') return v.toString();
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'nan';
    if (v === Infinity) return 'inf';
    if (v === -Infinity) return '-inf';
    if (Object.is(v, -0)) return '-0.0';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e16))
      return v.toExponential().replace(/e([+-])(\d)$/, 'e$10$2');
    return Number.isInteger(v) ? v.toFixed(1) : String(v);
  }
  if (Array.isArray(v)) return '[' + v.map(py_repr).join(', ') + ']';
  if (is_record(v))
    return (
      '{' +
      Object.entries(v)
        .map(([k, x]) => py_repr(k) + ': ' + py_repr(x))
        .join(', ') +
      '}'
    );
  return String(v);
}
function py_repr(v) {
  if (typeof v !== 'string') return py_str(v);
  const quote = v.includes("'") && !v.includes('"') ? '"' : "'";
  let out = quote;
  for (const c of v) {
    const n = c.codePointAt(0);
    if (c === '\\' || c === quote) out += '\\' + c;
    else if (c === '\t') out += '\\t';
    else if (c === '\n') out += '\\n';
    else if (c === '\r') out += '\\r';
    else if (c !== ' ' && /[\p{C}\p{Z}]/u.test(c))
      out +=
        n <= 255
          ? '\\x' + n.toString(16).padStart(2, '0')
          : n <= 65535
            ? '\\u' + n.toString(16).padStart(4, '0')
            : '\\U' + n.toString(16).padStart(8, '0');
    else out += c;
  }
  return out + quote;
}

export function is_sha256(v, { allow_empty = false } = {}) {
  const s = py_str(v);
  return (allow_empty && s === '') || /^[0-9a-f]{64}$/.test(s);
}
export function sha256_bytes(data) {
  return createHash('sha256').update(data).digest('hex');
}
export function sha256_file(p) {
  const h = createHash('sha256'),
    fd = fs.openSync(p, 'r'),
    buf = Buffer.alloc(1024 * 1024);
  try {
    for (;;) {
      const n = fs.readSync(fd, buf, 0, buf.length, null);
      if (n === 0) break;
      h.update(buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return h.digest('hex');
}
function asciiJson(text) {
  return text.replace(
    /[\u007f-\uffff]/g,
    (c) => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0')
  );
}
function canonicalText(v) {
  if (v === null) return 'null';
  if (typeof v === 'string') return asciiJson(JSON.stringify(v));
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') {
    if (!Number.isSafeInteger(v)) throw new Error('receipt numbers must be safe integers');
    return JSON.stringify(v);
  }
  if (Array.isArray(v)) return '[' + v.map(canonicalText).join(',') + ']';
  if (is_record(v))
    return (
      '{' +
      python_sort(Object.keys(v))
        .map((k) => canonicalText(k) + ':' + canonicalText(v[k]))
        .join(',') +
      '}'
    );
  throw new Error('unsupported canonical receipt value');
}
export function canonical_json(v) {
  return Buffer.from(canonicalText(v), 'ascii');
}
export function receipt_identity(receipt) {
  const payload = structuredClone(receipt);
  if (!is_record(payload.hashes)) payload.hashes = {};
  delete payload.hashes.receipt_identity_sha256;
  return sha256_bytes(canonical_json(payload));
}
function prettyText(v, depth = 0) {
  if (Array.isArray(v)) {
    if (!v.length) return '[]';
    return (
      '[\n' +
      v.map((x) => ' '.repeat(depth + 2) + prettyText(x, depth + 2)).join(',\n') +
      '\n' +
      ' '.repeat(depth) +
      ']'
    );
  }
  if (is_record(v)) {
    const keys = python_sort(Object.keys(v));
    if (!keys.length) return '{}';
    return (
      '{\n' +
      keys
        .map((k) => ' '.repeat(depth + 2) + canonicalText(k) + ': ' + prettyText(v[k], depth + 2))
        .join(',\n') +
      '\n' +
      ' '.repeat(depth) +
      '}'
    );
  }
  return canonicalText(v);
}
export function write_json_atomic(p, payload) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  if (!is_record(payload.hashes)) payload.hashes = {};
  payload.hashes.receipt_identity_sha256 = receipt_identity(payload);
  // Preflight canonical validation rejects unsupported/unsafe numerical fields.
  canonical_json(payload);
  const tmp = path.join(
    path.dirname(p),
    '.' + path.basename(p) + '.' + randomBytes(12).toString('hex')
  );
  let fd;
  try {
    fd = fs.openSync(tmp, 'wx', 0o600);
    fs.writeFileSync(fd, prettyText(payload) + '\n', 'ascii');
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = undefined;
    fs.renameSync(tmp, p);
  } catch (e) {
    if (fd !== undefined) fs.closeSync(fd);
    try {
      fs.unlinkSync(tmp);
    } catch {}
    throw e;
  }
}
export function read_utf8(p) {
  return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(fs.readFileSync(p));
}
export function read_json(p, { ascii = false } = {}) {
  if (ascii && fs.readFileSync(p).some((b) => b > 127))
    throw new Error('receipt must be ASCII JSON');
  return JSON.parse(read_utf8(p));
}
export function split_list(value) {
  let a;
  if (Array.isArray(value)) a = value.map(py_str);
  else if (value === null || value === undefined) return [];
  else {
    const text = py_strip(py_str(value));
    if (!text) return [];
    a = text.split(text.includes('|') ? '|' : ',');
  }
  return python_sort(new Set(a.map(py_strip).filter(Boolean)));
}
export function safe_relative_text(value) {
  const raw = py_strip(value).replaceAll('\\', '/');
  if (!raw || raw.startsWith('/') || raw.split('/').includes('..')) return null;
  const bits = raw.split('/').filter((s) => s && s !== '.');
  return bits.join('/') || '.';
}
export function receipt_safe_ref(v) {
  const safe = safe_relative_text(v);
  if (safe !== null) return safe;
  const name = path.basename(v);
  return name && name !== '.' && name !== '..' ? name : 'invalid-ref';
}
export function resolve_path(p) {
  let raw = String(p);
  if (raw === '~') raw = os.homedir();
  else if (raw.startsWith('~/')) raw = os.homedir() + raw.slice(1);
  let current = path.isAbsolute(raw) ? '/' : process.cwd();
  let parts = raw.split('/');
  let links = 0;
  while (parts.length) {
    const segment = parts.shift();
    if (!segment || segment === '.') continue;
    if (segment === '..') {
      current = path.dirname(current);
      continue;
    }
    const candidate = path.join(current, segment);
    let st;
    try {
      st = fs.lstatSync(candidate);
    } catch (e) {
      if (!['ENOENT', 'ENOTDIR'].includes(e.code)) throw e;
      current = candidate;
      continue;
    }
    if (st.isSymbolicLink()) {
      if (++links > 40) throw new Error('symlink resolution loop');
      const target = fs.readlinkSync(candidate);
      if (path.isAbsolute(target)) current = '/';
      parts = target.split('/').concat(parts);
    } else current = candidate;
  }
  return current;
}
export function relative_to_root(p, root) {
  const r = path.relative(resolve_path(root), resolve_path(p));
  if (r === '..' || r.startsWith('../') || path.isAbsolute(r))
    throw new Error('path is outside root');
  return r.split(path.sep).join('/') || '.';
}
export function within_root(p, root) {
  try {
    relative_to_root(p, root);
    return true;
  } catch {
    return false;
  }
}
export function unknown_diagnostic(mode, message, p = '') {
  return new Diagnostic('E-SRB-000', mode === 'strict' ? 'error' : 'warning', message, p);
}
function validateUnicode(v) {
  if (typeof v === 'string') {
    for (let i = 0; i < v.length; i++) {
      const c = v.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff) {
        const d = v.charCodeAt(++i);
        if (!(d >= 0xdc00 && d <= 0xdfff)) throw new Error('TOML contains an unpaired surrogate');
      } else if (c >= 0xdc00 && c <= 0xdfff) throw new Error('TOML contains an unpaired surrogate');
    }
  } else if (Array.isArray(v)) for (const x of v) validateUnicode(x);
  else if (is_record(v))
    for (const [k, x] of Object.entries(v)) {
      validateUnicode(k);
      validateUnicode(x);
    }
}
export function load_toml(p) {
  const data = parseToml(read_utf8(p), { integersAsBigInt: true });
  if (!is_record(data)) throw new Error('top-level TOML value must be a table');
  validateUnicode(data);
  return data;
}
