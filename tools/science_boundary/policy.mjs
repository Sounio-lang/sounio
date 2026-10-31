// Operational policy port of attestor.py; no numerical or scientific execution.
import fs from 'node:fs';
import path from 'node:path';
import {
  Diagnostic,
  PolicyRow,
  Policy,
  SCIENCE_SCHEMA,
  TSV_COLUMNS,
  CONCLUSIVE_RINGS,
  AUDITABLE_RINGS,
  VISIBILITIES,
  ENFORCEMENTS,
  EMPIRICAL_EVIDENCE,
  py_str,
  py_strip,
  py_truth,
  is_record,
  python_sort,
  split_list,
  safe_relative_text,
  resolve_path,
  within_root,
  unknown_diagnostic,
  read_utf8,
  load_toml,
} from './core.mjs';

const get = (object, key, fallback = undefined) =>
  Object.hasOwn(object, key) ? object[key] : fallback;
const stripped = (value) => py_strip(py_str(value));
const is_file = (value) => {
  try {
    return fs.statSync(value).isFile();
  } catch {
    return false;
  }
};

// Portability hardening for schema-invalid containers: Python/JS container repr
// is not an authority-bearing scalar encoding. Preserve valid scalar coercions.
const is_container = (value) => Array.isArray(value) || is_record(value);
const flat_string_list = (value) =>
  !is_record(value) && (!Array.isArray(value) || value.every((item) => typeof item === 'string'));
function policy_shape_diagnostics(science, mode) {
  const diagnostics = [];
  for (const key of [
    'ring',
    'evidence-status',
    'context-of-use',
    'visibility',
    'enforcement',
    'next-gate',
    'declared-by',
    'declared-at',
    'review-state',
  ]) {
    for (const alias of new Set([key, key.replaceAll('-', '_')]))
      if (Object.hasOwn(science, alias) && is_container(science[alias]))
        diagnostics.push(
          unknown_diagnostic(mode, '[science].' + alias + ' must be scalar metadata', 'sounio.toml')
        );
  }
  for (const key of [
    'allowed-claim-classes',
    'allowed_claim_classes',
    'evidence-refs',
    'evidence_refs',
  ])
    if (Object.hasOwn(science, key) && !flat_string_list(science[key]))
      diagnostics.push(
        unknown_diagnostic(mode, '[science].' + key + ' must be a flat string list', 'sounio.toml')
      );
  return diagnostics;
}

export function validate_policy_row(row, root, mode) {
  const diagnostics = [];
  if (!CONCLUSIVE_RINGS.has(row.ring) && !AUDITABLE_RINGS.has(row.ring))
    diagnostics.push(unknown_diagnostic(mode, `unknown ring: ${row.ring}`, row.path));
  if (!VISIBILITIES.has(row.visibility))
    diagnostics.push(unknown_diagnostic(mode, `unknown visibility: ${row.visibility}`, row.path));
  if (!ENFORCEMENTS.has(row.enforcement))
    diagnostics.push(unknown_diagnostic(mode, `unknown enforcement: ${row.enforcement}`, row.path));
  if (CONCLUSIVE_RINGS.has(row.ring) && !py_truth(row.evidence_refs))
    diagnostics.push(
      new Diagnostic(
        'E-SRB-003',
        'error',
        'conclusive ring declaration requires named evidence references',
        row.path
      )
    );
  for (const key of [
    'evidence_status',
    'context_of_use',
    'next_gate',
    'declared_by',
    'declared_at',
    'review_state',
  ])
    if (!py_truth(row[key])) diagnostics.push(unknown_diagnostic(mode, `missing ${key}`, row.path));
  const absolute = resolve_path(path.resolve(root, row.path));
  if (!within_root(absolute, root))
    diagnostics.push(unknown_diagnostic(mode, 'policy path escapes its root', row.path));
  else if (!fs.existsSync(absolute))
    diagnostics.push(unknown_diagnostic(mode, 'policy path does not exist', row.path));
  return diagnostics;
}

// csv.DictReader's default (non-strict) Excel dialect with TAB delimiter.
// Physical comment lines are filtered before CSV parsing, including while a
// quoted multiline field is open, exactly as the legacy generator did.
function* tsv_records(text) {
  text = (text.match(/[^\r\n]*(?:\r\n|\r|\n|$)/g) ?? [])
    .filter((line) => !line.startsWith('#'))
    .join('');
  let fields = [],
    value = '',
    state = 'start',
    touched = false,
    skip_lf = false,
    length = 0;
  const append = (ch) => {
    if (++length > 131072) throw new Error('field larger than field limit (131072)');
    value += ch;
  };
  const end_field = () => {
    fields.push(value);
    value = '';
    length = 0;
    state = 'start';
  };
  for (const ch of text) {
    if (skip_lf) {
      skip_lf = false;
      if (ch === '\n') continue;
    }
    if (state === 'quoted') {
      if (ch === '"') state = 'after_quote';
      else append(ch);
      continue;
    }
    if (state === 'after_quote' && ch === '"') {
      append('"');
      state = 'quoted';
      continue;
    }
    if (ch === '\t') {
      end_field();
      touched = true;
      continue;
    }
    if (ch === '\n' || ch === '\r') {
      if (fields.length || touched) {
        end_field();
        yield fields;
      } else yield [];
      fields = [];
      value = '';
      length = 0;
      state = 'start';
      touched = false;
      skip_lf = ch === '\r';
      continue;
    }
    touched = true;
    if (state === 'start' && ch === '"') state = 'quoted';
    else {
      append(ch);
      state = 'unquoted';
    }
  }
  // strict=False accepts an unterminated quoted field at EOF.
  if (fields.length || touched) {
    end_field();
    yield fields;
  }
}

export function load_tsv_policy(file, mode) {
  const root = resolve_path(path.dirname(file));
  const policy = new Policy({ root, source: resolve_path(file), kind: 'science-rings-tsv' });
  try {
    const records = tsv_records(read_utf8(file));
    const header = records.next().value ?? [];
    if (
      header.length !== TSV_COLUMNS.length ||
      header.some((value, index) => value !== TSV_COLUMNS[index])
    ) {
      policy.diagnostics.push(
        unknown_diagnostic(mode, 'science-rings.tsv header does not match schema v1')
      );
      return policy;
    }
    const seen = new Set();
    let line_number = 1;
    for (const fields of records) {
      if (!fields.length) continue; // DictReader ignores empty records after its header.
      line_number++;
      const values = Object.fromEntries(
        TSV_COLUMNS.map((key, index) => [key, index < fields.length ? fields[index] : null])
      );
      const path_text = safe_relative_text(values.path ?? '');
      if (path_text === null) {
        policy.diagnostics.push(
          unknown_diagnostic(mode, `invalid policy path on line ${line_number}`)
        );
        continue;
      }
      if (seen.has(path_text)) {
        policy.diagnostics.push(
          unknown_diagnostic(mode, `duplicate policy path on line ${line_number}`, path_text)
        );
        continue;
      }
      seen.add(path_text);
      const row = new PolicyRow({
        path: path_text,
        ...Object.fromEntries(
          [
            'ring',
            'evidence_status',
            'context_of_use',
            'visibility',
            'enforcement',
            'next_gate',
            'declared_by',
            'declared_at',
            'review_state',
          ].map((key) => [key, stripped(py_truth(values[key]) ? values[key] : '')])
        ),
        allowed_claim_classes: split_list(values.allowed_claim_classes),
        evidence_refs: split_list(values.evidence_refs),
      });
      policy.rows.push(row);
      policy.diagnostics.push(...validate_policy_row(row, root, mode));
    }
  } catch (error) {
    policy.diagnostics.push(
      unknown_diagnostic(mode, `cannot parse science-rings.tsv: ${error.message}`)
    );
  }
  if (!policy.rows.length)
    policy.diagnostics.push(unknown_diagnostic(mode, 'science-rings.tsv has no declarations'));
  return policy;
}

export function example_evidence_refs(example) {
  return split_list(get(example, 'evidence-refs', get(example, 'evidence_refs')));
}

export function validate_scientific_examples(data, root, mode) {
  const diagnostics = [],
    examples = get(data, 'example', []);
  if (!Array.isArray(examples))
    return [unknown_diagnostic(mode, '[[example]] must be an array of tables')];
  for (const [index, example] of examples.entries()) {
    if (!is_record(example)) {
      diagnostics.push(unknown_diagnostic(mode, `example ${index} is not a table`));
      continue;
    }
    const name = py_str(get(example, 'name', `example-${index}`));
    const maturity = stripped(get(example, 'maturity', ''));
    const context = stripped(get(example, 'context-of-use', get(example, 'context_of_use', '')));
    const refs_value = get(example, 'evidence-refs', get(example, 'evidence_refs'));
    if (!flat_string_list(refs_value))
      diagnostics.push(
        unknown_diagnostic(
          mode,
          'scientific example ' + name + ' evidence-refs must be a flat string list'
        )
      );
    const refs = example_evidence_refs(example);
    for (const [key, value] of [
      ['maturity', maturity],
      ['context-of-use', context],
      ['evidence-refs', refs],
    ])
      if (!py_truth(value))
        diagnostics.push(unknown_diagnostic(mode, `scientific example ${name} is missing ${key}`));
    const example_path = safe_relative_text(py_str(get(example, 'path', '')));
    if (example_path === null || !within_root(path.join(root, example_path || ''), root))
      diagnostics.push(unknown_diagnostic(mode, `scientific example ${name} has invalid path`));
    if (['calibrated', 'validated'].includes(maturity)) {
      const types = new Set(
        refs.filter((ref) => ref.includes(':')).map((ref) => ref.split(':', 1)[0])
      );
      const missing = python_sort([...EMPIRICAL_EVIDENCE].filter((type) => !types.has(type)));
      if (missing.length)
        diagnostics.push(
          unknown_diagnostic(
            mode,
            `scientific example ${name} maturity=${maturity} lacks typed evidence: ${missing.join(',')}`
          )
        );
    }
  }
  return diagnostics;
}

export function load_toml_policy(file, mode) {
  const root = resolve_path(path.dirname(file)),
    policy = new Policy({ root, source: resolve_path(file), kind: 'sounio-toml' });
  let data;
  try {
    data = load_toml(file);
  } catch (error) {
    policy.diagnostics.push(unknown_diagnostic(mode, `cannot parse sounio.toml: ${error.message}`));
    return policy;
  }
  if (is_record(data.epistemic)) {
    policy.legacy_present = true;
    policy.diagnostics.push(
      new Diagnostic(
        'W-SRB-LEGACY-001',
        'warning',
        'legacy [epistemic] metadata is read for compatibility and grants no boundary or claim authority',
        'sounio.toml'
      )
    );
  }
  const science = data.science;
  if (!is_record(science)) return policy;
  policy.diagnostics.push(...policy_shape_diagnostics(science, mode));
  if (stripped(get(science, 'schema', '')) !== SCIENCE_SCHEMA)
    policy.diagnostics.push(
      unknown_diagnostic(mode, `[science].schema must be ${SCIENCE_SCHEMA}`, 'sounio.toml')
    );
  const alias = (hyphen, underscore, fallback = '') =>
    get(science, hyphen, get(science, underscore, fallback));
  const row = new PolicyRow({
    path: '.',
    ring: stripped(get(science, 'ring', '')),
    evidence_status: stripped(alias('evidence-status', 'evidence_status')),
    context_of_use: stripped(alias('context-of-use', 'context_of_use')),
    visibility: stripped(get(science, 'visibility', '')),
    enforcement: 'advisory',
    next_gate: stripped(alias('next-gate', 'next_gate', 'package-boundary-receipt')),
    allowed_claim_classes: split_list(
      alias('allowed-claim-classes', 'allowed_claim_classes', undefined)
    ),
    evidence_refs: split_list(alias('evidence-refs', 'evidence_refs', undefined)),
    declared_by: stripped(alias('declared-by', 'declared_by', 'sounio.toml')),
    declared_at: stripped(alias('declared-at', 'declared_at', 'manifest-version')),
    review_state: stripped(alias('review-state', 'review_state', 'draft')),
  });
  policy.rows.push(row);
  policy.diagnostics.push(
    ...validate_policy_row(row, root, mode),
    ...validate_scientific_examples(data, root, mode)
  );
  return policy;
}

export function policy_has_science(file) {
  try {
    return is_record(load_toml(file).science);
  } catch {
    return true;
  }
}

export function discover_policy(source, explicit, mode) {
  if (py_truth(explicit)) {
    const file = resolve_path(explicit);
    if (!is_file(file))
      return new Policy({
        root: path.dirname(file),
        source: file,
        kind: 'missing',
        diagnostics: [
          unknown_diagnostic(mode, 'science manifest does not exist', path.basename(file)),
        ],
      });
    return path.basename(file) === 'sounio.toml' || path.extname(file) === '.toml'
      ? load_toml_policy(file, mode)
      : load_tsv_policy(file, mode);
  }
  for (let directory = path.dirname(resolve_path(source)); ; ) {
    const rings = path.join(directory, 'science-rings.tsv'),
      manifest = path.join(directory, 'sounio.toml');
    const has_rings = is_file(rings),
      has_manifest = is_file(manifest) && policy_has_science(manifest);
    if (has_rings && has_manifest)
      return new Policy({
        root: resolve_path(directory),
        source: null,
        kind: 'conflicting-declarations',
        diagnostics: [
          unknown_diagnostic(
            mode,
            'science-rings.tsv conflicts with [science] in sounio.toml at the same root'
          ),
        ],
      });
    if (has_rings) return load_tsv_policy(rings, mode);
    if (has_manifest) return load_toml_policy(manifest, mode);
    const parent = path.dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  return null;
}

export function classify_path(file, policy) {
  const resolved = resolve_path(file),
    matches = [];
  for (const row of policy.rows) {
    const absolute = resolve_path(path.resolve(policy.root, row.path));
    if (resolved === absolute || within_root(resolved, absolute))
      matches.push({ depth: absolute.split(path.sep).filter(Boolean).length, row });
  }
  if (!matches.length) return null;
  const paths = python_sort(matches.map((match) => match.row.path)),
    rank = new Map(paths.map((p, i) => [p, i]));
  matches.sort((a, b) => b.depth - a.depth || rank.get(a.row.path) - rank.get(b.row.path));
  return matches[0].row;
}
