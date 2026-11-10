// Claim provenance checks: content identity only, never scientific certification.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {
  Diagnostic,
  CLAIM_SCHEMA,
  EMPIRICAL_CLASSES,
  EMPIRICAL_EVIDENCE,
  GUM_CLASSES,
  GUM_EVIDENCE,
  LEGACY_EVIDENCE_TYPES,
  is_sha256,
  py_str,
  py_strip,
  is_record,
  python_sort,
  safe_relative_text,
  receipt_safe_ref,
  resolve_path,
  within_root,
  relative_to_root,
  unknown_diagnostic,
  sha256_file,
  load_toml,
} from './core.mjs';

const get = (object, key, fallback = undefined) =>
  Object.hasOwn(object, key) ? object[key] : fallback;
const stripped = (value) => py_strip(py_str(value));
const is_file = (file) => {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
};
const difference = (left, right) => python_sort([...left].filter((value) => !right.has(value)));

// Schema-invalid containers fail UNKNOWN instead of relying on nonportable repr.
function claim_scalar_diagnostics(contract, mode, diagnostic_path) {
  if (!is_record(contract))
    return [unknown_diagnostic(mode, 'claim contract must be a table', diagnostic_path)];
  const diagnostics = [];
  for (const key of ['schema', 'claim-id', 'requested-class', 'context-of-use', 'root-artifact'])
    if (Object.hasOwn(contract, key) && (Array.isArray(contract[key]) || is_record(contract[key])))
      diagnostics.push(
        unknown_diagnostic(
          mode,
          'claim contract ' + key + ' must be scalar metadata',
          diagnostic_path
        )
      );
  return diagnostics;
}

function iterable(value) {
  if (Array.isArray(value) || typeof value === 'string') return value;
  if (is_record(value)) return Object.keys(value);
  throw new TypeError('claim evidence must be iterable');
}

export function load_claim_contract(path_text, mode) {
  if (!path_text) return [null, []];
  const file = resolve_path(path_text),
    diagnostics = [];
  let contract;
  try {
    contract = load_toml(file);
  } catch (error) {
    return [
      null,
      [
        unknown_diagnostic(
          mode,
          `cannot parse claim contract: ${error.message}`,
          path.basename(file)
        ),
      ],
    ];
  }
  diagnostics.push(...claim_scalar_diagnostics(contract, mode, path.basename(file)));
  if (contract.schema !== CLAIM_SCHEMA)
    diagnostics.push(
      unknown_diagnostic(mode, `claim contract schema must be ${CLAIM_SCHEMA}`, path.basename(file))
    );
  for (const key of ['claim-id', 'requested-class', 'context-of-use', 'root-artifact'])
    if (!stripped(get(contract, key, '')))
      diagnostics.push(
        unknown_diagnostic(mode, `claim contract is missing ${key}`, path.basename(file))
      );
  const evidence = get(contract, 'evidence', []);
  if (!Array.isArray(evidence))
    diagnostics.push(
      unknown_diagnostic(
        mode,
        'claim contract evidence must be an array of tables',
        path.basename(file)
      )
    );
  else
    for (const [index, item] of evidence.entries()) {
      const digest = is_record(item) ? stripped(get(item, 'sha256', '')) : '';
      if (
        !is_record(item) ||
        !stripped(get(item, 'type', '')) ||
        !stripped(get(item, 'ref', '')) ||
        !is_sha256(digest)
      )
        diagnostics.push(
          unknown_diagnostic(
            mode,
            `claim evidence ${index} requires type, ref, and lowercase SHA-256`,
            path.basename(file)
          )
        );
    }
  return [contract, diagnostics];
}

export function claim_diagnostics(contract, root_row, source_rel, source, policy, compiler_path) {
  if (contract === null || contract === undefined) return [];
  const shape_diagnostics = claim_scalar_diagnostics(contract, 'strict', source_rel);
  if (shape_diagnostics.length) return shape_diagnostics;
  const diagnostics = [],
    requested = stripped(get(contract, 'requested-class', '')),
    context = stripped(get(contract, 'context-of-use', '')),
    root_artifact = safe_relative_text(py_str(get(contract, 'root-artifact', ''))),
    evidence = iterable(get(contract, 'evidence', []));
  const evidence_types = new Set(
    [...evidence]
      .filter(is_record)
      .map((item) => stripped(get(item, 'type', '')))
      .filter(Boolean)
  );
  const root = policy !== null && policy !== undefined ? policy.root : path.dirname(source);
  const expected_source_sha = sha256_file(source);
  let expected_policy_ref = '',
    expected_policy_sha = '',
    expected_compiler_sha = '';
  if (
    policy !== null &&
    policy !== undefined &&
    policy.source !== null &&
    policy.source !== undefined &&
    is_file(policy.source)
  ) {
    expected_policy_ref = relative_to_root(policy.source, policy.root);
    expected_policy_sha = sha256_file(policy.source);
  }
  if (compiler_path && is_file(compiler_path)) expected_compiler_sha = sha256_file(compiler_path);
  for (const item of evidence) {
    if (!is_record(item)) continue;
    const type = stripped(get(item, 'type', '')),
      ref = stripped(get(item, 'ref', '')),
      digest = stripped(get(item, 'sha256', ''));
    const relative_ref = safe_relative_text(ref);
    let valid_binding = false;
    if (relative_ref !== null) {
      if (type === 'source')
        valid_binding = relative_ref === source_rel && digest === expected_source_sha;
      else if (type === 'package')
        valid_binding = ref === expected_policy_ref && digest === expected_policy_sha;
      else if (type === 'compiler')
        valid_binding = Boolean(expected_compiler_sha) && digest === expected_compiler_sha;
      else {
        const evidence_path = resolve_path(path.join(root, relative_ref));
        valid_binding =
          within_root(evidence_path, root) &&
          is_file(evidence_path) &&
          sha256_file(evidence_path) === digest;
      }
    }
    if (!valid_binding)
      diagnostics.push(
        new Diagnostic(
          'E-SRB-006',
          'error',
          `claim evidence is not bound to verified content: ${type}:${receipt_safe_ref(ref)}`,
          source_rel
        )
      );
  }
  const legacy = python_sort([...evidence_types].filter((type) => LEGACY_EVIDENCE_TYPES.has(type)));
  if (legacy.length)
    diagnostics.push(
      new Diagnostic(
        'E-SRB-004',
        'error',
        `legacy scalar metadata cannot serve as claim evidence: ${legacy.join(',')}`,
        source_rel
      )
    );
  if (root_artifact !== source_rel)
    diagnostics.push(
      new Diagnostic(
        'E-SRB-006',
        'error',
        `claim root-artifact must bind the compiled source (${source_rel})`,
        source_rel
      )
    );
  const required = new Set(['source', 'package', 'compiler']);
  if (EMPIRICAL_CLASSES.has(requested)) {
    required.add('model');
    required.add('data-manifest');
  }
  const missing_provenance = difference(required, evidence_types);
  if (missing_provenance.length)
    diagnostics.push(
      new Diagnostic(
        'E-SRB-006',
        'error',
        `claim lacks provenance bindings: ${missing_provenance.join(',')}`,
        source_rel
      )
    );
  if (GUM_CLASSES.has(requested)) {
    const missing = difference(GUM_EVIDENCE, evidence_types);
    if (missing.length)
      diagnostics.push(
        new Diagnostic(
          'E-SRB-004',
          'error',
          `GUM claim requires named method and witness evidence; missing ${missing.join(',')}`,
          source_rel
        )
      );
  }
  if (EMPIRICAL_CLASSES.has(requested)) {
    const missing = difference(EMPIRICAL_EVIDENCE, evidence_types);
    if (missing.length)
      diagnostics.push(
        new Diagnostic(
          'E-SRB-005',
          'error',
          `${requested} claim cannot be supported by compile/runtime evidence alone; missing ${missing.join(',')}`,
          source_rel
        )
      );
  }
  if (
    root_row === null ||
    root_row === undefined ||
    !root_row.allowed_claim_classes.includes(requested)
  )
    diagnostics.push(
      new Diagnostic(
        'E-SRB-007',
        'error',
        `requested claim class is not authorized by the root ring: ${requested}`,
        source_rel
      )
    );
  if (root_row !== null && root_row !== undefined && context !== root_row.context_of_use)
    diagnostics.push(
      new Diagnostic(
        'E-SRB-004',
        'error',
        'claim context-of-use does not match the root declaration',
        source_rel
      )
    );
  return diagnostics;
}

export function source_bundle_hash(nodes) {
  const sorted_paths = python_sort(nodes.map((node) => node.path)),
    rank = new Map(sorted_paths.map((p, index) => [p, index]));
  const digest = crypto.createHash('sha256');
  for (const node of [...nodes].sort((a, b) => rank.get(a.path) - rank.get(b.path))) {
    if (
      typeof node.path !== 'string' ||
      typeof node.sha256 !== 'string' ||
      /[^\x00-\x7f]/.test(node.sha256)
    )
      throw new TypeError('source bundle requires string paths and ASCII digest text');
    if (/[\uD800-\uDFFF]/u.test(node.path))
      throw new TypeError('source bundle path has a lone Unicode surrogate');
    digest.update(Buffer.from(node.path, 'utf8'));
    digest.update(Buffer.from([0]));
    digest.update(Buffer.from(node.sha256, 'ascii'));
    digest.update(Buffer.from('\n'));
  }
  return digest.digest('hex');
}

export function verify_claim_evidence_bindings(receipt, root, compiler_path) {
  const claim = get(receipt, 'claim_contract');
  if (!is_record(claim)) return;
  for (const binding of iterable(get(claim, 'evidence_bindings', []))) {
    // Malformed binding shape is rejected, as the legacy attribute access did.
    if (!is_record(binding)) throw new TypeError('claim evidence binding must be a table');
    const type = py_str(get(binding, 'type', '')),
      ref = safe_relative_text(py_str(get(binding, 'ref', ''))),
      expected = py_str(get(binding, 'sha256', ''));
    if (type === 'compiler') {
      if (
        compiler_path === null ||
        compiler_path === undefined ||
        sha256_file(compiler_path) !== expected
      )
        throw new Error('claim compiler evidence hash mismatch');
      continue;
    }
    if (ref === null) throw new Error(`claim evidence path is invalid: ${type}`);
    const evidence_path = resolve_path(path.join(root, ref));
    if (!within_root(evidence_path, root) || sha256_file(evidence_path) !== expected)
      throw new Error(`claim evidence hash mismatch: ${type}:${ref}`);
  }
}
