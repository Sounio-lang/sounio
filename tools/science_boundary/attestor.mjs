#!/usr/bin/env node
// Operational policy/provenance metadata only. No scientific numerical reference.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  RECEIPT_SCHEMA,
  CONCLUSIVE_RINGS,
  ALLOWED_DEPENDENCIES,
  EXIT_REJECT,
  EXIT_UNKNOWN,
  Diagnostic,
  is_record,
  is_sha256,
  py_str,
  py_strip,
  python_sort,
  safe_relative_text,
  receipt_safe_ref,
  resolve_path,
  within_root,
  relative_to_root,
  sha256_file,
  receipt_identity,
  write_json_atomic,
  read_json,
  unknown_diagnostic,
} from './core.mjs';
import { discover_policy, classify_path } from './policy.mjs';
import { collect_closure, load_raw_ast_closure_report } from './closure.mjs';
import {
  load_claim_contract,
  claim_diagnostics,
  source_bundle_hash,
  verify_claim_evidence_bindings,
} from './claims.mjs';
const isFile = (p) => {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
};
const keysAre = (v, keys) =>
  is_record(v) && JSON.stringify(Object.keys(v).sort()) === JSON.stringify([...keys].sort());
const need = (ok, message) => {
  if (!ok) throw new Error(message);
};
export function build_receipt({
  source,
  policy,
  requested_mode,
  manifest_path = '',
  claim_path = '',
  compiler_path = '',
  engine_identity = '',
  closure_report_path = '',
  elf_path = '',
  artifact_label = '',
}) {
  let mode = requested_mode;
  if (mode === 'auto')
    mode =
      (policy && (policy.declared || Boolean(manifest_path))) || Boolean(claim_path)
        ? 'advisory'
        : 'off';
  const root = resolve_path(policy ? policy.root : path.dirname(resolve_path(source)));
  const diagnostics = policy ? [...policy.diagnostics] : [];
  if (mode !== 'off' && !policy)
    diagnostics.push(unknown_diagnostic(mode, 'no scientific declaration was discovered'));
  let closure = null;
  const rows = new Map(),
    nodes = [],
    edges = [];
  let unresolved = [];
  let source_rel = path.basename(source);
  if (fs.existsSync(source) && within_root(source, root))
    source_rel = relative_to_root(source, root);
  else if (mode !== 'off')
    diagnostics.push(
      unknown_diagnostic(mode, 'source path escapes policy root', path.basename(source))
    );
  if (mode !== 'off' && policy && policy.valid && within_root(source, root)) {
    if (closure_report_path) {
      try {
        const [c, ds] = load_raw_ast_closure_report(
          resolve_path(closure_report_path),
          resolve_path(source),
          root
        );
        closure = c;
        diagnostics.push(...ds);
      } catch (e) {
        if (!e.code) throw e;
        closure = collect_closure(resolve_path(source), root);
        diagnostics.push(
          unknown_diagnostic(mode, `cannot hash raw AST closure report: ${e.message}`)
        );
      }
    } else {
      closure = collect_closure(resolve_path(source), root);
      diagnostics.push(
        unknown_diagnostic(
          mode,
          'raw Madaros AST closure report is required for an authoritative verdict; host syntax audit is non-authoritative'
        )
      );
    }
    for (const [current, message] of closure.parse_failures)
      diagnostics.push(
        unknown_diagnostic(
          mode,
          `closure parser incomplete: ${message}`,
          relative_to_root(current, root)
        )
      );
    for (const [current, imported] of closure.unresolved)
      diagnostics.push(
        unknown_diagnostic(
          mode,
          `unresolved import in authoritative closure: ${imported}`,
          relative_to_root(current, root)
        )
      );
    if (closure.saturated)
      diagnostics.push(
        unknown_diagnostic(mode, `module closure reached capacity ${closure.capacity}`)
      );
    for (const node of closure.nodes) {
      const rel = relative_to_root(node, root),
        row = classify_path(node, policy);
      rows.set(rel, row);
      let ring;
      if (!row) {
        diagnostics.push(unknown_diagnostic(mode, 'module is unclassified', rel));
        ring = {
          ring: 'unclassified',
          evidence_status: 'unknown',
          context_of_use: 'undeclared',
          visibility: 'protected',
          review_state: 'unreviewed',
        };
      } else {
        ring = {
          ring: row.ring,
          evidence_status: row.evidence_status,
          context_of_use: row.context_of_use,
          visibility: row.visibility,
          review_state: row.review_state,
        };
        if (!CONCLUSIVE_RINGS.has(row.ring))
          diagnostics.push(unknown_diagnostic(mode, `non-conclusive ring: ${row.ring}`, rel));
      }
      nodes.push({ path: rel, sha256: sha256_file(node), ...ring });
    }
    for (const [caller, dependency] of closure.edges) {
      const cr = relative_to_root(caller, root),
        dr = relative_to_root(dependency, root),
        a = rows.get(cr),
        b = rows.get(dr);
      edges.push({ caller: cr, dependency: dr });
      if (!a || !b) continue;
      if (
        closure.collector === 'madaros-raw-ast-v1' &&
        CONCLUSIVE_RINGS.has(a.ring) &&
        CONCLUSIVE_RINGS.has(b.ring) &&
        !ALLOWED_DEPENDENCIES[a.ring].has(b.ring)
      )
        diagnostics.push(
          new Diagnostic(
            'E-SRB-001',
            'error',
            `ring dependency is forbidden: ${a.ring} -> ${b.ring}`,
            cr,
            dr
          )
        );
      if (
        closure.collector === 'madaros-raw-ast-v1' &&
        a.visibility === 'public' &&
        ['protected', 'embargoed'].includes(b.visibility) &&
        !(a.ring === 'research' && b.ring === 'scientific-package-candidate')
      )
        diagnostics.push(
          new Diagnostic(
            'E-SRB-002',
            'error',
            `public module cannot depend on ${b.visibility} module`,
            cr,
            dr
          )
        );
    }
    unresolved = closure.unresolved.map(([p, imported]) => ({
      caller: relative_to_root(p, root),
      import: imported,
    }));
  } else if (fs.existsSync(source))
    nodes.push({
      path: source_rel,
      sha256: sha256_file(source),
      ring: 'unclassified',
      evidence_status: 'not-evaluated',
      context_of_use: 'not-evaluated',
      visibility: 'protected',
      review_state: 'not-evaluated',
    });
  const [contract, claimDs] = load_claim_contract(claim_path, mode);
  diagnostics.push(...claimDs);
  if (claim_path && !within_root(resolve_path(claim_path), root))
    diagnostics.push(unknown_diagnostic(mode, 'claim contract path escapes policy root'));
  if (mode !== 'off' && !claimDs.length && isFile(source))
    diagnostics.push(
      ...claim_diagnostics(
        contract,
        rows.get(source_rel) || null,
        source_rel,
        source,
        policy,
        compiler_path
      )
    );
  let compiler_sha = '';
  if (compiler_path) {
    try {
      compiler_sha = sha256_file(resolve_path(compiler_path));
    } catch (e) {
      diagnostics.push(unknown_diagnostic(mode, `cannot hash compiler: ${e.message}`));
    }
  } else if (mode === 'strict')
    diagnostics.push(unknown_diagnostic(mode, 'strict receipt requires compiler identity'));
  if (mode === 'strict' && !py_strip(engine_identity))
    diagnostics.push(unknown_diagnostic(mode, 'strict receipt requires compiler engine identity'));
  const reject = diagnostics.some((d) => /^E-SRB-00[1-7]$/.test(d.code)),
    unknown = diagnostics.some((d) => d.code === 'E-SRB-000');
  const verdict = mode === 'off' || unknown ? 'UNKNOWN' : reject ? 'REJECT' : 'OK';
  let policy_sha = '',
    policy_source = '',
    policy_kind = 'none';
  if (policy && policy.source && isFile(policy.source)) {
    policy_sha = sha256_file(policy.source);
    policy_source = relative_to_root(policy.source, policy.root);
    policy_kind = policy.kind;
  }
  let claim_sha = '',
    summary = null;
  if (claim_path && isFile(claim_path)) claim_sha = sha256_file(claim_path);
  if (contract !== null) {
    const cp = resolve_path(claim_path),
      evidence = Array.isArray(contract.evidence) ? contract.evidence : [];
    summary = {
      source: within_root(cp, root) ? relative_to_root(cp, root) : path.basename(cp),
      claim_id: py_str(contract['claim-id'] ?? ''),
      requested_class: py_str(contract['requested-class'] ?? ''),
      context_of_use: py_str(contract['context-of-use'] ?? ''),
      root_artifact: py_str(contract['root-artifact'] ?? ''),
      evidence_types: python_sort(
        new Set(
          evidence
            .filter((x) => is_record(x) && py_str(x.type ?? ''))
            .map((x) => py_str(x.type ?? ''))
        )
      ),
      evidence_bindings: python_sort(
        evidence
          .filter(is_record)
          .map((x) => ({
            type: py_str(x.type ?? ''),
            ref: receipt_safe_ref(py_str(x.ref ?? '')),
            sha256: py_str(x.sha256 ?? ''),
          })),
        (x) => [x.type, x.ref, x.sha256]
      ),
    };
  }
  const sortedNodes = python_sort(nodes, (x) => x.path),
    sortedEdges = python_sort(edges, (x) => [x.caller, x.dependency]);
  const sortedDs = python_sort(diagnostics, (x) => [x.code, x.path, x.dependency, x.message]);
  const hashes = {
    source_bundle_sha256: source_bundle_hash(sortedNodes),
    policy_sha256: policy_sha,
    claim_contract_sha256: claim_sha,
    compiler_sha256: compiler_sha,
    closure_report_sha256: closure ? closure.report_sha256 : '',
    elf_sha256: elf_path ? sha256_file(elf_path) : '',
  };
  const receipt = {
    schema: RECEIPT_SCHEMA,
    verdict,
    mode,
    identity_only: true,
    graph: {
      root_artifact: source_rel,
      capacity: closure ? closure.capacity : 0,
      saturated: closure ? closure.saturated : false,
      nodes: sortedNodes,
      edges: sortedEdges,
      unresolved_imports: unresolved,
    },
    diagnostics: sortedDs.map((d) => d.to_dict()),
    hashes,
    engine: {
      name: 'madaros',
      identity: py_strip(engine_identity),
      boundary_collector: closure ? closure.collector : 'none',
    },
    policy: { kind: policy_kind, source: policy_source },
    claim_contract: summary,
    artifact: {
      kind: elf_path ? 'native-elf' : 'not-emitted',
      path:
        safe_relative_text(artifact_label) || (artifact_label ? path.basename(artifact_label) : ''),
    },
    assurance_level: 'identity-only',
    limitations: [
      'does_not_assert_scientific_truth',
      'does_not_assert_clinical_validation_or_clinical_authority',
      'does_not_assert_security_sandboxing',
      'does_not_assert_public_registry_status',
      'does_not_assert_attested_execution_or_independent_replay',
      closure && closure.collector === 'sounio-host-syntax-v1'
        ? 'host_syntax_closure_is_advisory_only'
        : 'raw_ast_closure_report_is_not_attested_execution',
    ],
  };
  return [receipt, verdict, sortedDs];
}
export function print_diagnostics(diagnostics, verdict, mode) {
  for (const d of diagnostics)
    console.error(
      `${d.severity}[${d.code}]:${d.path ? ' ' + d.path : ''}${d.dependency ? ' -> ' + d.dependency : ''} ${d.message}`
    );
  if (mode !== 'off') console.error(`science-boundary: mode=${mode} verdict=${verdict}`);
}
export function evaluate_command(args) {
  const source = resolve_path(args.source),
    policy = discover_policy(source, args.manifest, args.mode);
  const [r, v, ds] = build_receipt({
    source,
    policy,
    requested_mode: args.mode,
    manifest_path: args.manifest,
    claim_path: args.claim_contract,
    compiler_path: args.compiler,
    engine_identity: args.engine_identity,
    closure_report_path: args.closure_report,
    elf_path: args.elf,
    artifact_label: args.artifact_label,
  });
  if (args.receipt) write_json_atomic(resolve_path(args.receipt), r);
  print_diagnostics(ds, v, r.mode);
  return v === 'REJECT' ? EXIT_REJECT : v === 'UNKNOWN' && r.mode !== 'off' ? EXIT_UNKNOWN : 0;
}
export function verify_preflight_identity(r) {
  const expected = r?.hashes?.receipt_identity_sha256;
  need(expected && expected === receipt_identity(r), 'receipt identity hash mismatch');
}
export function validate_receipt_structure(r) {
  need(
    keysAre(r, [
      'schema',
      'verdict',
      'mode',
      'identity_only',
      'graph',
      'diagnostics',
      'hashes',
      'engine',
      'policy',
      'claim_contract',
      'artifact',
      'assurance_level',
      'limitations',
    ]),
    'receipt fields do not match schema v1'
  );
  need(r.schema === RECEIPT_SCHEMA, 'bad receipt schema');
  need(['OK', 'REJECT', 'UNKNOWN'].includes(r.verdict), 'bad receipt verdict');
  need(['off', 'advisory', 'strict'].includes(r.mode), 'bad receipt mode');
  need(
    r.identity_only === true && r.assurance_level === 'identity-only',
    'bad receipt assurance level'
  );
  const g = r.graph;
  need(
    keysAre(g, ['root_artifact', 'capacity', 'saturated', 'nodes', 'edges', 'unresolved_imports']),
    'bad receipt graph'
  );
  need(safe_relative_text(py_str(g.root_artifact ?? '')) !== null, 'bad receipt root artifact');
  need(Number.isSafeInteger(g.capacity) && g.capacity >= 0, 'bad receipt graph capacity');
  need(typeof g.saturated === 'boolean', 'bad receipt graph saturation');
  need(
    Array.isArray(g.nodes) && Array.isArray(g.edges) && Array.isArray(g.unresolved_imports),
    'bad receipt graph arrays'
  );
  for (const n of g.nodes)
    need(
      is_record(n) &&
        safe_relative_text(py_str(n.path ?? '')) !== null &&
        is_sha256(n.sha256 ?? ''),
      'bad receipt graph node'
    );
  for (const e of g.edges)
    need(
      is_record(e) &&
        safe_relative_text(py_str(e.caller ?? '')) !== null &&
        safe_relative_text(py_str(e.dependency ?? '')) !== null,
      'bad receipt graph edge'
    );
  for (const u of g.unresolved_imports)
    need(
      is_record(u) &&
        safe_relative_text(py_str(u.caller ?? '')) !== null &&
        Boolean(py_str(u.import ?? '')),
      'bad receipt unresolved import'
    );
  need(Array.isArray(r.diagnostics), 'bad receipt diagnostics');
  for (const d of r.diagnostics)
    need(
      is_record(d) &&
        Boolean(py_str(d.code ?? '')) &&
        ['warning', 'error'].includes(d.severity) &&
        Boolean(py_str(d.message ?? '')),
      'bad receipt diagnostic'
    );
  const hashFields = [
    'source_bundle_sha256',
    'policy_sha256',
    'claim_contract_sha256',
    'compiler_sha256',
    'closure_report_sha256',
    'elf_sha256',
    'receipt_identity_sha256',
  ];
  need(keysAre(r.hashes, hashFields), 'bad receipt hashes');
  need(is_sha256(r.hashes.source_bundle_sha256), 'bad source bundle hash');
  need(is_sha256(r.hashes.receipt_identity_sha256), 'bad receipt identity hash');
  for (const key of hashFields.filter(
    (k) => !['source_bundle_sha256', 'receipt_identity_sha256'].includes(k)
  ))
    need(is_sha256(r.hashes[key], { allow_empty: true }), `bad receipt hash: ${key}`);
  need(
    keysAre(r.engine, ['name', 'identity', 'boundary_collector']) &&
      r.engine.name === 'madaros' &&
      ['none', 'madaros-raw-ast-v1', 'sounio-host-syntax-v1'].includes(r.engine.boundary_collector),
    'bad receipt engine'
  );
  need(keysAre(r.policy, ['kind', 'source']), 'bad receipt policy');
  need(
    !r.policy.source || safe_relative_text(py_str(r.policy.source)) !== null,
    'bad receipt policy path'
  );
  const c = r.claim_contract;
  if (c !== null) {
    need(
      keysAre(c, [
        'source',
        'claim_id',
        'requested_class',
        'context_of_use',
        'root_artifact',
        'evidence_types',
        'evidence_bindings',
      ]) &&
        safe_relative_text(py_str(c.source ?? '')) !== null &&
        Boolean(py_str(c.claim_id ?? '')) &&
        Boolean(py_str(c.requested_class ?? '')) &&
        Boolean(py_str(c.context_of_use ?? '')) &&
        safe_relative_text(py_str(c.root_artifact ?? '')) !== null &&
        Array.isArray(c.evidence_types) &&
        c.evidence_types.every((x) => typeof x === 'string' && x) &&
        new Set(c.evidence_types).size === c.evidence_types.length,
      'bad receipt claim contract'
    );
    need(Array.isArray(c.evidence_bindings), 'bad receipt claim evidence binding');
    for (const b of c.evidence_bindings)
      need(
        keysAre(b, ['type', 'ref', 'sha256']) &&
          Boolean(py_str(b.type ?? '')) &&
          safe_relative_text(py_str(b.ref ?? '')) !== null &&
          is_sha256(b.sha256 ?? ''),
        'bad receipt claim evidence binding'
      );
  }
  need(
    keysAre(r.artifact, ['kind', 'path']) &&
      ['not-emitted', 'native-elf'].includes(r.artifact.kind) &&
      (!r.artifact.path || safe_relative_text(py_str(r.artifact.path)) !== null),
    'bad receipt artifact'
  );
  if (r.artifact.kind === 'native-elf')
    need(is_sha256(r.hashes.elf_sha256), 'native ELF receipt lacks an ELF hash');
  need(
    Array.isArray(r.limitations) && r.limitations.every((x) => typeof x === 'string' && x),
    'bad receipt limitations'
  );
}
export function finalize_command(args) {
  try {
    const r = read_json(args.preflight_receipt, { ascii: true });
    verify_preflight_identity(r);
    validate_receipt_structure(r);
    if (r.mode === 'strict')
      need(r.verdict === 'OK', 'strict receipt finalization requires an OK preflight');
    const source = resolve_path(args.source),
      policy = discover_policy(source, args.manifest, r.mode || 'advisory'),
      root = resolve_path(policy ? policy.root : path.dirname(source));
    need(
      source === resolve_path(path.join(root, r.graph.root_artifact)) &&
        r.graph.nodes.some((n) => n.path === r.graph.root_artifact),
      'finalization source differs from preflight root artifact'
    );
    for (const n of r.graph.nodes) {
      const p = resolve_path(path.join(root, n.path));
      need(
        within_root(p, root) && sha256_file(p) === n.sha256,
        `source changed after boundary preflight: ${n.path}`
      );
    }
    need(
      source_bundle_hash(r.graph.nodes) === r.hashes.source_bundle_sha256,
      'source bundle hash mismatch'
    );
    if (r.hashes.policy_sha256)
      need(
        policy && policy.source && sha256_file(policy.source) === r.hashes.policy_sha256,
        'policy changed after boundary preflight'
      );
    if (args.claim_contract)
      need(
        sha256_file(args.claim_contract) === r.hashes.claim_contract_sha256,
        'claim contract changed after boundary preflight'
      );
    else
      need(!r.hashes.claim_contract_sha256, 'claim contract is required for receipt finalization');
    // Finalization must bind a recorded compiler, matching verification's rule.
    if (r.hashes.compiler_sha256)
      need(Boolean(args.compiler), 'compiler is required for receipt finalization');
    if (args.compiler)
      need(
        sha256_file(args.compiler) === r.hashes.compiler_sha256,
        'compiler changed after boundary preflight'
      );
    verify_claim_evidence_bindings(r, root, args.compiler ? resolve_path(args.compiler) : null);
    const elf = resolve_path(args.elf);
    need(isFile(elf) && fs.statSync(elf).size > 0, 'compiler artifact is missing or empty');
    r.hashes.elf_sha256 = sha256_file(elf);
    r.artifact = {
      kind: 'native-elf',
      path: safe_relative_text(args.artifact_label) || path.basename(args.artifact_label),
    };
    write_json_atomic(resolve_path(args.receipt), r);
    return 0;
  } catch (e) {
    console.error(`error[E-SRB-000]: receipt finalization failed: ${e.message}`);
    return 2;
  }
}
export function verify_command(args) {
  try {
    const r = read_json(args.receipt, { ascii: true });
    verify_preflight_identity(r);
    validate_receipt_structure(r);
    const root = resolve_path(args.root);
    for (const n of r.graph.nodes) {
      const p = resolve_path(path.join(root, n.path));
      need(within_root(p, root) && sha256_file(p) === n.sha256, `source hash mismatch: ${n.path}`);
    }
    need(
      source_bundle_hash(r.graph.nodes) === r.hashes.source_bundle_sha256,
      'source bundle hash mismatch'
    );
    if (r.policy.source) {
      const p = resolve_path(path.join(root, r.policy.source));
      need(within_root(p, root), 'policy path escapes verification root');
      need(sha256_file(p) === r.hashes.policy_sha256, 'policy hash mismatch');
    }
    const cs = r.claim_contract?.source || '';
    if (cs) {
      const p = resolve_path(path.join(root, cs));
      need(within_root(p, root), 'claim path escapes verification root');
      need(sha256_file(p) === r.hashes.claim_contract_sha256, 'claim contract hash mismatch');
    }
    if (r.hashes.compiler_sha256)
      need(Boolean(args.compiler), 'compiler is required to verify this receipt');
    if (args.compiler)
      need(sha256_file(args.compiler) === r.hashes.compiler_sha256, 'compiler hash mismatch');
    verify_claim_evidence_bindings(r, root, args.compiler ? resolve_path(args.compiler) : null);
    if (r.hashes.elf_sha256) need(Boolean(args.elf), 'ELF is required to verify this receipt');
    if (args.elf) need(sha256_file(args.elf) === r.hashes.elf_sha256, 'ELF hash mismatch');
    console.log(`PACKAGE_BOUNDARY_RECEIPT_VERIFY_PASS verdict=${r.verdict} mode=${r.mode}`);
    return 0;
  } catch (e) {
    console.error(`PACKAGE_BOUNDARY_RECEIPT_VERIFY_FAIL reason=${e.message}`);
    return 1;
  }
}
export function main(argv) {
  try {
    const command = argv[0];
    if (argv.includes('--help') || argv.includes('-h')) {
      console.log('attestor.mjs evaluate|finalize|verify --source PATH --receipt PATH [options]');
      return 0;
    }
    const allowed = {
      evaluate: [
        'source',
        'mode',
        'manifest',
        'claim-contract',
        'receipt',
        'compiler',
        'engine-identity',
        'closure-report',
        'elf',
        'artifact-label',
      ],
      finalize: [
        'preflight-receipt',
        'source',
        'manifest',
        'claim-contract',
        'compiler',
        'elf',
        'artifact-label',
        'receipt',
      ],
      verify: ['receipt', 'root', 'compiler', 'elf'],
    };
    need(Object.hasOwn(allowed, command), 'expected evaluate, finalize or verify');
    const args = {
      source: '',
      mode: 'auto',
      manifest: '',
      claim_contract: '',
      receipt: '',
      compiler: '',
      engine_identity: '',
      closure_report: '',
      elf: '',
      artifact_label: '',
      root: process.cwd(),
      preflight_receipt: '',
    };
    for (let i = 1; i < argv.length; i++) {
      const m = /^--([^=]+)(?:=([\s\S]*))?$/.exec(argv[i]);
      need(m && allowed[command].includes(m[1]), `unrecognized argument: ${argv[i]}`);
      const value = m[2] !== undefined ? m[2] : argv[++i];
      need(value !== undefined, `missing value: --${m[1]}`);
      args[m[1].replaceAll('-', '_')] = value;
    }
    const required = {
      evaluate: ['source'],
      finalize: ['preflight_receipt', 'source', 'elf', 'artifact_label', 'receipt'],
      verify: ['receipt'],
    };
    for (const k of required[command])
      need(Boolean(args[k]), `required argument: --${k.replaceAll('_', '-')}`);
    need(['auto', 'off', 'advisory', 'strict'].includes(args.mode), 'invalid mode');
    return command === 'evaluate'
      ? evaluate_command(args)
      : command === 'finalize'
        ? finalize_command(args)
        : verify_command(args);
  } catch (e) {
    console.error(`error[E-SRB-000]: boundary attestor failed: ${e.message}`);
    return 2;
  }
}
if (process.argv[1] && resolve_path(process.argv[1]) === fileURLToPath(import.meta.url))
  process.exitCode = main(process.argv.slice(2));
