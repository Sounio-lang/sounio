// Literal operational port of attestor.py's syntax/closure surface.
// This module hashes and classifies metadata; it never runs scientific code.
import fs from 'node:fs';
import path from 'node:path';
import {
  ImportSpec,
  Closure,
  resolve_path,
  within_root,
  relative_to_root,
  sha256_file,
  read_utf8,
  python_sort,
} from './core.mjs';

const alpha = /^\p{L}$/u;
const alnum = /^[\p{L}\p{N}]$/u;
const decimal = /^\p{Nd}$/u;
const universal_newlines = (text) => text.replace(/\r\n?/g, '\n');

function is_file(value) {
  try {
    return fs.statSync(value).isFile();
  } catch (error) {
    if (['ENOENT', 'ENOTDIR', 'EINVAL', 'ELOOP'].includes(error.code)) return false;
    throw error;
  }
}

// Python int(string) accepts signs, underscores and Unicode decimal digits.
function capacity_integer(value) {
  // CPython's int scanner differs from str.strip: BOM and ASCII FS-US refuse.
  const text = value.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, '');
  if (!/^[+-]?\p{Nd}(?:_?\p{Nd})*$/u.test(text)) throw new Error('invalid capacity');
  // The tomllib-capable CPython baseline has a default 4300 decimal digit cap.
  if (Array.from(text).filter((char) => decimal.test(char)).length > 4300)
    throw new Error('invalid capacity');
  let ascii = '';
  for (const char of text) {
    if (char === '_') continue;
    if (char === '+' || char === '-') {
      ascii += char;
      continue;
    }
    const code = char.codePointAt(0);
    let first = code;
    while (first > 0 && decimal.test(String.fromCodePoint(first - 1))) first--;
    ascii += String((code - first) % 10);
  }
  return Number(ascii);
}
const bounded_capacity = (value) => Math.max(1, Math.min(capacity_integer(value), 4096));
const suffix_sio = (value) => value.slice(0, value.length - path.extname(value).length) + '.sio';

export function tokenize_import_surface(text) {
  const tokens = [],
    errors = [];
  let i = 0,
    line = 1;
  while (i < text.length) {
    const char = String.fromCodePoint(text.codePointAt(i));
    if (' \t\r'.includes(char)) {
      i += char.length;
      continue;
    }
    if (char === '\n') {
      tokens.push(['newline', '\n', line]);
      line++;
      i++;
      continue;
    }
    if (text.startsWith('//', i) || char === '#') {
      const end = text.indexOf('\n', i);
      i = end < 0 ? text.length : end;
      continue;
    }
    if (text.startsWith('/*', i)) {
      const start_line = line;
      let depth = 1;
      i += 2;
      while (i < text.length && depth) {
        if (text.startsWith('/*', i)) {
          depth++;
          i += 2;
        } else if (text.startsWith('*/', i)) {
          depth--;
          i += 2;
        } else {
          if (text[i] === '\n') line++;
          i++;
        }
      }
      if (depth) errors.push('unterminated block comment starting on line ' + start_line);
      continue;
    }
    if (char === '"' || char === "'") {
      const quote = char,
        start_line = line;
      let escaped = false,
        closed = false;
      i++;
      while (i < text.length) {
        const current = text[i];
        if (current === '\n') line++;
        if (escaped) escaped = false;
        else if (current === '\\') escaped = true;
        else if (current === quote) {
          i++;
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) errors.push('unterminated string starting on line ' + start_line);
      continue;
    }
    if (alpha.test(char) || char === '_') {
      const start = i;
      i += char.length;
      while (i < text.length) {
        const next = String.fromCodePoint(text.codePointAt(i));
        if (!(alnum.test(next) || next === '_')) break;
        i += next.length;
      }
      tokens.push(['ident', text.slice(start, i), line]);
      continue;
    }
    if (text.startsWith('::', i)) {
      tokens.push(['symbol', '::', line]);
      i += 2;
      continue;
    }
    tokens.push(['symbol', char, line]);
    i += char.length;
  }
  return [tokens, errors];
}

export function parse_imports(file) {
  let text;
  try {
    text = universal_newlines(read_utf8(file));
  } catch (error) {
    return [[], ['cannot read source: ' + error.message]];
  }
  const [tokens, errors] = tokenize_import_surface(text);
  const imports = [];
  let depth = 0,
    index = 0;
  while (index < tokens.length) {
    const [, value, line] = tokens[index];
    let import_index = index;
    if (depth === 0 && value === 'pub' && index + 1 < tokens.length) {
      if (['use', 'import'].includes(tokens[index + 1][1])) import_index = index + 1;
    }
    if (depth === 0 && ['use', 'import'].includes(tokens[import_index][1])) {
      let cursor = import_index + 1,
        expect_ident = true;
      const segments = [];
      while (cursor < tokens.length) {
        const [token_kind, token_value] = tokens[cursor];
        if (['\n', ';', '{', '*', ','].includes(token_value)) break;
        if (expect_ident && token_kind === 'ident') {
          segments.push(token_value);
          expect_ident = false;
        } else if (!expect_ident && token_value === '::') expect_ident = true;
        else break;
        cursor++;
      }
      if (segments.length) imports.push(new ImportSpec(segments, line));
      else errors.push('invalid import on line ' + line);
      while (cursor < tokens.length && !['\n', ';'].includes(tokens[cursor][1])) cursor++;
      index = cursor + 1;
      continue;
    }
    if (value === '{') depth++;
    else if (value === '}' && depth > 0) depth--;
    index++;
  }
  return [imports, errors];
}

export function* candidate_import_paths(source, spec, root) {
  const relative = suffix_sio(path.join(...spec.segments));
  const module_relative = path.join(...spec.segments, 'mod.sio');
  let current = resolve_path(path.dirname(source));
  const seen = new Set();
  while (true) {
    for (const suffix of [relative, module_relative]) {
      const candidate = resolve_path(path.resolve(current, suffix));
      if (!seen.has(candidate)) {
        seen.add(candidate);
        yield candidate;
      }
    }
    if (current === root || !within_root(path.dirname(current), root)) break;
    current = path.dirname(current);
  }
  const stdlib_root = process.env.SOUNIO_STDLIB_PATH || '';
  if (stdlib_root) {
    for (const suffix of [relative, module_relative])
      yield resolve_path(path.resolve(stdlib_root, suffix));
  }
  for (const prefix of [path.join(root, 'stdlib')]) {
    for (const suffix of [relative, module_relative])
      yield resolve_path(path.resolve(prefix, suffix));
  }
  const package_name = spec.segments[0].replaceAll('_', '-');
  const package_root = path.join(root, 'packages', package_name, 'src');
  if (spec.segments.length === 1) yield resolve_path(path.join(package_root, 'lib.sio'));
  else
    yield resolve_path(path.join(package_root, suffix_sio(path.join(...spec.segments.slice(1)))));
}

export function resolve_import(source, spec, root) {
  for (const candidate of candidate_import_paths(source, spec, root))
    if (is_file(candidate)) return candidate;
  return null;
}

export function collect_closure(source, root) {
  let capacity;
  try {
    capacity = bounded_capacity(process.env.SOUNIO_SCIENCE_BOUNDARY_MAX_NODES ?? '256');
  } catch {
    capacity = 256;
  }
  const closure = new Closure({ root, source });
  closure.capacity = capacity;
  const queue = [resolve_path(source)],
    visited = new Set();
  while (queue.length) {
    const current = queue.shift();
    if (visited.has(current)) continue;
    if (visited.size >= capacity) {
      closure.saturated = true;
      break;
    }
    visited.add(current);
    closure.nodes.push(current);
    const [imports, errors] = parse_imports(current);
    closure.parse_failures.push(...errors.map((error) => [current, error]));
    for (const spec of imports) {
      const resolved = resolve_import(current, spec, root);
      if (resolved === null) {
        closure.unresolved.push([current, spec.display + '@' + spec.line]);
        continue;
      }
      if (!within_root(resolved, root)) {
        closure.unresolved.push([current, 'root-escape:' + spec.display + '@' + spec.line]);
        continue;
      }
      closure.edges.push([current, resolved]);
      if (!visited.has(resolved) && !queue.includes(resolved)) {
        if (visited.size + queue.length >= capacity) closure.saturated = true;
        else queue.push(resolved);
      }
    }
  }
  closure.nodes = python_sort(closure.nodes, (item) => relative_to_root(item, root));
  closure.edges = python_sort(
    [...new Map(closure.edges.map((edge) => [JSON.stringify(edge), edge])).values()],
    (edge) => [relative_to_root(edge[0], root), relative_to_root(edge[1], root)]
  );
  closure.unresolved = python_sort(closure.unresolved, (item) => [
    relative_to_root(item[0], root),
    item[1],
  ]);
  closure.parse_failures = python_sort(closure.parse_failures, (item) => [
    relative_to_root(item[0], root),
    item[1],
  ]);
  return closure;
}

export function closure_report_path(value, root) {
  const candidate = resolve_path(path.isAbsolute(value) ? value : path.resolve(root, value));
  return within_root(candidate, root) ? candidate : null;
}

export function load_raw_ast_closure_report(file, source, root) {
  const closure = new Closure({ root, source });
  closure.collector = 'madaros-raw-ast-v1';
  closure.report_sha256 = sha256_file(file);
  const diagnostics = [];
  let lines;
  try {
    lines = universal_newlines(read_utf8(file)).split(/[\n\v\f\x1c-\x1e\x85\u2028\u2029]/u);
  } catch (error) {
    closure.parse_failures.push([source, 'cannot read raw AST closure report: ' + error.message]);
    return [closure, diagnostics];
  }
  const header_index = lines.indexOf('SOUNIO_BOUNDARY_CLOSURE_V1');
  if (header_index < 0) {
    closure.parse_failures.push([source, 'invalid raw AST closure report header']);
    return [closure, diagnostics];
  }
  lines = lines.slice(header_index);
  let status = '';
  const declared_nodes = [],
    declared_edges = [];
  // The real V1 emitter writes all four controls exactly once. Incomplete or
  // contradictory metadata must not inherit a false/default authority state.
  const required_controls = new Set(['status', 'capacity', 'saturated', 'parse_failed']);
  const seen_controls = new Set();
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i],
      fields = line.split('\t'),
      kind = fields[0] || '',
      line_number = i + 1;
    if (required_controls.has(kind)) {
      if (seen_controls.has(kind)) {
        closure.parse_failures.push([
          source,
          'duplicate raw AST control record: ' + kind + ' on report line ' + line_number,
        ]);
        continue;
      }
      seen_controls.add(kind);
    }
    if (kind === 'status' && fields.length === 2) status = fields[1];
    else if (kind === 'capacity' && fields.length === 2) {
      try {
        closure.capacity = bounded_capacity(fields[1]);
      } catch {
        closure.parse_failures.push([source, 'invalid capacity on report line ' + line_number]);
      }
    } else if (kind === 'saturated' && fields.length === 2) {
      if (!['true', 'false'].includes(fields[1]))
        closure.parse_failures.push([
          source,
          'invalid saturated boolean on report line ' + line_number,
        ]);
      else closure.saturated = fields[1] === 'true';
    } else if (kind === 'parse_failed' && fields.length === 2) {
      if (!['true', 'false'].includes(fields[1]))
        closure.parse_failures.push([
          source,
          'invalid parse_failed boolean on report line ' + line_number,
        ]);
      else if (fields[1] === 'true')
        closure.parse_failures.push([source, 'raw AST parser reported failure']);
    } else if (kind === 'node' && fields.length === 2) {
      const node = closure_report_path(fields[1], root);
      if (node === null || !is_file(node))
        closure.parse_failures.push([source, 'invalid raw AST node on report line ' + line_number]);
      else declared_nodes.push(node);
    } else if (kind === 'edge' && fields.length === 3) {
      const caller = closure_report_path(fields[1], root),
        dependency = closure_report_path(fields[2], root);
      if (caller === null || dependency === null)
        closure.parse_failures.push([
          source,
          'raw AST edge escapes root on report line ' + line_number,
        ]);
      else declared_edges.push([caller, dependency]);
    } else if (kind === 'unresolved' && fields.length === 3) {
      const caller = closure_report_path(fields[1], root);
      if (caller === null)
        closure.parse_failures.push([
          source,
          'raw AST unresolved caller escapes root on report line ' + line_number,
        ]);
      else closure.unresolved.push([caller, fields[2]]);
    } else if (line)
      closure.parse_failures.push([source, 'invalid raw AST record on report line ' + line_number]);
  }
  for (const kind of required_controls) {
    if (!seen_controls.has(kind))
      closure.parse_failures.push([source, 'missing raw AST control record: ' + kind]);
  }
  const node_set = new Set(declared_nodes);
  if (!node_set.has(resolve_path(source)))
    closure.parse_failures.push([source, 'raw AST closure does not contain the root source']);
  for (const [caller, dependency] of declared_edges) {
    if (!node_set.has(caller) || !node_set.has(dependency))
      closure.parse_failures.push([source, 'raw AST edge references an undeclared node']);
  }
  if (!['complete', 'incomplete'].includes(status))
    closure.parse_failures.push([source, 'raw AST closure status is missing or invalid']);
  else if (status === 'incomplete' && !closure.parse_failures.length)
    closure.parse_failures.push([source, 'raw AST closure is incomplete']);
  if (status === 'complete' && (closure.saturated || closure.unresolved.length)) {
    closure.parse_failures.push([
      source,
      'raw AST closure claims complete with unresolved or saturated state',
    ]);
  }
  closure.nodes = python_sort([...node_set], (item) => relative_to_root(item, root));
  closure.edges = python_sort(
    [...new Map(declared_edges.map((edge) => [JSON.stringify(edge), edge])).values()],
    (edge) => [relative_to_root(edge[0], root), relative_to_root(edge[1], root)]
  );
  closure.unresolved = python_sort(closure.unresolved, (item) => [
    relative_to_root(item[0], root),
    item[1],
  ]);
  return [closure, diagnostics];
}
