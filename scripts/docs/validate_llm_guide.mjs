import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const manifestFile = 'docs/llm-guide/example-manifest.json';
const guides = ['docs/guide/LLM_PROGRAMMING_GUIDE.md', 'docs/guide/LLM_PROGRAMMING_GUIDE_PT.md', 'docs/guide/LLM_PROGRAMMING_GUIDE_EL.md'];
const cookbook = 'docs/llm-guide/cookbook.md';
const atlas = 'docs/llm-guide/examples.md';
const snippetDocuments = [...guides, cookbook, atlas];
const quarantineContextLines = 12;
const verificationStates = new Set(['source-only', 'check-passed', 'run-passed', 'expected-refusal', 'unexpected-failure']);
const frontdoors = ['llms.txt', '.github/copilot-instructions.md', '.github/SUPPORT.md', 'docs/README.md'];
const spec = 'docs/spec/LANGUAGE_SPECIFICATION.md';
const topics = ['orientation', 'workflow', 'syntax', 'types', 'functions', 'control-flow', 'ownership', 'effects', 'scientific', 'modules', 'targets', 'diagnostics', 'recipes', 'prompts', 'delivery'];
const identifier = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const normalizeSource = text => text.replaceAll('\r\n', '\n').replace(/\n$/, '');
const nonempty = value => typeof value === 'string' && value.trim().length > 0;
const labelKey = value => value.trim().replace(/\s+/g, ' ').toLowerCase();

function gitRead(argumentsList, options = {}) {
  return execFileSync('git', argumentsList, {
    cwd: root,
    env: { ...process.env, GIT_OPTIONAL_LOCKS: '0', GIT_NO_LAZY_FETCH: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
    maxBuffer: 64 * 1024 * 1024,
    ...options,
  });
}

function parseGitBatch(bytes, objectIds) {
  const objects = new Map();
  let cursor = 0;
  for (const objectId of objectIds) {
    const headerEnd = bytes.indexOf(10, cursor);
    if (headerEnd < 0) throw new Error('truncated Git batch header');
    const header = /^([a-f0-9]{40}) blob (\d+)$/.exec(bytes.subarray(cursor, headerEnd).toString('utf8'));
    if (!header || header[1] !== objectId) throw new Error('invalid Git batch object identity or type');
    const size = Number(header[2]);
    const contentStart = headerEnd + 1;
    const contentEnd = contentStart + size;
    if (!Number.isSafeInteger(size) || contentEnd >= bytes.length || bytes[contentEnd] !== 10) throw new Error('truncated or invalid Git batch blob');
    objects.set(objectId, bytes.subarray(contentStart, contentEnd));
    cursor = contentEnd + 1;
  }
  if (cursor !== bytes.length) throw new Error('unexpected trailing Git batch data');
  return objects;
}

function safePath(value) {
  return nonempty(value) && !/[\\\0\r\n:]/.test(value) && !path.posix.isAbsolute(value)
    && value.split('/').every(segment => segment && segment !== '.' && segment !== '..' && segment !== '.git');
}

function withinRoot(base, target) {
  const relative = path.relative(base, target);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function containerLine(line) {
  return line.replace(/^(?: {0,3}>[ \t]?)+/, '');
}

function scanMarkdown(text, file) {
  const lines = text.replaceAll('\r\n', '\n').split('\n');
  const sections = [];
  const snippets = [];
  const quarantines = [];
  const errors = [];
  const prose = [];
  let fence = null;
  let pending = null;
  let inComment = false;
  let contextStart = 0;
  for (const [index, rawLine] of lines.entries()) {
    const line = containerLine(rawLine);
    if (fence) {
      const closing = /^\s*(`{3,}|~{3,})[ \t]*$/.exec(line);
      if (closing && closing[1][0] === fence.delimiter[0] && closing[1].length >= fence.delimiter.length) {
        if (fence.sounio) snippets.push({ id: fence.id, line: fence.line, contextStart: fence.contextStart, source: fence.body.join('\n') + (fence.body.length ? '\n' : '') });
        fence = null;
        contextStart = index + 1;
      } else fence.body.push(line.slice(Math.min(fence.indent, /^ */.exec(line)[0].length)));
      prose.push('');
      continue;
    }
    const trimmed = line.trim();
    const marker = !inComment && /^<!-- llm:(section|example|quarantine) ([A-Za-z0-9][A-Za-z0-9._-]*) -->$/.exec(trimmed);
    const opening = !inComment && /^( *)(?:(?:[-+*]|\d+[.)])[ \t]+)?(`{3,}|~{3,})([^\n]*)$/.exec(line);
    if (pending && !(opening && /^(?:sio|sounio)(?:\s|$)/i.test(opening[3].trim()))) {
      errors.push(`${file}:${pending.line}: example marker must immediately precede a sio/sounio fence`);
      pending = null;
    }
    if (opening) {
      const sounio = /^(?:sio|sounio)(?:\s|$)/i.test(opening[3].trim());
      if (sounio && !pending) errors.push(`${file}:${index + 1}: missing immediately preceding example marker`);
      fence = { delimiter: opening[2], indent: opening[1].length, sounio, id: pending?.id, line: index + 1, contextStart, body: [] };
      pending = null;
      prose.push('');
      continue;
    }
    if (marker) {
      if (marker[1] === 'section') { sections.push(marker[2]); contextStart = index; }
      else if (marker[1] === 'quarantine') quarantines.push({ id: marker[2], line: index + 1 });
      else pending = { id: marker[2], line: index + 1 };
      prose.push('');
      continue;
    }
    if (!inComment && /<!--\s*llm:(?:section|example|quarantine)\b/.test(line)) errors.push(`${file}:${index + 1}: malformed llm marker`);
    let visible = '';
    let cursor = 0;
    while (cursor < line.length) {
      const boundary = line.indexOf(inComment ? '-->' : '<!--', cursor);
      if (boundary < 0) {
        if (!inComment) visible += line.slice(cursor);
        break;
      }
      if (!inComment) visible += line.slice(cursor, boundary);
      cursor = boundary + (inComment ? 3 : 4);
      inComment = !inComment;
    }
    prose.push(visible);
    if (/^ {0,3}(?:#{1,6}[ \t]+|(?:=+|-+)[ \t]*$)/.test(visible)) contextStart = index;
  }
  if (fence) errors.push(`${file}:${fence.line}: unterminated fence`);
  if (pending) errors.push(`${file}:${pending.line}: orphan example marker`);
  if (inComment) errors.push(`${file}: unterminated HTML comment`);
  return { sections, snippets, quarantines, prose: prose.join('\n'), errors };
}

function sectionIssues(sections, file) {
  const errors = [];
  const seen = new Set();
  for (const section of sections) {
    if (seen.has(section)) errors.push(`${file}: duplicate section ${section}`);
    if (!topics.includes(section)) errors.push(`${file}: unknown section ${section}`);
    seen.add(section);
  }
  for (const topic of topics) if (!seen.has(topic)) errors.push(`${file}: missing section ${topic}`);
  return errors;
}

function manifestIssues(manifest) {
  const errors = [];
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) return ['manifest must be an object'];
  if (manifest.schema !== 'sounio.llm-guide.examples.v1') errors.push('unknown manifest schema');
  if (typeof manifest.source_revision !== 'string' || !/^[a-f0-9]{40}$/.test(manifest.source_revision)) errors.push('invalid manifest source_revision');
  if (!Array.isArray(manifest.examples) || !manifest.examples.length) return [...errors, 'manifest examples must be a nonempty array'];
  const seen = new Set();
  for (const [index, example] of manifest.examples.entries()) {
    const prefix = `manifest example ${index + 1}`;
    if (!example || typeof example !== 'object' || Array.isArray(example)) {
      errors.push(`${prefix}: must be an object`);
      continue;
    }
    if (typeof example.id !== 'string' || !identifier.test(example.id)) errors.push(`${prefix}: invalid id`);
    if (seen.has(example.id)) errors.push(`${prefix}: duplicate id ${example.id}`);
    seen.add(example.id);
    if (example.verification !== undefined && (!example.verification || typeof example.verification !== 'object'
      || Array.isArray(example.verification) || !verificationStates.has(example.verification.state))) errors.push(`${prefix}: invalid verification.state`);
    if (!safePath(example.source) || !example.source.endsWith('.sio')) errors.push(`${prefix}: unsafe or non-sio source path`);
    if (typeof example.source_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(example.source_sha256)) errors.push(`${prefix}: invalid source_sha256`);
    for (const field of ['topic', 'scope']) if (!nonempty(example[field])) errors.push(`${prefix}: empty ${field}`);
    for (const field of ['stdout_contains', 'stderr_contains', 'diagnostic']) {
      if (example[field] !== null && !nonempty(example[field])) errors.push(`${prefix}: ${field} must be a nonempty string or null`);
    }
    const exit = example.expected_exit;
    if (exit !== 'nonzero' && !(Number.isSafeInteger(exit) && exit >= 0)) errors.push(`${prefix}: invalid expected_exit`);
    if (!['positive', 'negative'].includes(example.kind)) errors.push(`${prefix}: invalid kind`);
    if (!['check', 'run', 'check-fail'].includes(example.mode)) errors.push(`${prefix}: invalid mode`);
    if (example.kind === 'positive') {
      if (example.mode === 'check-fail' || (example.mode === 'check' && exit !== 0) || exit === 'nonzero') errors.push(`${prefix}: inconsistent positive mode/expected_exit`);
      if (example.diagnostic !== null) errors.push(`${prefix}: positive diagnostic must be null`);
    }
    if (example.kind === 'negative') {
      if (example.mode !== 'check-fail' || !(exit === 'nonzero' || (Number.isSafeInteger(exit) && exit > 0))) errors.push(`${prefix}: negative requires check-fail and a nonzero expected_exit`);
      if (!nonempty(example.diagnostic) || ![example.stdout_contains, example.stderr_contains].some(stream => nonempty(stream) && stream.includes(example.diagnostic))) {
        errors.push(`${prefix}: negative diagnostic must occur in one asserted stream`);
      }
    }
  }
  return errors;
}

function snippetIssues(parsed, file, examples, sources, requireComplete = false) {
  const errors = [];
  const seen = new Set();
  for (const snippet of parsed.snippets) {
    if (!snippet.id) continue;
    if (seen.has(snippet.id)) errors.push(`${file}:${snippet.line}: duplicate example ${snippet.id}`);
    seen.add(snippet.id);
    const example = examples.get(snippet.id);
    if (!example) errors.push(`${file}:${snippet.line}: unknown example ${snippet.id}`);
    else if (sources.has(example.source) && normalizeSource(snippet.source) !== normalizeSource(sources.get(example.source).toString('utf8'))) {
      errors.push(`${file}:${snippet.line}: example ${snippet.id} differs from complete source ${example.source}`);
    }
    if (example?.verification?.state === 'unexpected-failure') {
      const exampleMarkerIndex = snippet.line - 2;
      const firstLine = Math.max(snippet.contextStart, exampleMarkerIndex - quarantineContextLines);
      const matchingMarker = parsed.quarantines.some(marker => marker.id === snippet.id && marker.line > firstLine && marker.line < snippet.line - 1);
      if (!matchingMarker) errors.push(`${file}:${snippet.line}: unexpected-failure example ${snippet.id} requires quarantine marker in adjacent preceding context`);
      const context = stripHiddenElements(parsed.prose.split('\n').slice(firstLine, exampleMarkerIndex).join('\n'));
      const displayed = stripHtmlTags(withoutInlineCode(context)).replace(/[*_]/g, '').replace(/\s+/g, ' ');
      if (!/\bnot a working(?: or confirmed-refusal)? recipe\b/i.test(displayed)) errors.push(`${file}:${snippet.line}: unexpected-failure example ${snippet.id} requires a visible quarantine warning before code saying "not a working recipe" or "not a working or confirmed-refusal recipe"`);
    }
  }
  if (requireComplete) for (const exampleId of examples.keys()) if (!seen.has(exampleId)) errors.push(`${file}: missing atlas example ${exampleId}`);
  return errors;
}

function stripHtmlTags(text) {
  let current = text;
  let previous;
  do {
    previous = current;
    current = current.replace(/<[^>]*>/g, '');
  } while (current !== previous);
  return current;
}

function stripHiddenElements(text) {
  let current = text;
  let previous;
  do {
    previous = current;
    current = current.replace(/<([A-Za-z][\w:-]*)\b(?=[^>]*\b(?:hidden(?:\s|=|>)|aria-hidden\s*=\s*["']true["']|style\s*=\s*["'][^"']*(?:display\s*:\s*none|visibility\s*:\s*hidden)))[^>]*>[\s\S]*?<\/\1\s*>/gi, '');
  } while (current !== previous);
  return current;
}

function withoutInlineCode(text) {
  return text.replace(/(`+)([^`]|(?!\1)`)*?\1/g, match => ' '.repeat(match.length));
}

function unescapeMarkdown(text) {
  return text.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~])/g, '$1')
    .replace(/&(?:amp|lt|gt|quot|apos|#(\d+)|#x([a-f0-9]+));/gi, (entity, decimal, hex) => {
      if (decimal || hex) {
        const codepoint = Number.parseInt(decimal || hex, decimal ? 10 : 16);
        return codepoint > 0 && codepoint <= 0x10ffff ? String.fromCodePoint(codepoint) : '\uFFFD';
      }
      return { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'" }[entity.toLowerCase()];
    });
}

function destinationAt(text, start) {
  let cursor = start;
  while (/\s/.test(text[cursor] ?? '') && cursor < text.length) cursor += 1;
  if (text[cursor] === '<') {
    const end = text.indexOf('>', cursor + 1);
    return end < 0 ? null : { target: text.slice(cursor + 1, end), end: end + 1 };
  }
  const begin = cursor;
  let depth = 0;
  while (cursor < text.length) {
    const character = text[cursor];
    if (character === '\\' && cursor + 1 < text.length) { cursor += 2; continue; }
    if (/\s/.test(character) || (character === ')' && !depth)) break;
    if (character === '(') depth += 1;
    if (character === ')') depth -= 1;
    cursor += 1;
  }
  return depth ? null : { target: text.slice(begin, cursor), end: cursor };
}

function bracketEnd(text, start) {
  let depth = 1;
  for (let cursor = start + 1; cursor < text.length; cursor += 1) {
    if (text[cursor] === '\\') { cursor += 1; continue; }
    if (text[cursor] === '[') depth += 1;
    if (text[cursor] === ']' && --depth === 0) return cursor;
  }
  return -1;
}

function linkTargets(prose) {
  const errors = [];
  const targets = [];
  const references = new Map();
  let body = prose.replace(/^ {0,3}\[([^\]\n]+)\]:[ \t]*(?:\n[ \t]*)?([^\n]*)$/gm, (definition, label, remainder) => {
    const destination = destinationAt(remainder, 0);
    if (!destination || !destination.target) errors.push(`invalid link reference ${label}`);
    else {
      const key = labelKey(label);
      if (!references.has(key)) references.set(key, destination.target);
      targets.push(destination.target);
    }
    return '';
  });
  body = withoutInlineCode(body);
  for (let cursor = 0; cursor < body.length; cursor += 1) {
    if (body[cursor] === '\\') { cursor += 1; continue; }
    if (body[cursor] !== '[') continue;
    const closing = bracketEnd(body, cursor);
    if (closing < 0) continue;
    const label = body.slice(cursor + 1, closing);
    if (body[closing + 1] === '(') {
      const destination = destinationAt(body, closing + 2);
      if (!destination) errors.push(`invalid inline link destination for ${label}`);
      else {
        let end = destination.end;
        while (/\s/.test(body[end] ?? '') && end < body.length) end += 1;
        if (['"', "'"].includes(body[end]) || body[end] === '(') {
          const delimiter = body[end] === '(' ? ')' : body[end];
          end += 1;
          while (end < body.length && body[end] !== delimiter) {
            if (body[end] === '\\') end += 1;
            end += 1;
          }
          end += 1;
          while (/\s/.test(body[end] ?? '') && end < body.length) end += 1;
        }
        if (body[end] !== ')') errors.push(`unterminated inline link for ${label}`);
        else { targets.push(destination.target); cursor = end; continue; }
      }
    } else if (body[closing + 1] === '[') {
      const referenceEnd = bracketEnd(body, closing + 1);
      if (referenceEnd < 0) errors.push(`unterminated link reference ${label}`);
      else {
        const key = labelKey(body.slice(closing + 2, referenceEnd) || label);
        if (!references.has(key)) errors.push(`undefined link reference ${key}`);
        cursor = referenceEnd;
        continue;
      }
    } else {
      const target = references.get(labelKey(label));
      if (target !== undefined) targets.push(target);
    }
    cursor = closing;
  }
  for (const match of body.matchAll(/<(?:a|img)\b[^>]*\b(?:href|src)\s*=\s*["']([^"']+)["'][^>]*>/gi)) targets.push(match[1]);
  return { targets: targets.map(unescapeMarkdown), errors };
}

function headingIds(prose) {
  const ids = new Set();
  const counts = new Map();
  const lines = prose.split('\n');
  for (const [index, line] of lines.entries()) {
    const heading = /^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+[ \t]*)?$/.exec(line);
    const setext = index + 1 < lines.length && /^ {0,3}(?:=+|-+)[ \t]*$/.test(lines[index + 1]) && line.trim() && !/^\s*(?:[-+*>]|\d+[.)])\s/.test(line);
    if (!heading && !setext) continue;
    const title = unescapeMarkdown(heading ? heading[1] : line.trim()).replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1');
    const slug = stripHtmlTags(title.toLowerCase()).replace(/[^\p{L}\p{M}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
    let candidate = slug;
    let occurrence = counts.get(slug) ?? 0;
    while (ids.has(candidate)) candidate = `${slug}-${++occurrence}`;
    counts.set(slug, occurrence);
    ids.add(candidate);
  }
  for (const match of prose.matchAll(/<[A-Za-z][^>]*\s(?:id|name)\s*=\s*["']([^"']+)["']/g)) ids.add(unescapeMarkdown(match[1]));
  return ids;
}

function resolveLink(file, target) {
  if (/^(?:[A-Za-z][A-Za-z0-9+.-]*:|\/\/)/.test(target)) return { external: true };
  const boundary = target.indexOf('#');
  const rawPath = boundary < 0 ? target : target.slice(0, boundary);
  const fragment = boundary < 0 ? '' : target.slice(boundary + 1);
  try {
    const pathname = decodeURIComponent(rawPath.split('?')[0]);
    const anchor = decodeURIComponent(fragment);
    if (/[\\\0]/.test(pathname) || pathname.startsWith('/')) return { error: 'escaping link' };
    const resolved = pathname ? path.posix.normalize(path.posix.join(path.posix.dirname(file), pathname)).replace(/\/+$/, '') : file;
    if (resolved !== '.' && !safePath(resolved)) return { error: 'escaping link' };
    return { file: resolved, anchor };
  } catch {
    return { error: 'invalid link encoding' };
  }
}

async function linkIssues(file, parsed, repository) {
  const links = linkTargets(parsed.prose);
  const errors = links.errors.map(error => `${file}: ${error}`);
  const destinations = new Set();
  for (const target of new Set(links.targets)) {
    const resolved = resolveLink(file, target);
    if (resolved.external) continue;
    if (resolved.error) { errors.push(`${file}: ${resolved.error} ${target}`); continue; }
    destinations.add(resolved.file);
    try {
      const entry = await repository.entry(resolved.file);
      if (!entry) { errors.push(`${file}: broken or escaping link ${target}`); continue; }
      if (resolved.anchor && /\.(?:md|markdown)$/i.test(resolved.file) && entry.kind === 'file') {
        const targetProse = resolved.file === file ? parsed.prose : scanMarkdown((await repository.read(resolved.file)).toString('utf8'), resolved.file).prose;
        if (!headingIds(targetProse).has(resolved.anchor)) errors.push(`${file}: missing anchor ${target}`);
      }
    } catch (error) { errors.push(`${file}: unreadable link ${target}: ${error.message}`); }
  }
  return { errors, destinations, count: links.targets.length };
}

function staleClaimIssues(file, prose) {
  const errors = [];
  const patterns = [
    /^(?:no|sem)\s+(?:semicolons?|pontos?[- ]e[- ]vírgulas?)(?:[.!:]|\s*[—–-]|\s*$)/i,
    /^(?:never use|do not use|não use)\s+(?:semicolons?|pontos?[- ]e[- ]vírgulas?)[.!]?\s*$/i,
    /^(?:no|sem)\s+(?:dynamic features|generics|closure literals|closures|traits|genéricos|closures literais)(?:[.!:]|\s*[—–-]|\s*$)/i,
    /^closures?\s+(?:do not|don't|does not|doesn't|never)\s+work[.!]?\s*$/i,
    /^(?:functions must declare side effects or the compiler rejects them|funções devem declarar efeitos ou o compilador as rejeita)[.!:]?\s*$/i,
  ];
  for (const line of prose.split('\n')) {
    const claim = line.trim().replace(/^(?:[-*+]\s+|\d+[.)]\s+)/, '').replace(/[*_`]/g, '');
    if (patterns.some(pattern => pattern.test(claim))) errors.push(`${file}: stale blanket claim: ${claim}`);
  }
  return errors;
}

async function validate(manifest, documents, repository) {
  const errors = manifestIssues(manifest).map(error => `${manifestFile}: ${error}`);
  const rows = Array.isArray(manifest?.examples) ? manifest.examples.filter(example => example && typeof example === 'object' && !Array.isArray(example)) : [];
  const examples = new Map(rows.map(example => [example.id, example]));
  const sources = new Map();
  let baselineVerified = 0;
  const revisionValid = /^[a-f0-9]{40}$/.test(manifest?.source_revision ?? '');
  let baselineReady = revisionValid;
  if (revisionValid && repository.prepareBaseline) {
    try {
      await repository.prepareBaseline(manifest.source_revision, [...new Set(rows.map(example => example.source).filter(source => safePath(source) && source.endsWith('.sio')))]);
    } catch (error) {
      errors.push(`cannot load source_revision baseline: ${error.message}`);
      baselineReady = false;
    }
  }
  for (const example of rows) {
    if (!safePath(example.source) || !example.source.endsWith('.sio')) continue;
    try {
      const entry = await repository.entry(example.source);
      if (!entry || entry.kind !== 'file') { errors.push(`${example.id}: source is missing, escaping, or not a regular file: ${example.source}`); continue; }
      if (!sources.has(example.source)) sources.set(example.source, await repository.read(example.source));
      const bytes = sources.get(example.source);
      if (sha256(bytes) !== example.source_sha256) errors.push(`${example.id}: source SHA256 mismatch: ${example.source}`);
      if (baselineReady) {
        const baseline = await repository.baseline(manifest.source_revision, example.source);
        if (!baseline || !['100644', '100755'].includes(baseline.mode)) errors.push(`${example.id}: source is absent or not a regular blob at source_revision: ${example.source}`);
        else if (!bytes.equals(baseline.bytes)) errors.push(`${example.id}: source differs from source_revision: ${example.source}`);
        else baselineVerified += 1;
      }
    } catch (error) { errors.push(`${example.id}: cannot verify source ${example.source}: ${error.message}`); }
  }
  let snippets = 0;
  let links = 0;
  const sectionSets = [];
  for (const [file, text] of documents) {
    const parsed = scanMarkdown(text, file);
    if (snippetDocuments.includes(file)) errors.push(...parsed.errors);
    if (guides.includes(file)) {
      errors.push(...sectionIssues(parsed.sections, file));
      sectionSets.push([...new Set(parsed.sections)].sort().join(','));
    }
    if (snippetDocuments.includes(file)) {
      errors.push(...snippetIssues(parsed, file, examples, sources, file === atlas));
      snippets += parsed.snippets.length;
    }
    if (file === 'llms.txt') {
      errors.push(...parsed.errors);
      if (parsed.snippets.length) errors.push(`${file}: Sounio fences are prohibited; link to the guide or example atlas`);
    }
    const checkedLinks = await linkIssues(file, parsed, repository);
    errors.push(...checkedLinks.errors);
    links += checkedLinks.count;
    if (guides.includes(file) && !checkedLinks.destinations.has(spec)) errors.push(`${file}: missing canonical language specification link`);
    if ([...guides, cookbook, 'llms.txt', '.github/copilot-instructions.md'].includes(file)) errors.push(...staleClaimIssues(file, parsed.prose));
  }
  for (const file of [...guides, cookbook, ...frontdoors]) if (!documents.has(file)) errors.push(`missing required document ${file}`);
  if (sectionSets.some(sectionSet => sectionSet !== sectionSets[0])) errors.push('multilingual section topic sets differ');
  return {
    schema: 'sounio.llm-guide.validation.v1',
    source_revision: manifest?.source_revision ?? null,
    scope: 'structural-only',
    execution: { scope: 'structural-only', compiler_invoked: false, receipts_verified: false },
    documents: documents.size,
    guide_topics: topics.length,
    examples: rows.length,
    source_files: sources.size,
    baseline_verified_examples: baselineVerified,
    snippets,
    links,
    errors: [...new Set(errors)],
  };
}

function diskRepository() {
  const reads = new Map();
  const entries = new Map();
  const baselines = new Map();
  const revisions = new Set();
  let physicalRoot;
  return {
    async entry(relative) {
      if (relative !== '.' && !safePath(relative)) return null;
      if (!entries.has(relative)) entries.set(relative, (async () => {
        try {
          physicalRoot ??= await realpath(root);
          const absolute = path.join(root, relative);
          if (!withinRoot(physicalRoot, await realpath(absolute))) return null;
          const information = await stat(absolute);
          return { kind: information.isFile() ? 'file' : information.isDirectory() ? 'directory' : 'other' };
        } catch (error) {
          if (['ENOENT', 'ENOTDIR', 'ELOOP'].includes(error.code)) return null;
          throw error;
        }
      })());
      return entries.get(relative);
    },
    async read(relative) {
      if (!safePath(relative) || (await this.entry(relative))?.kind !== 'file') throw new Error('not a repository file');
      if (!reads.has(relative)) reads.set(relative, readFile(path.join(root, relative)));
      return reads.get(relative);
    },
    async prepareBaseline(revision, relatives) {
      const requested = new Set(relatives.filter(relative => !baselines.has(`${revision}:${relative}`)));
      if (!requested.size) return;
      if (!revisions.has(revision)) {
        const type = gitRead(['cat-file', '-t', revision], { encoding: 'utf8' }).trim();
        if (type !== 'commit') throw new Error('source_revision must name a Git commit');
        revisions.add(revision);
      }
      const listing = gitRead(['--literal-pathspecs', 'ls-tree', '-z', revision, '--', ...requested], { encoding: 'utf8' });
      const records = [];
      for (const record of listing.split('\0').filter(Boolean)) {
        const entry = /^(\d+) blob ([a-f0-9]{40})\t([^\0]+)$/.exec(record);
        if (entry && requested.has(entry[3])) records.push({ mode: entry[1], objectId: entry[2], source: entry[3] });
      }
      const objectIds = [...new Set(records.map(record => record.objectId))];
      const bytes = objectIds.length ? gitRead(['cat-file', '--batch'], { input: `${objectIds.join('\n')}\n` }) : Buffer.alloc(0);
      const objects = parseGitBatch(bytes, objectIds);
      for (const relative of requested) baselines.set(`${revision}:${relative}`, null);
      for (const record of records) baselines.set(`${revision}:${record.source}`, { mode: record.mode, bytes: objects.get(record.objectId) });
    },
    async baseline(revision, relative) {
      const key = `${revision}:${relative}`;
      if (!baselines.has(key)) await this.prepareBaseline(revision, [relative]);
      return baselines.get(key);
    },
  };
}

function changedSupportFiles(revision) {
  const options = { encoding: 'utf8' };
  const tracked = gitRead(['diff', '--name-only', '-z', revision, '--', 'docs/llm-guide'], options);
  const untracked = gitRead(['ls-files', '--others', '--exclude-standard', '-z', '--', 'docs/llm-guide'], options);
  return [...new Set((tracked + untracked).split('\0').filter(file => file.endsWith('.md')))];
}

function fixture() {
  const source = 'fn main() -> i32 {\n    return 0;\n}\n';
  const example = { id: 'hello', source: 'examples/hello.sio', source_sha256: sha256(source), topic: 'orientation', kind: 'positive', mode: 'check', expected_exit: 0, stdout_contains: null, stderr_contains: null, diagnostic: null, scope: 'Structural fixture only.' };
  const manifest = { schema: 'sounio.llm-guide.examples.v1', source_revision: 'a'.repeat(40), examples: [example] };
  const revision = manifest.source_revision;
  const block = `<!-- llm:example hello -->\n\`\`\`sio\n${source}\`\`\`\n`;
  const sections = topics.map(topic => `<!-- llm:section ${topic} -->\n## ${topic}\n`).join('\n');
  const documents = new Map(guides.map(file => [file, `${sections}\n[Specification](../spec/LANGUAGE_SPECIFICATION.md)\n${block}`]));
  documents.set(cookbook, block);
  documents.set(atlas, block);
  for (const file of frontdoors) documents.set(file, '# Context\n');
  const files = new Map([...documents, [example.source, source], [spec, '# Specification\n\n## Café `type`\n## Repeat\n## Repeat\n## Repeat-1\n\nSetext heading\n-----\n\n<a id="custom"></a>\n']].map(([file, text]) => [file, Buffer.from(text)]));
  const baselines = new Map([[example.source, { mode: '100644', bytes: Buffer.from(source) }]]);
  const directories = new Set(['.', 'examples', 'examples/medlang']);
  const repository = {
    async entry(file) { return files.has(file) ? { kind: 'file' } : directories.has(file) ? { kind: 'directory' } : null; },
    async read(file) { if (!files.has(file)) throw new Error('missing fixture'); return files.get(file); },
    async baseline(requestedRevision, file) { return requestedRevision === revision ? baselines.get(file) : null; },
  };
  return { manifest, documents, files, baselines, repository, source, block };
}

async function selfTest() {
  const cases = [];
  const check = async (name, mutate, expected = null) => {
    const state = fixture();
    await mutate(state);
    const result = await validate(state.manifest, state.documents, state.repository);
    if (result.scope !== 'structural-only' || result.execution.compiler_invoked || result.execution.receipts_verified) throw new Error('Self-test changed structural execution scope');
    const passed = expected === null ? result.errors.length === 0 : result.errors.some(error => error.includes(expected));
    cases.push({ name, passed });
    if (!passed) throw new Error(`Self-test failed: ${name}: ${JSON.stringify(result.errors)}`);
  };
  const changeGuide = (state, transform, file = guides[0]) => state.documents.set(file, transform(state.documents.get(file)));
  const quarantineWarning = '> **Quarantined — unexpected failure. Do not use as working code; not a working recipe.**';
  await check('valid complete bilingual fixture', () => {});
  await check('unexpected failure cannot appear without quarantine', state => {
    state.manifest.examples[0].verification = { state: 'unexpected-failure' };
  }, 'requires quarantine marker');
  const markQuarantined = state => {
    state.manifest.examples[0].verification = { state: 'unexpected-failure' };
    for (const file of snippetDocuments) changeGuide(state, text => text.replace('<!-- llm:example hello -->', `<!-- llm:quarantine hello -->\n${quarantineWarning}\n<!-- llm:example hello -->`), file);
  };
  await check('actual adjacent quarantine marker and warning accepted', markQuarantined);
  await check('Portuguese quarantine warning accepted', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '> **Quarentena — falha inesperada. Não use como código funcional; not a working recipe.**'), guides[1]);
  });
  for (const file of snippetDocuments) {
    await check(`${file} requires its own quarantine marker`, state => {
      markQuarantined(state);
      changeGuide(state, text => text.replace('<!-- llm:quarantine hello -->\n', ''), file);
    }, `${file}:`);
    await check(`${file} requires its own visible warning`, state => {
      markQuarantined(state);
      changeGuide(state, text => text.replace(`${quarantineWarning}\n`, ''), file);
    }, 'requires a visible quarantine warning');
  }
  await check('wrong quarantine ID cannot cover snippet', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('llm:quarantine hello', 'llm:quarantine other'));
  }, 'requires quarantine marker');
  await check('another real manifest record cannot quarantine this snippet', state => {
    markQuarantined(state);
    state.manifest.examples.push({ ...state.manifest.examples[0], id: 'hello-alias', verification: { state: 'check-passed' } });
    state.documents.set(atlas, state.documents.get(atlas) + state.block.replace('llm:example hello', 'llm:example hello-alias'));
    changeGuide(state, text => text.replace('llm:quarantine hello', 'llm:quarantine hello-alias'));
  }, 'unexpected-failure example hello requires quarantine marker');
  await check('quarantine marker in a table cannot cover a source snippet', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('<!-- llm:quarantine hello -->', '| ID | Quarantine marker |\n|---|---|\n| hello | <!-- llm:quarantine hello --> |'));
  }, 'unexpected-failure example hello requires quarantine marker');
  await check('malformed quarantine marker rejected', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('llm:quarantine hello -->', 'llm:quarantine hello extra -->'));
  }, 'malformed llm marker');
  await check('commented quarantine marker ignored', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('<!-- llm:quarantine hello -->', '<!--\n<!-- llm:quarantine hello -->\n-->'));
  }, 'requires quarantine marker');
  await check('hidden warning does not count', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '<!-- Quarantined — unexpected failure; not a working recipe. -->'));
  }, 'requires a visible quarantine warning');
  await check('HTML hidden warning does not count', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '<span hidden>Quarantined — unexpected failure; not a working recipe.</span>'));
  }, 'requires a visible quarantine warning');
  await check('CSS hidden warning does not count', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '<div style="display: none">Quarantined — unexpected failure; not a working recipe.</div>'));
  }, 'requires a visible quarantine warning');
  await check('warning entirely in code span does not count', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '`Quarantined — unexpected failure; not a working recipe.`'));
  }, 'requires a visible quarantine warning');
  await check('warning may name state using inline code', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '**Warning:** `unexpected-failure`. Not a working recipe.'));
  });
  await check('manifest link alone is not a visible warning', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, 'See the manifest for verification.'));
  }, 'requires a visible quarantine warning');
  await check('quarantine resolved claim cannot serve as a warning', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, 'Quarantine resolved; working code below.'));
  }, 'requires a visible quarantine warning');
  await check('quarantine label alone is not an approved nonuse warning', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '**Quarantined — unexpected failure.**'));
  }, 'requires a visible quarantine warning');
  await check('confirmed-refusal nonuse phrase may wrap across lines', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(quarantineWarning, '> **Not a working or confirmed-refusal\n> recipe on this profile.**'));
  });
  await check('warning after code is too late', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace(`${quarantineWarning}\n`, '') + `\n${quarantineWarning}\n`);
  }, 'requires a visible quarantine warning');
  await check('distant quarantine does not cover snippet', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('<!-- llm:example hello -->', '\n'.repeat(13) + '<!-- llm:example hello -->'));
  }, 'requires quarantine marker');
  await check('another section separates quarantine context', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('<!-- llm:example hello -->', '### Another example\n<!-- llm:example hello -->'));
  }, 'requires quarantine marker');
  await check('another fence separates quarantine context', state => {
    markQuarantined(state);
    changeGuide(state, text => text.replace('<!-- llm:example hello -->', '```text\nA command\n```\n<!-- llm:example hello -->'));
  }, 'requires quarantine marker');
  await check('support snippets remain exempt from quarantine', state => {
    markQuarantined(state);
    state.documents.set('docs/llm-guide/support.md', state.block);
  });
  await check('passed state needs no quarantine and proves nothing external', state => {
    state.manifest.examples[0].verification = { state: 'check-passed', external_proof: 'not validated by this structural gate' };
  });
  await check('misspelled failure state cannot bypass quarantine', state => {
    state.manifest.examples[0].verification = { state: 'unexpected-failur' };
  }, 'invalid verification.state');
  await check('unknown verification state rejected', state => state.manifest.examples[0].verification = { state: 'passed' }, 'invalid verification.state');
  await check('null verification rejected', state => state.manifest.examples[0].verification = null, 'invalid verification.state');
  await check('verification without state rejected', state => state.manifest.examples[0].verification = {}, 'invalid verification.state');
  for (const stateName of ['source-only', 'check-passed', 'run-passed', 'expected-refusal']) {
    await check(`known ${stateName} state accepted structurally`, state => state.manifest.examples[0].verification = { state: stateName });
  }
  await check('baseline catalog prepared once with unique source paths', state => {
    state.repository.prepareBaseline = async (revision, sources) => {
      if (revision !== state.manifest.source_revision || sources.length !== 1 || sources[0] !== 'examples/hello.sio') throw new Error('incorrect preload request');
    };
  });
  await check('baseline catalog failure is reported once', state => {
    state.repository.prepareBaseline = async () => { throw new Error('offline baseline unavailable'); };
  }, 'cannot load source_revision baseline');
  await check('missing section', state => changeGuide(state, text => text.replace('<!-- llm:section types -->', '')), 'missing section types');
  await check('duplicate section', state => changeGuide(state, text => `${text}\n<!-- llm:section types -->`), 'duplicate section types');
  await check('unknown section', state => changeGuide(state, text => text.replace('section types', 'section invented')), 'unknown section invented');
  await check('translation topic mismatch', state => changeGuide(state, text => text.replace('section types', 'section invented'), guides[1]), 'multilingual section topic sets differ');
  await check('Greek topic mismatch', state => changeGuide(state, text => text.replace('section types', 'section invented'), guides[2]), 'multilingual section topic sets differ');
  await check('Greek guide is mandatory', state => state.documents.delete(guides[2]), 'missing required document');
  await check('Greek source drift refused', state => changeGuide(state, text => text.replace('return 0;', 'return 1;'), guides[2]), 'differs from complete source');
  await check('commented section ignored', state => changeGuide(state, text => text.replace('<!-- llm:section types -->', '<!--\n<!-- llm:section types -->\n-->')), 'missing section types');
  await check('fenced section ignored', state => changeGuide(state, text => text.replace('<!-- llm:section types -->', '```text\n<!-- llm:section types -->\n```')), 'missing section types');
  await check('missing marker', state => changeGuide(state, text => text.replace('<!-- llm:example hello -->\n', '')), 'missing immediately preceding');
  await check('blank line breaks adjacency', state => changeGuide(state, text => text.replace('example hello -->\n', 'example hello -->\n\n')), 'must immediately precede');
  await check('unknown snippet id', state => changeGuide(state, text => text.replace('example hello', 'example invented')), 'unknown example invented');
  await check('duplicate id within document', state => changeGuide(state, text => text + state.block), 'duplicate example hello');
  await check('same id across translations allowed', () => {});
  await check('malformed marker', state => changeGuide(state, text => text.replace('example hello -->', 'example hello extra -->')), 'malformed llm marker');
  await check('orphan marker', state => changeGuide(state, text => text + '<!-- llm:example orphan -->'), 'orphan example marker');
  await check('marker on wrong fence language', state => changeGuide(state, text => text.replace('```sio', '```text')), 'must immediately precede');
  await check('sounio tilde fence', state => changeGuide(state, text => text.replace('```sio', '~~~~sounio').replace(/```\n$/, '~~~~~\n')));
  await check('indented fence source normalization', state => changeGuide(state, text => text.replace(state.block, state.block.split('\n').map(line => line ? `  ${line}` : line).join('\n'))));
  await check('blockquoted fence source normalization', state => changeGuide(state, text => text.replace(state.block, state.block.split('\n').map(line => `> ${line}`).join('\n'))));
  await check('short closing fence refused', state => changeGuide(state, text => text.replace('```sio', '````sio')), 'unterminated fence');
  await check('source truncation', state => changeGuide(state, text => text.replace('    return 0;\n', '')), 'differs from complete source');
  await check('source indentation matters', state => changeGuide(state, text => text.replace('    return', '  return')), 'differs from complete source');
  await check('extra final blank line matters', state => changeGuide(state, text => text.replace('}\n```', '}\n\n```')), 'differs from complete source');
  await check('CRLF document normalization', state => changeGuide(state, text => text.replaceAll('\n', '\r\n')));
  await check('source final newline optional', state => {
    state.files.set('examples/hello.sio', Buffer.from(state.source.slice(0, -1)));
    state.baselines.get('examples/hello.sio').bytes = state.files.get('examples/hello.sio');
    state.manifest.examples[0].source_sha256 = sha256(state.files.get('examples/hello.sio'));
  });
  await check('duplicate manifest id', state => state.manifest.examples.push({ ...state.manifest.examples[0] }), 'duplicate id hello');
  await check('repeated source under unique alias allowed', state => {
    state.manifest.examples.push({ ...state.manifest.examples[0], id: 'cook-hello' });
    state.documents.set(atlas, state.block + state.block.replace('example hello', 'example cook-hello'));
  });
  await check('missing atlas example', state => state.documents.set(atlas, '# Empty atlas\n'), 'missing atlas example hello');
  await check('duplicate atlas example', state => state.documents.set(atlas, state.block.repeat(2)), 'duplicate example hello');
  await check('atlas is optional until it exists', state => state.documents.delete(atlas));
  await check('changed source with stale hash', state => state.files.set('examples/hello.sio', Buffer.from('changed\n')), 'source SHA256 mismatch');
  await check('updated hash cannot hide baseline change', state => {
    state.files.set('examples/hello.sio', Buffer.from('changed\n'));
    state.manifest.examples[0].source_sha256 = sha256('changed\n');
  }, 'source differs from source_revision');
  await check('source path substitution', state => state.manifest.examples[0].source = 'examples/missing.sio', 'source is missing');
  await check('unsafe source path', state => state.manifest.examples[0].source = '../outside.sio', 'unsafe or non-sio');
  await check('absolute source path', state => state.manifest.examples[0].source = '/outside.sio', 'unsafe or non-sio');
  await check('noncanonical source path', state => state.manifest.examples[0].source = 'examples/../hello.sio', 'unsafe or non-sio');
  await check('missing baseline blob', state => state.baselines.clear(), 'source is absent');
  await check('baseline symlink rejected', state => state.baselines.get('examples/hello.sio').mode = '120000', 'not a regular blob');
  await check('source directory rejected', state => state.repository.entry = async file => ({ kind: file.endsWith('.sio') ? 'directory' : 'file' }), 'not a regular file');
  await check('escaping realpath rejected', state => state.repository.entry = async file => file.endsWith('.sio') ? null : { kind: 'file' }, 'source is missing, escaping');
  await check('bad schema', state => state.manifest.schema = 'other', 'unknown manifest schema');
  await check('bad revision', state => state.manifest.source_revision = 'HEAD', 'invalid manifest source_revision');
  await check('unknown baseline revision', state => state.manifest.source_revision = 'b'.repeat(40), 'source is absent');
  await check('bad hash', state => state.manifest.examples[0].source_sha256 = 'xyz', 'invalid source_sha256');
  await check('well-formed incorrect manifest hash', state => state.manifest.examples[0].source_sha256 = '0'.repeat(64), 'source SHA256 mismatch');
  await check('empty examples', state => state.manifest.examples = [], 'nonempty array');
  await check('null manifest', state => state.manifest = null, 'manifest must be an object');
  await check('null row', state => state.manifest.examples.push(null), 'must be an object');
  await check('blank scope', state => state.manifest.examples[0].scope = ' ', 'empty scope');
  await check('blank topic', state => state.manifest.examples[0].topic = '', 'empty topic');
  await check('missing asserted stream field', state => delete state.manifest.examples[0].stderr_contains, 'stderr_contains must');
  await check('invalid kind', state => state.manifest.examples[0].kind = 'unknown', 'invalid kind');
  await check('invalid mode', state => state.manifest.examples[0].mode = 'compile', 'invalid mode');
  await check('fractional exit', state => state.manifest.examples[0].expected_exit = 0.5, 'invalid expected_exit');
  await check('check requires exit zero', state => state.manifest.examples[0].expected_exit = 42, 'inconsistent positive');
  await check('run may expect explicit nonzero program return', state => Object.assign(state.manifest.examples[0], { mode: 'run', expected_exit: 42 }));
  await check('check-only library needs no main', state => {
    const library = 'fn identity(value: i64) -> i64 { value }\n';
    state.files.set('examples/hello.sio', Buffer.from(library));
    state.baselines.get('examples/hello.sio').bytes = Buffer.from(library);
    state.manifest.examples[0].source_sha256 = sha256(library);
    for (const file of [...guides, cookbook, atlas]) changeGuide(state, text => text.replace(state.source, library), file);
  });
  const makeNegative = state => Object.assign(state.manifest.examples[0], { kind: 'negative', mode: 'check-fail', expected_exit: 'nonzero', diagnostic: 'E001', stderr_contains: 'error[E001]: expected expression' });
  await check('negative diagnostic bound to stderr', makeNegative);
  await check('negative diagnostic bound to stdout', state => { makeNegative(state); Object.assign(state.manifest.examples[0], { stderr_contains: null, stdout_contains: 'error[E001]', expected_exit: 1 }); });
  await check('negative arbitrary exit insufficient', state => { makeNegative(state); state.manifest.examples[0].stderr_contains = null; }, 'negative diagnostic must occur');
  await check('negative diagnostic mismatch', state => { makeNegative(state); state.manifest.examples[0].diagnostic = 'E002'; }, 'negative diagnostic must occur');
  await check('negative split streams insufficient', state => { makeNegative(state); Object.assign(state.manifest.examples[0], { stdout_contains: 'E0', stderr_contains: '01' }); }, 'negative diagnostic must occur');
  await check('negative exit zero refused', state => { makeNegative(state); state.manifest.examples[0].expected_exit = 0; }, 'negative requires check-fail');
  await check('negative run refused', state => { makeNegative(state); state.manifest.examples[0].mode = 'run'; }, 'negative requires check-fail');
  await check('positive check-fail refused', state => state.manifest.examples[0].mode = 'check-fail', 'inconsistent positive');
  const append = (state, text, file = guides[0]) => changeGuide(state, original => `${original}\n${text}\n`, file);
  await check('missing canonical spec link', state => changeGuide(state, text => text.replace('[Specification](../spec/LANGUAGE_SPECIFICATION.md)', '')), 'missing canonical');
  await check('inline local anchor', state => append(state, '[Types](../spec/LANGUAGE_SPECIFICATION.md#café-type)'));
  await check('percent-encoded anchor', state => append(state, '[Types](../spec/LANGUAGE_SPECIFICATION.md#caf%C3%A9-type)'));
  await check('duplicate heading slug', state => append(state, '[Repeat](../spec/LANGUAGE_SPECIFICATION.md#repeat-1-1)'));
  await check('setext anchor', state => append(state, '[Heading](../spec/LANGUAGE_SPECIFICATION.md#setext-heading)'));
  await check('explicit HTML anchor', state => append(state, '[Heading](../spec/LANGUAGE_SPECIFICATION.md#custom)'));
  await check('same document anchor', state => append(state, '[Types](#types)'));
  await check('missing anchor', state => append(state, '[Missing](../spec/LANGUAGE_SPECIFICATION.md#absent)'), 'missing anchor');
  await check('broken local link', state => append(state, '[Missing](missing.md)'), 'broken or escaping link');
  await check('escaping link', state => append(state, '[Outside](../../../outside.md)'), 'escaping link');
  await check('encoded escaping link', state => append(state, '[Outside](%2e%2e/%2e%2e/%2e%2e/outside.md)'), 'escaping link');
  await check('invalid percent encoding', state => append(state, '[Bad](bad%ZZ.md)'), 'invalid link encoding');
  await check('reference links normalized', state => append(state, '[Types][MY   SPEC]\n\n[my spec]: <../spec/LANGUAGE_SPECIFICATION.md#café-type> "Language spec"'));
  await check('multiline reference definition', state => append(state, '[spec][target]\n\n[target]:\n  ../spec/LANGUAGE_SPECIFICATION.md#café-type'));
  await check('backtick code inside link label', state => append(state, '[`Specification`](../spec/LANGUAGE_SPECIFICATION.md)'));
  await check('collapsed reference', state => append(state, '[spec][]\n\n[spec]: ../spec/LANGUAGE_SPECIFICATION.md'));
  await check('shortcut reference', state => append(state, '[spec]\n\n[spec]: ../spec/LANGUAGE_SPECIFICATION.md'));
  await check('undefined reference', state => append(state, '[spec][undefined]'), 'undefined link reference');
  await check('unused broken reference still checked', state => append(state, '[unused]: missing.md'), 'broken or escaping link');
  await check('reference missing anchor', state => append(state, '[spec][target]\n[target]: ../spec/LANGUAGE_SPECIFICATION.md#absent'), 'missing anchor');
  await check('nested label and title', state => append(state, '[A [nested] label](../spec/LANGUAGE_SPECIFICATION.md "Title")'));
  await check('balanced parentheses in path', state => { state.files.set('docs/guide/file(one).md', Buffer.from('# Fine\n')); append(state, '[Fine](file(one).md#fine)'); });
  await check('escaped parentheses in path', state => { state.files.set('docs/guide/file(one).md', Buffer.from('# Fine\n')); append(state, '[Fine](file\\(one\\).md#fine)'); });
  await check('angle destination with spaces', state => { state.files.set('docs/guide/my file.md', Buffer.from('# Fine\n')); append(state, '[Fine](<my file.md#fine>)'); });
  await check('HTML local link', state => append(state, '<a href="missing.md">Missing</a>'), 'broken or escaping link');
  await check('image local link', state => append(state, '![Missing](missing.png)'), 'broken or escaping link');
  await check('inline code link ignored', state => append(state, '`[Missing](missing.md)`'));
  await check('commented link ignored', state => append(state, '<!-- [Missing](missing.md) -->'));
  await check('code fence link ignored', state => append(state, '~~~text\n[Missing](missing.md)\n~~~'));
  await check('external links ignored', state => append(state, '[Web](https://example.invalid/a_(b)) [Mail](mailto:hello@example.invalid) [Web](//example.invalid/path)'));
  await check('frontdoor local links checked', state => append(state, '[Missing](missing.md)', 'llms.txt'), 'broken or escaping link');
  await check('support local links checked', state => append(state, '[Missing](missing.md)', '.github/SUPPORT.md'), 'broken or escaping link');
  await check('changed support links checked', state => state.documents.set('docs/llm-guide/support.md', '[Missing](missing.md)'), 'broken or escaping link');
  await check('support unmarked Sounio examples need no bindings', state => state.documents.set('docs/llm-guide/stdlib-index.md', '# Modules\n\n```sio\nuse example::library\n```\n'));
  await check('Copilot is links-only for snippets', state => state.documents.set('.github/copilot-instructions.md', '```sio\nuse example::library\n```\n'));
  await check('safe directory link with trailing slash', state => append(state, '[Examples](../../examples/)'));
  await check('safe nested directory link', state => append(state, '[Medlang](../../examples/medlang/)'));
  await check('safe repository root directory link', state => append(state, '[Repository](../../)'));
  await check('directory cannot be source evidence', state => state.manifest.examples[0].source = 'examples', 'unsafe or non-sio');
  await check('missing required document', state => state.documents.delete('llms.txt'), 'missing required document');
  await check('unmanifested root Sounio code refused', state => append(state, '```sio\nfn main() {}\n```', 'llms.txt'), 'missing immediately preceding');
  await check('even registered exact-source root Sounio code is prohibited', state => append(state, state.block, 'llms.txt'), 'Sounio fences are prohibited');
  await check('registered root Sounio drift cannot bypass source checks', state => append(state, state.block.replace('return 0;', 'return 42;'), 'llms.txt'), 'Sounio fences are prohibited');
  await check('duplicate registered root Sounio snippets are prohibited', state => append(state, state.block.repeat(2), 'llms.txt'), 'Sounio fences are prohibited');
  await check('root text fence allowed', state => append(state, '```text\n./bin/souc check file.sio\n```', 'llms.txt'));
  await check('stale blanket semicolon ban', state => append(state, '- No semicolons'), 'stale blanket claim');
  await check('stale semicolon ban with example', state => append(state, '- NO semicolons: `let value = 5` not `let value = 5;`'), 'stale blanket claim');
  await check('stale blanket generics ban', state => append(state, '- No generics.'), 'stale blanket claim');
  await check('legitimate semicolon in exact source allowed', () => {});
  await check('scoped advice and warnings allowed', state => append(state, 'Trailing semicolons are accepted.\nDo not claim "no semicolons".\nNo generics on this particular target route without evidence.'));
  await check('mocked compiler receipt never executes', state => state.manifest.execution_receipt = { state: 'passed', command: 'must-never-execute' });
  const firstObject = 'a'.repeat(40);
  const secondObject = 'b'.repeat(40);
  const binaryBody = Buffer.from([0, 255, 10, 65]);
  const unicodeBody = Buffer.from('Café\r\n');
  const batchFrame = (objectId, body) => Buffer.concat([Buffer.from(`${objectId} blob ${body.length}\n`), body, Buffer.from('\n')]);
  const batchCase = (name, bytes, objectIds, expectedBodies = null) => {
    let passed = false;
    try {
      const parsed = parseGitBatch(bytes, objectIds);
      passed = expectedBodies !== null && parsed.size === expectedBodies.length && expectedBodies.every(([objectId, body]) => parsed.get(objectId)?.equals(body));
    } catch { passed = expectedBodies === null; }
    cases.push({ name, passed });
    if (!passed) throw new Error(`Self-test failed: ${name}`);
  };
  batchCase('empty Git batch', Buffer.alloc(0), [], []);
  batchCase('binary Git blob preserves exact bytes', batchFrame(firstObject, binaryBody), [firstObject], [[firstObject, binaryBody]]);
  batchCase('UTF8 Git blob byte length preserved', batchFrame(firstObject, unicodeBody), [firstObject], [[firstObject, unicodeBody]]);
  batchCase('multiple Git blobs parsed by byte framing', Buffer.concat([batchFrame(firstObject, binaryBody), batchFrame(secondObject, unicodeBody)]), [firstObject, secondObject], [[firstObject, binaryBody], [secondObject, unicodeBody]]);
  batchCase('empty Git blob preserved', batchFrame(firstObject, Buffer.alloc(0)), [firstObject], [[firstObject, Buffer.alloc(0)]]);
  batchCase('missing Git header rejected', Buffer.from(firstObject), [firstObject]);
  batchCase('missing promised Git object rejected', Buffer.from(`${firstObject} missing\n`), [firstObject]);
  batchCase('wrong Git object identity rejected', batchFrame(secondObject, binaryBody), [firstObject]);
  batchCase('nonblob Git object rejected', Buffer.from(`${firstObject} tree 0\n\n`), [firstObject]);
  batchCase('unsafe Git object length rejected', Buffer.from(`${firstObject} blob 9007199254740992\n\n`), [firstObject]);
  batchCase('truncated Git body rejected', batchFrame(firstObject, binaryBody).subarray(0, -2), [firstObject]);
  batchCase('missing Git body delimiter rejected', batchFrame(firstObject, binaryBody).subarray(0, -1), [firstObject]);
  batchCase('extra Git objects rejected', batchFrame(firstObject, binaryBody), []);
  batchCase('extra Git trailing bytes rejected', Buffer.concat([batchFrame(firstObject, binaryBody), Buffer.from('extra')]), [firstObject]);
  batchCase('omitted second Git object rejected', batchFrame(firstObject, binaryBody), [firstObject, secondObject]);
  return { schema: 'sounio.llm-guide.self-test.v1', scope: 'structural-only', execution: { compiler_invoked: false, filesystem_mutations: false }, cases: cases.length, errors: [] };
}

async function main() {
  const argumentsSet = new Set(process.argv.slice(2));
  for (const argument of argumentsSet) if (!['--json', '--self-test'].includes(argument)) throw new Error(`Unknown argument: ${argument}`);
  let result;
  if (argumentsSet.has('--self-test')) result = await selfTest();
  else {
    const repository = diskRepository();
    const manifest = JSON.parse((await repository.read(manifestFile)).toString('utf8'));
    const revision = /^[a-f0-9]{40}$/.test(manifest?.source_revision ?? '') ? manifest.source_revision : 'HEAD';
    const files = new Set([...guides, cookbook, ...frontdoors, ...changedSupportFiles(revision)]);
    if (await repository.entry(atlas)) files.add(atlas);
    const documents = new Map();
    for (const file of files) if ((await repository.entry(file))?.kind === 'file') documents.set(file, (await repository.read(file)).toString('utf8'));
    result = await validate(manifest, documents, repository);
  }
  if (argumentsSet.has('--json')) console.log(JSON.stringify(result, null, 2));
  else if (result.errors.length) console.error(`LLM guide structural validation failed:\n${result.errors.map(error => `- ${error}`).join('\n')}`);
  else if (argumentsSet.has('--self-test')) console.log(`LLM guide validator self-tests passed (${result.cases} pure cases; no compiler execution).`);
  else console.log(`LLM guide structural validation passed: ${result.documents} documents, ${result.examples} examples, ${result.source_files} source files, ${result.snippets} snippets, ${result.links} links; no compiler execution.`);
  if (result.errors.length) process.exitCode = 1;
}

export { manifestIssues, scanMarkdown, snippetIssues };

const invokedDirectly = import.meta.main ?? (process.argv[1] && (path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
  || await realpath(process.argv[1]).catch(() => null) === fileURLToPath(import.meta.url)));

if (invokedDirectly) {
  main().catch(error => {
    const result = { schema: 'sounio.llm-guide.validation.v1', scope: 'structural-only', execution: { compiler_invoked: false, receipts_verified: false }, errors: [error.message] };
    if (process.argv.includes('--json')) console.log(JSON.stringify(result, null, 2));
    else console.error(`LLM guide structural validation failed: ${error.message}`);
    process.exitCode = 1;
  });
}
