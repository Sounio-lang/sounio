#!/usr/bin/env python3
"""Extract Sounio refinement type definitions into a structured JSONL dataset.

Produces a JSONL training/oracle dataset for Pilar VI (Refinement-Type SMT
Constrained Decoding).
Usage: python3 scripts/research/extract_refinements.py --repo /workspace/sounio
"""
import argparse, json, re, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
RE_TYPE_REF = re.compile(
    r'^\s*(?:pub\s+)?type\s+(\w+)\s*=\s*\{\s*(\w+)\s*:\s*([A-Za-z_][\w<>]*)\s*\|\s*([^}]+?)\s*\}',
    re.MULTILINE
)
RE_STRUCT   = re.compile(
    r'^\s*(?:pub\s+)?struct\s+(\w+)\s*\{\s*value\s*:\s*([A-Za-z_][\w<>]*)\s*\}',
    re.MULTILINE
)
RE_IS_VALID = re.compile(
    r'^\s*(?:pub\s+)?fn\s+([a-z0-9_]+)_is_valid\s*\(\s*(\w+)\s*:\s*[A-Za-z_][\w<>]*\s*\)'
    r'\s*->\s*bool\s*(?:with\s+[\w,\s]+)?\s*\{\s*([^}]+?)\s*\}',
    re.MULTILINE
)
RE_FN_PARAM = re.compile(r'fn\s+(\w+)\s*\(([^)]*)\)')

SKIP_DIRS  = ('archive/', 'artifacts/', '.git/', 'target/', 'build/', 'node_modules/')
OUT_RELPATH = 'datasets/sounio-ai-refinements/refinement_types.jsonl'

# Map snake_case fn prefix → PascalCase struct name
def snake_to_pascal(s: str) -> str:
    return ''.join(p.capitalize() for p in s.split('_'))

# ---------------------------------------------------------------------------
# Predicate parser
# ---------------------------------------------------------------------------
TOK_RE = re.compile(
    r'\s*(\|\||&&|>=|<=|==|!=|>|<|=|\(|\)|'
    r'[A-Za-z_][A-Za-z0-9_\.]*|[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+)'
)

def tokenize(s: str) -> list:
    toks, pos = [], 0
    while pos < len(s):
        m = TOK_RE.match(s, pos)
        if not m or m.group(1) is None:
            break
        toks.append(m.group(1))
        pos = m.end()
    return toks

OP_MAP = {'>': 'gt', '>=': 'gte', '<': 'lt', '<=': 'lte', '==': 'eq', '!=': 'neq'}

class Parser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def eat(self):
        x = self.peek(); self.i += 1; return x
    def parse_or(self):
        node = self.parse_and()
        while self.peek() == '||':
            self.eat(); right = self.parse_and()
            node = {'op': 'or', 'left': node, 'right': right}
        return node
    def parse_and(self):
        node = self.parse_cmp()
        while self.peek() == '&&':
            self.eat(); right = self.parse_cmp()
            node = {'op': 'and', 'left': node, 'right': right}
        return node
    def parse_cmp(self):
        if self.peek() == '(':
            self.eat(); node = self.parse_or()
            if self.peek() == ')': self.eat()
            return node
        lhs = self.eat()
        op  = self.eat()
        rhs = self.eat()
        if op not in OP_MAP or rhs is None:
            return None
        try:
            rv = float(rhs) if '.' in rhs or 'e' in rhs.lower() else int(rhs)
        except ValueError:
            rv = rhs
        return {'op': OP_MAP[op], 'var': lhs, 'rhs': rv}

def parse_predicate(s: str):
    toks = tokenize(s)
    if not toks: return None
    try:
        return Parser(toks).parse_or()
    except (IndexError, KeyError):
        return None

# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------
def domain_of(rel: str) -> str:
    r = rel.lower()
    if 'clinical' in r or 'medlang' in r or 'medical' in r: return 'clinical'
    if 'pbpk'     in r or 'darwin'  in r:                   return 'pbpk'
    if 'physics'  in r or 'particle' in r:                  return 'physics'
    if r.startswith('stdlib/math') or 'stats' in r or 'special' in r: return 'math'
    return 'general'

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def line_of(text: str, offset: int) -> int:
    return text.count('\n', 0, offset) + 1

def extract_from_file(path: Path, repo: Path) -> list:
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return []
    rel  = str(path.relative_to(repo))
    recs = []
    # Pattern A: `type X = { v: T | pred }`
    for m in RE_TYPE_REF.finditer(text):
        name, var, base, pred = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        recs.append({'name': name, 'base_type': base, 'var': var,
                     'predicate_str': pred, 'predicate_tree': parse_predicate(pred),
                     'source_file': rel, 'line': line_of(text, m.start()),
                     'domain': domain_of(rel), 'usages': []})
    # Pattern B: struct X { value: T } + fn x_is_valid(v: T) -> bool { pred }
    structs = {m.group(1): (m.group(2), line_of(text, m.start()))
               for m in RE_STRUCT.finditer(text)}
    for m in RE_IS_VALID.finditer(text):
        prefix, var, body = m.group(1), m.group(2), m.group(3).strip()
        pascal = snake_to_pascal(prefix)
        if pascal not in structs: continue
        base, sline = structs[pascal]
        recs.append({'name': pascal, 'base_type': base, 'var': var,
                     'predicate_str': body, 'predicate_tree': parse_predicate(body),
                     'source_file': rel, 'line': sline,
                     'domain': domain_of(rel), 'usages': []})
    return recs

def collect_files(repo: Path) -> list:
    out = []
    for p in repo.rglob('*.sio'):
        rel = str(p.relative_to(repo))
        if any(rel.startswith(d) for d in SKIP_DIRS): continue
        out.append(p)
    return out

def find_all_usages(names: set, files: list, repo: Path, cap: int = 8) -> dict:
    """Single-pass usage scan across all files for all names."""
    usages: dict = {n: [] for n in names}
    pat = re.compile(r'fn\s+\w+\s*\(([^)]*)\)')
    for p in files:
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
        for m in pat.finditer(text):
            params = m.group(1)
            for name in names:
                if len(usages[name]) < cap and re.search(rf'\b{re.escape(name)}\b', params):
                    usages[name].append(m.group(0).strip())
    return usages

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Extract Sounio refinement types.')
    ap.add_argument('--repo', default='/workspace/sounio')
    args = ap.parse_args()
    repo = Path(args.repo)
    out  = repo / OUT_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(repo)
    print(f'Scanning {len(files)} .sio files…', file=sys.stderr)

    recs, seen = [], set()
    for p in files:
        for r in extract_from_file(p, repo):
            key = (r['name'], r['source_file'])
            if key in seen: continue
            seen.add(key); recs.append(r)

    print(f'Found {len(recs)} refinement types — scanning usages…', file=sys.stderr)
    names = {r['name'] for r in recs}
    usages = find_all_usages(names, files, repo)
    for r in recs:
        r['usages'] = usages.get(r['name'], [])

    by_dom, by_base = {}, {}
    with out.open('w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
            by_dom[r['domain']]    = by_dom.get(r['domain'], 0) + 1
            by_base[r['base_type']] = by_base.get(r['base_type'], 0) + 1

    print(f'\nExtracted {len(recs)} refinement types → {out}')
    print('  by domain:    ' + ', '.join(f'{k}={v}' for k, v in sorted(by_dom.items())))
    print('  by base_type: ' + ', '.join(f'{k}={v}' for k, v in sorted(by_base.items())))

if __name__ == '__main__':
    main()
