#!/usr/bin/env python3
"""Extract bilingual Sounio ↔ Lean 4 training pairs from formal/.

Produces a JSONL training dataset for Pilar IV (Lean-Proof-Guided Generation).
Usage: python3 scripts/research/extract_lean_pairs.py --repo /workspace/sounio
"""
import argparse, hashlib, json, re, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
RE_LEAN_DECL = re.compile(
    r'^(?:@\[[^\]]+\]\s*)?(?:noncomputable\s+)?(theorem|lemma|def|abbrev|example)\s+(\w+)',
    re.MULTILINE
)
RE_SIO_FN  = re.compile(r'^(?:pub\s+)?fn\s+(\w+)\s*\(', re.MULTILINE)
RE_MIRROR  = re.compile(r'`(stdlib/[^`\s]+\.sio|self-hosted/[^`\s]+\.sio|examples/[^`\s]+\.sio)`')

NAME_MAP = {
    'knightian':   'stdlib/epistemic/knightian.sio',
    'gum':         'stdlib/epistemic/gum.sio',
    'knowledge':   'stdlib/epistemic/knowledge.sio',
    'hessianad':   'self-hosted/compiler/lean_single.sio',
    'typechecker': 'self-hosted/check/check.sio',
    'effects':     'self-hosted/check/effects.sio',
    'lineartypes': 'self-hosted/check/linear.sio',
    'vancomycin':  'stdlib/clinical/vancomycin_pbpk.sio',
    'epistemic':   'stdlib/epistemic/knowledge.sio',
}

DOMAIN_MAP = [
    ('clinical',   ('vancomycin', 'clinical')),
    ('epistemic',  ('knightian', 'gum', 'knowledge', 'epistemic', 'klibanoff', 'frechet', 'measconf')),
    ('compiler',   ('typechecker', 'effects', 'lineartypes', 'hessian', 'linear', 'channel',
                    'scheduler', 'gradient', 'elflinker', 'ieee754')),
    ('algebra',    ('octonion', 'clifford', 'multiquad', 'fano', 'erdos', 'moserspindle',
                    'cayley', 'composition', 'tropical', 'degrey')),
    ('regulatory', ('regulatory',)),
]

OUT_RELPATH = 'datasets/sounio-ai-lean-pairs/lean_pairs.jsonl'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chash(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def domain_of(stem):
    s = stem.lower()
    for dom, keys in DOMAIN_MAP:
        if any(k in s for k in keys):
            return dom
    return 'compiler'


def resolve_sio(lean_path, lean_text, repo):
    stem = lean_path.stem.lower().replace('sounio', '', 1)
    for key, rel in NAME_MAP.items():
        if key in stem:
            p = repo / rel
            if p.exists():
                return p
    m = RE_MIRROR.search(lean_text[:2000])
    if m:
        p = repo / m.group(1)
        if p.exists():
            return p
    return None


def extract_lean_blocks(text):
    """Return [(kind, name, body), ...]."""
    matches = list(RE_LEAN_DECL.finditer(text))
    out = []
    for i, m in enumerate(matches):
        nxt = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():nxt].rstrip()
        body = re.sub(r'\n-- =+[\s\S]*$', '', body)
        body = re.sub(r'\n(?:end|namespace)\s+[\w\.]+\s*$', '', body)
        out.append((m.group(1), m.group(2), body.strip()))
    return out


def extract_sio_fns(text):
    """Return {name: body} for top-level Sounio functions."""
    out = {}
    for m in RE_SIO_FN.finditer(text):
        name, start = m.group(1), m.start()
        depth, end, started = 0, start, False
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1; started = True
            elif ch == '}':
                depth -= 1
                if started and depth == 0:
                    end = start + i + 1
                    break
        out[name] = text[start:end]
    return out


def fuzzy_pair(thm_name, sio_fns):
    """Find Sounio fn whose name shares a 3+ char token with the theorem."""
    base = thm_name.lower()
    tokens = [t for t in re.split(r'_+', base) if len(t) >= 3]
    if not tokens:
        return None
    best = None
    for fn_name in sio_fns:
        fl = fn_name.lower()
        if any(t in fl or (len(fl) >= 3 and fl in base) for t in tokens):
            if best is None or len(fn_name) < len(best):
                best = fn_name
    return best


def has_real_proof(body):
    if ':=' not in body:
        return False
    rhs = body.split(':=', 1)[1]
    if 'by' not in rhs:
        return False
    tac = rhs.split('by', 1)[1]
    tac = re.sub(r'--[^\n]*', '', tac)
    tac = re.sub(r'/-[\s\S]*?-/', '', tac)
    tac = re.sub(r'\bsorry\b', '', tac).strip()
    return len(tac) > 0


def statement_only(body):
    """Return theorem signature with `:= by sorry` body."""
    head = body.split(':=', 1)[0].rstrip()
    return f'{head} := by\n  sorry'



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Extract Sounio↔Lean training pairs.')
    ap.add_argument('--repo', default='/workspace/sounio')
    args = ap.parse_args()
    repo = Path(args.repo)
    out  = repo / OUT_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)

    lean_files = sorted((repo / 'formal').rglob('*.lean'))
    print(f'Scanning {len(lean_files)} .lean files…', file=sys.stderr)

    seen = set()
    counts = {'lean_from_sounio': 0, 'sounio_from_lean': 0, 'lean_proof_sketch': 0}
    paired = unpaired = total_thms = 0

    with out.open('w') as fh:
        for lp in lean_files:
            try:
                ltext = lp.read_text(errors='replace')
            except OSError:
                continue
            blocks = extract_lean_blocks(ltext)
            total_thms += len(blocks)
            sp = resolve_sio(lp, ltext, repo)
            sio_fns = {}
            if sp is not None and sp.exists():
                paired += 1
                try:
                    sio_fns = extract_sio_fns(sp.read_text(errors='replace'))
                except OSError:
                    sio_fns = {}
            else:
                unpaired += 1
            lean_rel = str(lp.relative_to(repo))
            sio_rel  = str(sp.relative_to(repo)) if sp else ''
            dom = domain_of(lp.stem)
            for kind, name, body in blocks:
                key = chash(lean_rel + '::' + name)
                if key in seen:
                    continue
                seen.add(key)
                fn_name = fuzzy_pair(name, sio_fns) if sio_fns else None
                fn_body = sio_fns.get(fn_name, '') if fn_name else ''
                hp = has_real_proof(body)
                base = {
                    'lean_file': lean_rel, 'sounio_file': sio_rel,
                    'theorem_name': name, 'has_proof': hp, 'domain': dom,
                }
                if fn_body:
                    fh.write(json.dumps({**base, 'task': 'lean_from_sounio',
                        'instruction': 'Given the Sounio implementation, write the corresponding Lean 4 specification or theorem statement.',
                        'input': fn_body, 'output': body}) + '\n')
                    counts['lean_from_sounio'] += 1
                    fh.write(json.dumps({**base, 'task': 'sounio_from_lean',
                        'instruction': 'Given the Lean 4 specification, implement the Sounio function that satisfies it.',
                        'input': body, 'output': fn_body}) + '\n')
                    counts['sounio_from_lean'] += 1
                if hp and kind in ('theorem', 'lemma'):
                    fh.write(json.dumps({**base, 'task': 'lean_proof_sketch',
                        'instruction': 'Fill in the Lean 4 proof tactics for this theorem.',
                        'input': statement_only(body), 'output': body}) + '\n')
                    counts['lean_proof_sketch'] += 1

    total = sum(counts.values())
    print(f'\nExtracted {total} records → {out}')
    print(f'  total theorem-like blocks: {total_thms}')
    print(f'  files with sounio pair:    {paired}')
    print(f'  files without pair:        {unpaired}')
    for task, n in counts.items():
        print(f'  {task}: {n}')


if __name__ == '__main__':
    main()
