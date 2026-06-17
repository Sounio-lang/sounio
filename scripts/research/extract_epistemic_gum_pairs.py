#!/usr/bin/env python3
"""Extract f64 ↔ Knowledge<T>/Epistemic function pairs from Sounio source.

Produces a JSONL training dataset for Pilar II (Epistemic GUM Code Generation).
Usage: python3 scripts/research/extract_epistemic_gum_pairs.py --repo /workspace/sounio
"""
import argparse, hashlib, json, re, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
RE_FN_HEAD = re.compile(
    r'^(pub\s+)?fn\s+(\w+)\s*\(', re.MULTILINE
)
RE_KNOW   = re.compile(r'Knowledge<|Epistemic\s*\{|measure\(')
RE_GUM    = re.compile(
    r'\.variance\s*[\+\*/]'
    r'|\.val\s*\*\s*\w+\.val\s*\*\s*\w+\.variance'
    r'|\w+\.variance\s*/\s*\(\w+\.val'
)
RE_F64_FN = re.compile(r'(?:pub\s+)?fn\s+(\w+)_f64\s*\(')

SCAN_DIRS    = ['stdlib/clinical', 'stdlib/darwin_pbpk', 'examples', 'tests/run-pass']
VANC_PAT     = re.compile(r'vancomycin', re.IGNORECASE)
OUT_RELPATH  = 'datasets/sounio-ai-epistemic-gum/epistemic_gum_pairs.jsonl'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collect_files(repo: Path) -> list:
    files = []
    for d in SCAN_DIRS:
        p = repo / d
        if not p.exists():
            continue
        for f in p.rglob('*.sio'):
            if d == 'tests/run-pass' and not VANC_PAT.search(f.name):
                continue
            files.append(f)
    return files


def extract_fns(text: str) -> list:
    """Return [(name, start_line, end_line), ...] for top-level fns."""
    results = []
    for m in RE_FN_HEAD.finditer(text):
        name  = m.group(2)
        start = text[:m.start()].count('\n')
        depth, end = 0, start
        for i, ch in enumerate(text[m.start():]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = text[:m.start() + i].count('\n')
                    break
        results.append((name, start, end))
    return results


def window(lines: list, start: int, end: int, ctx: int = 30) -> str:
    lo = max(0, start - ctx)
    hi = min(len(lines), end + ctx + 1)
    return '\n'.join(lines[lo:hi])


def chash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def infer_meta(path: Path, text: str) -> dict:
    drug = next(
        (d for d in ('vancomycin', 'rapamycin', 'gentamicin', 'tobramycin')
         if d in text.lower() or d in str(path).lower()),
        ''
    )
    p = str(path)
    domain = ('clinical'  if 'clinical'  in p else
              'pbpk'      if ('pbpk' in p or 'darwin' in p) else
              'epistemic' if 'epistemic' in p else 'general')
    return {
        'drug': drug, 'domain': domain,
        'has_knowledge_t': bool(re.search(r'Knowledge<', text)),
        'has_pbox':        bool(re.search(r'PBox\s*\{', text)),
    }

# ---------------------------------------------------------------------------
# Task extractors
# ---------------------------------------------------------------------------
def task_a(path: Path, text: str, lines: list, seen: set) -> list:
    """f64_to_epistemic: _f64 fn with a Knowledge counterpart."""
    recs, fn_map = [], {n: (s, e) for n, s, e in extract_fns(text)}
    for m in RE_F64_FN.finditer(text):
        base = m.group(1)
        if base not in fn_map or (base + '_f64') not in fn_map:
            continue
        f64s, f64e = fn_map[base + '_f64']
        eps,  epe  = fn_map[base]
        ep_body = '\n'.join(lines[eps:epe + 1])
        if not RE_KNOW.search(ep_body):
            continue
        f64_body = '\n'.join(lines[f64s:f64e + 1])
        h = chash(f64_body + ep_body)
        if h in seen: continue
        seen.add(h)
        recs.append({
            'task': 'f64_to_epistemic',
            'instruction': 'Convert this Sounio function from plain f64 arithmetic to use Knowledge<T> with proper GUM uncertainty propagation.',
            'input':  f64_body,
            'output': window(lines, eps, epe, 5),
            'source_file': '', 'metadata': infer_meta(path, text),
        })
    return recs


def task_b(path: Path, text: str, lines: list, seen: set) -> list:
    """epistemic_generation: any fn that uses Knowledge<, Epistemic{, measure(."""
    recs = []
    for name, start, end in extract_fns(text):
        body = '\n'.join(lines[start:end + 1])
        if not RE_KNOW.search(body): continue
        h = chash(body)
        if h in seen: continue
        seen.add(h)
        recs.append({
            'task': 'epistemic_generation',
            'instruction': f'Generate a Sounio function that computes [{name}] with epistemic uncertainty tracking using Knowledge<T>.',
            'input':  lines[start].strip(),
            'output': window(lines, start, end, 5),
            'source_file': '', 'metadata': infer_meta(path, text),
        })
    return recs


def task_c(path: Path, text: str, lines: list, seen: set) -> list:
    """gum_propagation: fns with explicit delta-method variance arithmetic."""
    recs = []
    for name, start, end in extract_fns(text):
        body = '\n'.join(lines[start:end + 1])
        if not RE_GUM.search(body): continue
        h = chash(body)
        if h in seen: continue
        seen.add(h)
        var_lines = [l.strip() for l in lines[start:end + 1] if RE_GUM.search(l)]
        recs.append({
            'task': 'gum_propagation',
            'instruction': 'Implement GUM (JCGM 100:2008) delta-method uncertainty propagation for this Sounio arithmetic operation.',
            'input':  '; '.join(var_lines[:3]),
            'output': body,
            'source_file': '', 'metadata': infer_meta(path, text),
        })
    return recs

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Extract epistemic GUM training pairs.')
    ap.add_argument('--repo', default='/workspace/sounio')
    args   = ap.parse_args()
    repo   = Path(args.repo)
    out    = repo / OUT_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)

    files  = collect_files(repo)
    print(f'Scanning {len(files)} .sio files…', file=sys.stderr)

    seen: set = set()
    counts = {'f64_to_epistemic': 0, 'epistemic_generation': 0, 'gum_propagation': 0}

    with out.open('w') as fh:
        for path in files:
            try:
                text = path.read_text(errors='replace')
            except OSError:
                continue
            lines = text.splitlines()
            rel   = str(path.relative_to(repo))
            for rec in (task_a(path, text, lines, seen) +
                        task_b(path, text, lines, seen) +
                        task_c(path, text, lines, seen)):
                rec['source_file'] = rel
                fh.write(json.dumps(rec) + '\n')
                counts[rec['task']] += 1

    total = sum(counts.values())
    print(f'\nExtracted {total} records → {out}')
    for task, n in counts.items():
        print(f'  {task}: {n}')

if __name__ == '__main__':
    main()
