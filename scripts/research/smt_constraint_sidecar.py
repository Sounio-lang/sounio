#!/usr/bin/env python3
"""Z3 SMT sidecar for refinement-type constrained decoding (Pilar VI).

Loads the refinement-type dataset emitted by extract_refinements.py and acts as
a feasibility oracle for partial Sounio programs during LLM beam search / DPO
pair generation.

CLI:  python3 smt_constraint_sidecar.py \
          --code "let x: Percentage = -5.0" \
          --refinements datasets/sounio-ai-refinements/refinement_types.jsonl
HTTP: python3 smt_constraint_sidecar.py --serve --port 7788
"""
import argparse, json, re, sys
from pathlib import Path

try:
    from z3 import Real, Int, Bool, And, Or, Not, Solver, sat, unsat
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ---------------------------------------------------------------------------
# Refinement loading
# ---------------------------------------------------------------------------
def load_refinements(path: Path) -> list:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def index_by_name(refs: list) -> dict:
    return {r['name']: r for r in refs}

# ---------------------------------------------------------------------------
# Predicate tree → Z3
# ---------------------------------------------------------------------------
_OP_TO_Z3 = {
    'gt':  lambda a, b: a >  b,
    'gte': lambda a, b: a >= b,
    'lt':  lambda a, b: a <  b,
    'lte': lambda a, b: a <= b,
    'eq':  lambda a, b: a == b,
    'neq': lambda a, b: a != b,
}

def predicate_to_z3(tree: dict, var_val):
    """Convert a predicate_tree node into a Z3 BoolRef using `var_val` for the
    refinement's bound variable. Numeric `rhs` is coerced to the type of
    `var_val` so int/real comparisons stay homogeneous.
    """
    if tree is None or not HAS_Z3:
        return None
    op = tree.get('op')
    if op in ('and', 'or'):
        l = predicate_to_z3(tree.get('left'),  var_val)
        r = predicate_to_z3(tree.get('right'), var_val)
        if l is None or r is None: return None
        return And(l, r) if op == 'and' else Or(l, r)
    if op in _OP_TO_Z3:
        rhs = tree.get('rhs')
        if isinstance(rhs, str):
            try: rhs = float(rhs)
            except ValueError: return None
        return _OP_TO_Z3[op](var_val, rhs)
    return None

# ---------------------------------------------------------------------------
# Partial-code parsing
# ---------------------------------------------------------------------------
RE_TYPED_LET   = re.compile(r'(?:let|var)\s+(\w+)\s*:\s*([A-Z]\w*)\s*=\s*([^\n;]+)')
RE_TYPED_PARAM = re.compile(r'(\w+)\s*:\s*([A-Z]\w*)\b')
RE_NUMERIC     = re.compile(r'^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$')

def _coerce_value(s: str):
    s = s.strip().rstrip(',;')
    m = RE_NUMERIC.match(s)
    if not m: return None
    raw = m.group(1)
    return float(raw) if '.' in raw or 'e' in raw.lower() else int(raw)

def _z3_var_for(base_type: str):
    if base_type.lower().startswith('i'): return Int('x')
    return Real('x')

def check_assignment(partial_code: str, refinements: list) -> dict:
    """Check whether typed-and-assigned variables in `partial_code` satisfy
    their refinement predicates. Returns {feasible, violated, confidence}.
    Unrecognised types or partial assignments are skipped (treated as feasible).
    """
    idx = index_by_name(refinements)
    for m in RE_TYPED_LET.finditer(partial_code):
        _, tname, rhs = m.group(1), m.group(2), m.group(3)
        rec = idx.get(tname)
        if rec is None: continue
        val = _coerce_value(rhs)
        if val is None: continue
        tree = rec.get('predicate_tree')
        if tree is None: continue
        if not HAS_Z3:
            ok = _regex_check(tree, val)
            if ok is False:
                return {'feasible': False,
                        'violated': f"{tname}: {rec.get('predicate_str','')} (got {val})",
                        'confidence': 0.5}
            continue
        x = _z3_var_for(rec.get('base_type', 'f64'))
        constraint = predicate_to_z3(tree, x)
        if constraint is None: continue
        s = Solver(); s.add(x == val); s.add(constraint)
        if s.check() == unsat:
            return {'feasible': False,
                    'violated': f"{tname}: {rec.get('predicate_str','')} (got {val})",
                    'confidence': 0.95}
    return {'feasible': True, 'violated': None, 'confidence': 0.95 if HAS_Z3 else 0.5}

def _regex_check(tree: dict, val) -> bool:
    """Fallback evaluator when Z3 is unavailable."""
    op = tree.get('op')
    if op in ('and', 'or'):
        l = _regex_check(tree['left'], val); r = _regex_check(tree['right'], val)
        return (l and r) if op == 'and' else (l or r)
    rhs = tree.get('rhs')
    try: rhs = float(rhs)
    except (TypeError, ValueError): return True
    fv = float(val)
    return {'gt': fv>rhs, 'gte': fv>=rhs, 'lt': fv<rhs,
            'lte': fv<=rhs, 'eq': fv==rhs, 'neq': fv!=rhs}.get(op, True)

# ---------------------------------------------------------------------------
# Batch + HTTP
# ---------------------------------------------------------------------------
def batch_check(samples: list, refinements: list) -> list:
    return [check_assignment(s, refinements)['feasible'] for s in samples]

def serve(refinements: list, port: int):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print('flask not installed; install via `pip install flask`', file=sys.stderr); sys.exit(2)
    app = Flask(__name__)
    @app.post('/check')
    def _check():
        payload = request.get_json(force=True) or {}
        refs = payload.get('refinements') or refinements
        return jsonify(check_assignment(payload.get('code', ''), refs))
    app.run(host='127.0.0.1', port=port)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Z3 SMT sidecar for Sounio refinements.')
    ap.add_argument('--refinements', default='datasets/sounio-ai-refinements/refinement_types.jsonl')
    ap.add_argument('--code', default=None)
    ap.add_argument('--serve', action='store_true')
    ap.add_argument('--port', type=int, default=7788)
    args = ap.parse_args()

    refs = load_refinements(Path(args.refinements))
    if not HAS_Z3:
        print('warning: z3-solver not installed; using regex fallback', file=sys.stderr)
    if args.serve:
        serve(refs, args.port); return
    if args.code is None:
        ap.error('--code is required unless --serve is set')
    res = check_assignment(args.code, refs)
    if res['feasible']:
        print('FEASIBLE')
    else:
        print(f'INFEASIBLE: {res["violated"]}')
        sys.exit(1)

if __name__ == '__main__':
    main()
