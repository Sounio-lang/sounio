#!/usr/bin/env python3
"""Machine-generated census of what stdlib declares and whether the language knows it.

Written because a hand-kept inventory becomes another stale door. This regenerates
from the tree, so it goes wrong loudly the moment the tree moves.

Per declared type it answers five questions, and they are the five that separated
stdlib/cybernetic (alive, no types) from the ZD family (types, no life):

  declared      where it is
  importers     is the module alive outside itself
  param_use     is the type usable in a parameter position anywhere
  kind          does the compiler have a TypeExprKind naming the same concept
  mut_spine     does the *mut lowering spine handle that kind (parameter position)

TWO THINGS THIS CENSUS IS NOT, corrected 2026-08-20 before either misled anyone.

1. `kind` being false is NORMAL and healthy. Every ordinary struct is lowered by
   the generic `TypeExprKind::TypeNamed` path, which is present on BOTH spines and
   type-checks correctly -- passing an i64 where a declared struct is expected
   gives E009. A plain data type neither needs nor should have a dedicated kind.
   So "17 of 3,156 have a compiler kind" is NOT a defect count: a dedicated kind
   is for a type making a SPECIAL CLAIM, and those are rare by design.

2. The `kind` column has FALSE POSITIVES. Kinds are extracted from ast.sio by the
   pattern `Type<Name>,`, which turns the AST node types `TypeExpr` and
   `TypeExprKind` into phantom kinds "Expr" and "ExprKind". Any stdlib struct so
   named matches spuriously -- two of the five names reported off the *mut spine
   in the first run were exactly that.

   For the authoritative figure on claims lost in parameter position, read
   scripts/ci/silent_type_spine_ratchet_gate.sh: 20 kinds handled by one spine and
   not the other, measured from the two `match` bodies directly rather than by
   name collision with stdlib.

What this census IS good for: `param_uses` -- 53% of stdlib types never appear in
a parameter position, and that figure is robust across two extraction strategies
with opposite biases -- and `module_importers`, which says what is alive at all.

Usage:  python3 scripts/dev/stdlib_type_census.py [--json out.json] [--module NAME]
"""
import json, os, re, subprocess, sys, collections

ROOT = subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
os.chdir(ROOT)

def git_files(pat):
    out = subprocess.run(["git","ls-files",pat],capture_output=True,text=True).stdout
    return [l for l in out.split("\n") if l]

def read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f: return f.read()
    except OSError: return ""

DECL = re.compile(r'^\s*(?:pub\s+)?(struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)', re.M)

def compiler_kinds():
    """TypeExprKind variants, and which of them the *mut spine handles."""
    ast = read("self-hosted/parser/ast.sio")
    kinds = set(re.findall(r'\bType([A-Za-z0-9_]+),', ast))
    chk = read("self-hosted/check/check.sio")
    m = re.search(r'fn checker_lower_type_expr_mut\b', chk)
    mut = set()
    if m:
        window = chk[m.start(): m.start()+14000]
        mut = set(re.findall(r'TypeExprKind::Type([A-Za-z0-9_]+)', window))
    return kinds, mut

# The tree is read ONCE. An earlier revision re-read every .sio per module and
# took over two minutes for ~50 modules; the census has to be cheap enough that
# nobody is tempted to keep a hand-written copy instead.
_CORPUS = None
def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = [(f, read(f)) for f in git_files("*.sio")
                   if not f.startswith(("archive/", "bootstrap/"))]
    return _CORPUS

def all_module_importers(mods):
    """One pass, and one regex per FILE rather than one per (file, module).

    Two earlier revisions timed out: the first re-read the tree per module, the
    second ran ~50 patterns over every file. Here each file yields its own set of
    module-ish tokens once, and the sets are intersected. The census must be cheap
    or somebody will keep a hand copy, which is the failure this whole thing
    exists to prevent.
    """
    want = set(mods)
    tok = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)::|^\s*use\s+([A-Za-z_][A-Za-z0-9_]*)', re.M)
    out = {m: 0 for m in mods}
    for f, src in corpus():
        names = set()
        for a, b in tok.findall(src):
            n = a or b
            if n in want: names.add(n)
        own = f.split("/")[1] if f.startswith("stdlib/") and "/" in f[7:] else None
        for n in names:
            if n != own: out[n] += 1
    return out

def main():
    args = sys.argv[1:]
    only = None
    if "--module" in args: only = args[args.index("--module")+1]
    jsonout = args[args.index("--json")+1] if "--json" in args else None

    kinds, mut = compiler_kinds()

    # every param-position type name used anywhere in the corpus
    param_names = collections.Counter()
    for f, src in corpus():
        # Two extraction strategies were run against the whole tree and both put
        # "never used in a parameter" at 53%: `[^)]*` (single-line, misses the
        # ~6% multi-line signatures) gave 1,659, and `[^()]*` with re.S (catches
        # multi-line, refuses parameter lists containing parentheses such as fn
        # types) gave 1,662. Each has a bias and they point opposite ways, so the
        # figure is robust to the method rather than an artefact of it. That is
        # worth more than either number alone.
        #
        # re.S so multi-line signatures count: ~6% of the corpus (7,604 of
        # 124,073) opens the paren on one line and closes it on another. An
        # earlier revision missed those and would have over-reported "never used
        # in a parameter" by that much.
        for m in re.finditer(r'fn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^()]*)\)', src, re.S):
            for t in re.findall(r':\s*([A-Za-z_][A-Za-z0-9_]*)', m.group(1)):
                param_names[t] += 1

    mods = sorted({p.split("/")[1] for p in git_files("stdlib/*") if "/" in p[7:]})
    if only: mods = [m for m in mods if m == only]
    imps = all_module_importers(mods)
    rows = []
    for mod in mods:
        files = git_files("stdlib/%s/*.sio" % mod)
        if not files: continue
        imp = imps.get(mod, 0)
        for f in files:
            for kindword, name in DECL.findall(read(f)):
                has_kind = name in kinds
                rows.append({
                    "module": mod, "name": name, "decl": kindword, "file": f,
                    "module_importers": imp,
                    "param_uses": param_names.get(name, 0),
                    "compiler_kind": has_kind,
                    "mut_spine": (name in mut) if has_kind else None,
                })

    if jsonout:
        with open(jsonout,"w") as fh: json.dump(rows, fh, indent=1)

    print("STDLIB_TYPE_CENSUS types=%d modules=%d compiler_kinds=%d mut_spine_kinds=%d"
          % (len(rows), len({r['module'] for r in rows}), len(kinds), len(mut)))
    print()
    print("%-16s %6s %6s %6s %6s  %s" % ("MODULE","TYPES","IMPS","PARAM","KIND","dead-in-params"))
    per = collections.defaultdict(list)
    for r in rows: per[r["module"]].append(r)
    for mod in sorted(per, key=lambda m: -len(per[m])):
        rs = per[mod]
        used = sum(1 for r in rs if r["param_uses"] > 0)
        withk = sum(1 for r in rs if r["compiler_kind"])
        print("%-16s %6d %6d %6d %6d  %d" % (
            mod, len(rs), rs[0]["module_importers"], used, withk, len(rs)-used))

if __name__ == "__main__":
    main()
