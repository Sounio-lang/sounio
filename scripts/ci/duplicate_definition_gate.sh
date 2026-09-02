#!/usr/bin/env bash
# duplicate_definition_gate.sh — one name, one definition, per scope.
#
# WHY. #2341 removed one of two identical definitions of the tuple-slot
# accessors and said in its own title that NO GATE COULD SEE IT. That was
# true, and it was not one case: a sweep on 2026-08-31 found TEN functions
# defined twice at top level in the same file, five of them with genuinely
# different bodies under the same name -- `compiler_mode_positional_arg` has
# thirteen call sites and two bodies that skip different flags.
#
# Nothing catches this because Sounio requires helpers before callers, so BOTH
# copies typecheck and both are well-formed. The checker has no complaint to
# make about either one individually.
#
# It is not only a correctness question. #2305's rebase produced 22
# `error[E016] field initializer has wrong type` and read as two incompatible
# refactors of the Lowerer. It was not: a dead duplicate was keeping git from
# aligning hunks. Once #2341 removed it, the same merge came out clean, zero
# conflicts, zero errors.
#
# THREE WAYS TO MEASURE THIS WRONG, all of which I did before this settled:
#
#   1. count names           -> 104 "duplicates", mostly add/get/find in
#                               check/defs.sio, which are methods in DIFFERENT
#                               impl blocks. Hence scope tracking below.
#   2. compare bodies byte-wise -> 6 of 10 called "different"; several were the
#                               same code wrapped across more lines.
#   3. normalise whitespace  -> still called nc_emit_seta_al different: the same
#                               three nc_emit_byte calls, `;`-separated on one
#                               line versus three lines.
#
# So the comparison is on TOKEN SEQUENCES with comments and separators stripped.
# A weaker comparison reports formatting as divergence and would fail honest PRs.
#
# ---------------------------------------------------------------------------
# CROSS-MODULE half, added 2026-09-01 for #2368.
#
# The within-file scan above is blind to one name defined in TWO MODULES, and
# that blindness cost the project ~9,000 lines of dead optimiser.
# `compile_multimodule_native_advanced` is defined in both
# compiler/module_loader.sio:3190 and compiler/module_native_driver.sio:1239.
# Only the module_loader body runs inl_run_pass / lopt_optimize_module /
# tco_run_pass. main.sio:52 imports the module_native_driver one BY NAME, so
# the pipeline copy is unreachable and those three passes never ran at all.
# Across modules the loser is not merely dead, it is not even a candidate: no
# resolution happens, the import names a module and takes what is there.
#
# SCOPE, and it is a CHOICE. Measured over self-hosted/ on 2026-09-01, four
# cuts of "same top-level name in two files":
#
#     all top-level fn, any visibility     959   (507 identical / 452 divergent)
#     name imported anywhere by `use`       78   ( 14 / 64)
#     top-level `pub fn` (THIS GATE)        49   ( 22 / 27)
#     top-level `pub fn` AND imported       30   ( 14 / 16)
#
# 959 is not a gate, it is a wall: bootstrap/bootstrap_v0.sio alone is a frozen
# self-contained snapshot that redefines most of parser/ and native/ on purpose.
# The `pub fn` cut is what a module EXPORTS, it is 49 rows, and it contains
# #2368 -- so that is what is frozen.
#
# DO NON-PUB DUPLICATES MATTER? Yes, and this gate does not see them. Measured
# the same day: of 784 `use mod::{name}` imports in self-hosted/ that resolve to
# a top-level fn, 709 bind a `pub fn` and 75 bind a NON-pub one -- the importer
# does not enforce `pub`. So the pub cut is a LOWER BOUND on what can collide,
# not the collision set. It is the cut with a workable count today. Whoever
# widens it should start from the four numbers above, not re-derive them.
#
# Methods inside `impl` blocks are excluded because they are not module-level
# exports: two types may each have `get`, and that is the shape that produced a
# false 104 on the within-file scan's first attempt.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "duplicate_definition_gate"

REF="scripts/ci/duplicate_definition.frozen"
OUT="${GATE_ARTIFACT:-artifacts/gates/duplicate_definition.json}"
mkdir -p "$(dirname "$OUT")"

python3 - "$REF" "$OUT" "${SOUNIO_DUPDEF_ROOT:-self-hosted}" <<'PY'
import re, sys, json, glob, os, collections

ref_path, out_path, scan_root = sys.argv[1], sys.argv[2], sys.argv[3]

def tokens(text):
    text = re.sub(r'//.*$', '', text, flags=re.M)
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+|\d+|[^\s;]', text)

def structure(line):
    """One line with comments AND string/char literals removed.

    Only for finding braces and definitions -- never for comparing bodies,
    where two different string constants must still read as different.

    A FOURTH way to measure this wrong, found 2026-09-01: braces were counted
    inside string literals. module_frontend.sio:4355 holds the pattern
    "CorePair { left:", one line into a function, and that unmatched brace kept
    the body-skip walking for 4012 lines -- past a genuine duplicate at 6425,
    which this gate then reported as absent. Stripping literals brings 1905 more
    function names into view across self-hosted/ and raises the duplicate count
    from 6 to the frozen figures below."""
    line = re.sub(r'//.*$', '', line)
    line = re.sub(r'"([^"\\]|\\.)*"', '""', line)
    line = re.sub(r"'([^'\\]|\\.)*'", "''", line)
    return line

def scan(path):
    """Scope by BRACE DEPTH, walking with an explicit index.

    Two bugs lived here before this settled, and both presented as `impl`
    detection failing when neither was:

      1. popping the impl scope on a line equal to '}' -- closing braces in
         this tree are indented, so impl blocks never closed.
      2. accounting a function body's braces and then letting the loop walk
         through those same lines, counting them twice. depth reached -92 in
         check/defs.sio, and a negative depth makes every pop condition true,
         so the stack emptied and all 36 `get` methods landed in <top> as one
         name defined 36 times.

    Hence the explicit index: the body is skipped, not merely counted."""
    lines = open(path, errors='replace').read().split('\n')
    seen = collections.defaultdict(list)
    depth = 0
    impl_stack = []            # (name, depth at which the impl block opened)
    i = 0
    while i < len(lines):
        code = structure(lines[i])
        # `impl Trait for Type` is scoped by the TYPE, not the trait. Two impls
        # of one trait for two types are two different scopes; keying on the
        # trait collapsed them and called every method they share a duplicate.
        m = re.match(r'\s*impl\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*<[^>]*>)?\s+for\s+([A-Za-z_][A-Za-z0-9_]*)', code) \
            or re.match(r'\s*impl\s+([A-Za-z_][A-Za-z0-9_]*)', code)
        f = None if m else re.match(r'\s*(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', code)
        if f:
            j, d, started, decl_only = i, 0, False, False
            while j < len(lines):
                body = structure(lines[j])
                # A trait DECLARES methods with no body -- `fn radd(self) -> Self`
                # and nothing else. This walk looks for braces, so it ran past
                # the declaration and swallowed what followed, usually the impl
                # block, whose method then surfaced as a duplicate of the
                # signature it implements. Stop at the next definition or at the
                # end of the enclosing block, and record nothing for it.
                if j > i and not started and re.match(r'\s*(pub\s+)?(fn|impl|struct|trait)\s', body):
                    decl_only = True
                    break
                if j > i and not started and re.match(r'\s*\}', body):
                    decl_only = True
                    break
                d += body.count('{') - body.count('}')
                if '{' in body: started = True
                if started and d <= 0: break
                j += 1
            if decl_only:
                i = j
                continue
            scope = impl_stack[-1][0] if impl_stack else '<top>'
            # A definition guarded by #[cfg(...)] is an ALTERNATIVE, not a
            # duplicate: the arms are mutually exclusive by construction and
            # exactly one is compiled. Counting them made every
            # architecture-switched helper in examples/ look like a defect.
            guarded = i > 0 and re.match(r'\s*#\[\s*cfg', lines[i - 1])
            if not guarded:
                # The `pub` flag is recorded HERE, from the match that already
                # ran, rather than re-reading the line later: a second parse of
                # the same text is a second chance for the two to disagree.
                seen[(scope, f.group(2))].append(
                    (i + 1, tokens('\n'.join(lines[i:j+1])), bool(f.group(1))))
            i = j + 1                       # SKIP the body outright
            continue
        opened = depth
        depth += code.count('{') - code.count('}')
        if m:
            impl_name = m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else m.group(1)
            impl_stack.append((impl_name, opened))
        while impl_stack and depth <= impl_stack[-1][1]:
            impl_stack.pop()
        i += 1
    return seen

identical, divergent = [], []
files = sorted(glob.glob(f'{scan_root}/**/*.sio', recursive=True))
min_files = int(os.environ.get('SOUNIO_DUPDEF_MIN_FILES', '100'))
if len(files) < min_files:
    print(f"  CONTROL-FAIL: the scan saw only {len(files)} .sio files under {scan_root}/;")
    print( "                that is the pattern failing, not a tree without duplicates.")
    sys.exit(3)

exported = collections.defaultdict(list)   # name -> [(path, line, tokens)]

for path in files:
    for (sc, name), defs in scan(path).items():
        # CROSS-MODULE half. Scoped to top-level `pub fn` -- see the header.
        if sc == '<top>':
            for ln, tok, is_pub in defs:
                if is_pub:
                    exported[name].append((path, ln, tok))
        if len(defs) < 2: continue
        lines = [d[0] for d in defs]
        same = all(d[1] == defs[0][1] for d in defs[1:])
        (identical if same else divergent).append(
            {"file": path, "scope": sc, "name": name, "lines": lines})

x_identical, x_divergent = [], []
for name, defs in exported.items():
    if len({d[0] for d in defs}) < 2: continue     # one module: not cross-module
    defs = sorted(defs)
    same = all(d[2] == defs[0][2] for d in defs[1:])
    (x_identical if same else x_divergent).append(
        {"name": name, "defs": [f"{d[0]}:{d[1]}" for d in defs]})

frozen = {"identical": 0, "divergent": 0,
          "cross_module_identical": 0, "cross_module_divergent": 0}
if os.path.exists(ref_path):
    for line in open(ref_path):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split('=', 1)
        frozen[k.strip()] = int(v.strip())

for row in sorted(divergent, key=lambda r: (r["file"], r["name"])):
    print(f"  DIVERGENT  {row['file']}  [{row['scope']}] {row['name']}  lines {row['lines']}")
for row in sorted(identical, key=lambda r: (r["file"], r["name"])):
    print(f"  identical  {row['file']}  [{row['scope']}] {row['name']}  lines {row['lines']}")
for row in sorted(x_divergent, key=lambda r: r["name"]):
    print(f"  XMOD-DIVERGENT  {row['name']}  {' '.join(row['defs'])}")
for row in sorted(x_identical, key=lambda r: r["name"]):
    print(f"  xmod-identical  {row['name']}  {' '.join(row['defs'])}")

print(f"[duplicate-definition] identical={len(identical)} (frozen {frozen['identical']}) "
      f"divergent={len(divergent)} (frozen {frozen['divergent']})")
print(f"[duplicate-definition] cross_module_identical={len(x_identical)} "
      f"(frozen {frozen['cross_module_identical']}) "
      f"cross_module_divergent={len(x_divergent)} "
      f"(frozen {frozen['cross_module_divergent']})")

json.dump({"identical": identical, "divergent": divergent,
           "cross_module_identical": x_identical,
           "cross_module_divergent": x_divergent,
           "frozen": frozen}, open(out_path, "w"), indent=1)

fails = []
if len(divergent) > frozen["divergent"]:
    fails.append(f"divergent duplicates rose {frozen['divergent']} -> {len(divergent)}")
if len(identical) > frozen["identical"]:
    fails.append(f"identical duplicates rose {frozen['identical']} -> {len(identical)}")
if len(x_divergent) > frozen["cross_module_divergent"]:
    fails.append("cross-module divergent duplicates rose "
                 f"{frozen['cross_module_divergent']} -> {len(x_divergent)}")
if len(x_identical) > frozen["cross_module_identical"]:
    fails.append("cross-module identical duplicates rose "
                 f"{frozen['cross_module_identical']} -> {len(x_identical)}")

if fails:
    print()
    print("  A second definition of the same name in the same scope means one of")
    print("  them is dead, and Sounio typechecks both -- so nothing else will tell")
    print("  you. When the bodies DIVERGE, every caller silently binds to whichever")
    print("  the compiler resolves.")
    print("  ACROSS modules the same shape is worse, because the loser is not even")
    print("  reachable: #2368's `compile_multimodule_native_advanced` exists twice,")
    print("  main.sio imports the copy WITHOUT the inliner/loop/TCO pipeline by name,")
    print("  and ~9,000 lines of optimiser never ran.", file=sys.stderr)
    for f in fails: print(f"  REFUSE: {f}", file=sys.stderr)
    sys.exit(1)

if (len(divergent) < frozen["divergent"] or len(identical) < frozen["identical"]
        or len(x_divergent) < frozen["cross_module_divergent"]
        or len(x_identical) < frozen["cross_module_identical"]):
    print(f"  OK, and lower than frozen. Update {ref_path}:")
    print(f"    identical={len(identical)}")
    print(f"    divergent={len(divergent)}")
    print(f"    cross_module_identical={len(x_identical)}")
    print(f"    cross_module_divergent={len(x_divergent)}")
sys.exit(0)
PY
rc=$?
[ $rc -eq 0 ] || gate_fail "duplicate definitions rose (see above)"
echo "DUPLICATE_DEFINITION_GATE_OK"
