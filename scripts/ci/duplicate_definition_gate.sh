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
        code = re.sub(r'//.*$', '', lines[i])
        m = re.match(r'\s*impl\s+([A-Za-z_][A-Za-z0-9_]*)', code)
        f = None if m else re.match(r'\s*(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', code)
        if f:
            j, d, started = i, 0, False
            while j < len(lines):
                d += lines[j].count('{') - lines[j].count('}')
                if '{' in lines[j]: started = True
                if started and d <= 0: break
                j += 1
            scope = impl_stack[-1][0] if impl_stack else '<top>'
            seen[(scope, f.group(2))].append((i + 1, tokens('\n'.join(lines[i:j+1]))))
            i = j + 1                       # SKIP the body outright
            continue
        opened = depth
        depth += code.count('{') - code.count('}')
        if m:
            impl_stack.append((m.group(1), opened))
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

for path in files:
    for (sc, name), defs in scan(path).items():
        if len(defs) < 2: continue
        lines = [d[0] for d in defs]
        same = all(d[1] == defs[0][1] for d in defs[1:])
        (identical if same else divergent).append(
            {"file": path, "scope": sc, "name": name, "lines": lines})

frozen = {"identical": 0, "divergent": 0}
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

print(f"[duplicate-definition] identical={len(identical)} (frozen {frozen['identical']}) "
      f"divergent={len(divergent)} (frozen {frozen['divergent']})")

json.dump({"identical": identical, "divergent": divergent,
           "frozen": frozen}, open(out_path, "w"), indent=1)

fails = []
if len(divergent) > frozen["divergent"]:
    fails.append(f"divergent duplicates rose {frozen['divergent']} -> {len(divergent)}")
if len(identical) > frozen["identical"]:
    fails.append(f"identical duplicates rose {frozen['identical']} -> {len(identical)}")

if fails:
    print()
    print("  A second definition of the same name in the same scope means one of")
    print("  them is dead, and Sounio typechecks both -- so nothing else will tell")
    print("  you. When the bodies DIVERGE, every caller silently binds to whichever")
    print("  the compiler resolves.", file=sys.stderr)
    for f in fails: print(f"  REFUSE: {f}", file=sys.stderr)
    sys.exit(1)

if len(divergent) < frozen["divergent"] or len(identical) < frozen["identical"]:
    print(f"  OK, and lower than frozen. Update {ref_path}:")
    print(f"    identical={len(identical)}")
    print(f"    divergent={len(divergent)}")
sys.exit(0)
PY
rc=$?
[ $rc -eq 0 ] || gate_fail "duplicate definitions rose (see above)"
echo "DUPLICATE_DEFINITION_GATE_OK"
