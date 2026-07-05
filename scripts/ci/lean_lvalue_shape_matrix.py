#!/usr/bin/env python3
"""Witness generator for lean_single lvalue-shape coverage.

Enumerates the ROOT x CHAIN x OP shape lattice for lvalue statements
(read / assign / borrow-mut) against the seed lean_single compiler, so
missing-handler and silent-miscompile gaps show up as explicit RED rows
in a TSV matrix instead of silently dropping stores or mis-borrowing.

Usage:
    python3 scripts/ci/lean_lvalue_shape_matrix.py [OUT_DIR]

Env:
    SOUNIO_LVALUE_SEED           path to the lean_single compiler binary
                                  (default: bin/souc-lean-single-x86_64)
    SOUNIO_SHAPE_MATRIX_BASELINE if "1", always exit 0 (documenting reds
                                  first; no build-breaking).

Output: a sorted TSV to stdout with columns:
    name  compile_rc  run_rc  verdict  first_output_line
followed by a summary line "SHAPES: <green>/<total> green".
"""

import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

COMPILE_TIMEOUT_S = 120
RUN_TIMEOUT_S = 10

K = 2  # constant array index used throughout the chain lattice
J = 1  # constant secondary array index (two-level indexing)

STRUCT_DEFS = """\
struct Inner { a: i64, b: i64, iarr: [i64; 3] }
struct Mid { sarr: [i64; 3], aarr: [Inner; 3], s: i64 }
struct Holder { s: i64, inner: Inner, sarr: [i64; 3], aarr: [Inner; 3], mid: Mid }

fn zero_inner() -> Inner {
    Inner { a: 0, b: 0, iarr: [0; 3] }
}
fn zero_mid() -> Mid {
    Mid { sarr: [0; 3], aarr: [zero_inner(); 3], s: 0 }
}
fn make_holder() -> Holder {
    Holder {
        s: %(s0)d,
        inner: Inner { a: %(ia)d, b: %(ib)d, iarr: [%(iarr0)d, %(iarr1)d, %(iarr2)d] },
        sarr: [%(sarr0)d, %(sarr1)d, %(sarr2)d],
        aarr: [
            Inner { a: %(aa0)d, b: %(ab0)d, iarr: [0, 0, 0] },
            Inner { a: %(aa1)d, b: %(ab1)d, iarr: [0, 0, 0] },
            Inner { a: %(aa2)d, b: %(ab2)d, iarr: [%(a2i0)d, %(a2i1)d, %(a2i2)d] }
        ],
        mid: Mid {
            sarr: [%(msarr0)d, %(msarr1)d, %(msarr2)d],
            aarr: [
                Inner { a: %(maa0)d, b: %(mab0)d, iarr: [0, 0, 0] },
                Inner { a: %(maa1)d, b: %(mab1)d, iarr: [0, 0, 0] },
                Inner { a: %(maa2)d, b: %(mab2)d, iarr: [0, 0, 0] }
            ],
            s: %(mids)d,
        },
    }
}
"""

# Distinct sentinel values so any aliasing/wrong-address bug produces
# mismatches rather than accidental agreement.
SENTINELS = dict(
    s0=101,
    ia=201, ib=202, iarr0=203, iarr1=204, iarr2=205,
    sarr0=301, sarr1=302, sarr2=303,
    aa0=401, ab0=402, aa1=411, ab1=412, aa2=421, ab2=422,
    a2i0=431, a2i1=432, a2i2=433,
    msarr0=501, msarr1=502, msarr2=503,
    maa0=601, mab0=602, maa1=611, mab1=612, maa2=621, mab2=622,
    mids=701,
)

BUMP_HELPERS = """\
fn bump_i64(e: &! i64) with Mut {
    *e = *e + 100
}
fn bump_inner(e: &! Inner) with Mut {
    e.a = e.a + 100
}
"""


class Chain:
    """One projection sequence off a root expression."""

    def __init__(self, cid, leaf, expr_fn, read_expr_fn, expected_fn, assign_value_fn):
        self.cid = cid
        self.leaf = leaf  # "scalar" | "aggregate"
        # expr_fn(root) -> place expression string (used as an lvalue)
        self.expr_fn = expr_fn
        # read_expr_fn(root) -> expression string used to read the leaf
        # scalar value for verification (== expr_fn for scalar leaves;
        # for aggregate leaves this projects to `.a` for a single i64).
        self.read_expr_fn = read_expr_fn
        # expected_fn() -> initial sentinel value of the scalar read above
        self.expected_fn = expected_fn
        # assign_value_fn() -> (assign_stmt_value_expr, expected_after_read)
        self.assign_value_fn = assign_value_fn


CHAINS = [
    Chain(
        "s", "scalar",
        lambda r: f"{r}.s",
        lambda r: f"{r}.s",
        lambda: SENTINELS["s0"],
        lambda: ("9001", 9001),
    ),
    Chain(
        "inner_scalar", "scalar",
        lambda r: f"{r}.inner.a",
        lambda r: f"{r}.inner.a",
        lambda: SENTINELS["ia"],
        lambda: ("9002", 9002),
    ),
    Chain(
        "sarr", "scalar",
        lambda r: f"{r}.sarr[{K}]",
        lambda r: f"{r}.sarr[{K}]",
        lambda: SENTINELS["sarr2"],
        lambda: ("9003", 9003),
    ),
    Chain(
        "aarr_elem", "aggregate",
        lambda r: f"{r}.aarr[{K}]",
        lambda r: f"{r}.aarr[{K}].a",
        lambda: SENTINELS["aa2"],
        lambda: (f"Inner {{ a: 9004, b: 9104, iarr: [0, 0, 0] }}", 9004),
    ),
    Chain(
        "aarr_field", "scalar",
        lambda r: f"{r}.aarr[{K}].a",
        lambda r: f"{r}.aarr[{K}].a",
        lambda: SENTINELS["aa2"],
        lambda: ("9005", 9005),
    ),
    Chain(
        "mid_sarr", "scalar",
        lambda r: f"{r}.mid.sarr[{J}]",
        lambda r: f"{r}.mid.sarr[{J}]",
        lambda: SENTINELS["msarr1"],
        lambda: ("9006", 9006),
    ),
    Chain(
        "mid_aarr_field", "scalar",
        lambda r: f"{r}.mid.aarr[{J}].a",
        lambda r: f"{r}.mid.aarr[{J}].a",
        lambda: SENTINELS["maa1"],
        lambda: ("9007", 9007),
    ),
    Chain(
        "aarr_iarr", "scalar",
        lambda r: f"{r}.aarr[{K}].iarr[{J}]",
        lambda r: f"{r}.aarr[{K}].iarr[{J}]",
        lambda: SENTINELS["a2i1"],
        lambda: ("9008", 9008),
    ),
]

ROOTS = ["local", "box", "sharedref", "mutref"]
OPS = ["read", "assign", "borrowmut"]


def struct_defs_block():
    return STRUCT_DEFS % SENTINELS


def skip_combo(root, op):
    # ASSIGN/BORROW_MUT through a shared (&Holder) ref root are
    # incoherent -- shared refs are read-only by construction.
    if root == "sharedref" and op != "read":
        return True
    return False


def name_for(root, chain, op):
    return f"shape_{root}_{chain.cid}_{op}"


def gen_op_body(root_expr, chain, op, indent="    "):
    """Return (setup_lines, verify_expr) for the op, operating on
    `root_expr` (already resolved to the in-scope root binding name)."""
    place = chain.expr_fn(root_expr)
    read_expr = chain.read_expr_fn(root_expr)
    lines = []
    if op == "read":
        expected = chain.expected_fn()
        lines.append(f"let got = {read_expr}")
        verify = f"got == {expected}"
    elif op == "assign":
        value_expr, expected = chain.assign_value_fn()
        lines.append(f"{place} = {value_expr}")
        lines.append(f"let got = {read_expr}")
        verify = f"got == {expected}"
    elif op == "borrowmut":
        if chain.leaf == "aggregate":
            base = chain.expected_fn()
            lines.append(f"bump_inner(&! {place})")
            lines.append(f"let got = {read_expr}")
            verify = f"got == {base + 100}"
        else:
            base = chain.expected_fn()
            lines.append(f"bump_i64(&! {place})")
            lines.append(f"let got = {read_expr}")
            verify = f"got == {base + 100}"
    else:
        raise ValueError(op)
    return lines, verify


def gen_source(root, chain, op, name):
    body_lines, verify = gen_op_body("h", chain, op)
    body = "\n".join(f"    {ln}" for ln in body_lines)

    header = (
        "//@ run-pass\n"
        f"// Generated lvalue-shape witness: root={root} chain={chain.cid} op={op}\n"
        f"// name={name}\n\n"
    )
    src = header + struct_defs_block() + "\n" + BUMP_HELPERS + "\n"

    if root in ("local", "box"):
        if root == "local":
            root_setup = "    var h = make_holder()"
        else:
            root_setup = "    var hb = Box::new(make_holder())"
            body_lines2, verify2 = gen_op_body("(*hb)", chain, op)
            body = "\n".join(f"    {ln}" for ln in body_lines2)
            verify = verify2

        src += (
            "fn main() -> i64 with IO, Mut, Panic, Div, Alloc {\n"
            f"{root_setup}\n"
            f"{body}\n"
            f"    if {verify} {{\n"
            f'        print("{name} PASS\\n")\n'
            "        0\n"
            "    } else {\n"
            f'        print("{name} FAIL\\n")\n'
            "        1\n"
            "    }\n"
            "}\n"
        )
    else:
        # sharedref / mutref: main builds the Holder and calls a helper
        # `op(h: &Holder)` / `op(h: &! Holder)` that performs the op and
        # verification internally, printing its own verdict.
        if root == "sharedref":
            sig = "fn op(h: &Holder) -> i64 {"
            call = "op(&h)"
        else:
            sig = "fn op(h: &! Holder) -> i64 with Mut {"
            call = "op(&! h)"

        src += (
            f"{sig}\n"
            f"{body}\n"
            f"    if {verify} {{\n"
            "        0\n"
            "    } else {\n"
            "        1\n"
            "    }\n"
            "}\n\n"
            "fn main() -> i64 with IO, Mut, Panic, Div, Alloc {\n"
            "    var h = make_holder()\n"
            f"    let r = {call}\n"
            "    if r == 0 {\n"
            f'        print("{name} PASS\\n")\n'
            "    } else {\n"
            f'        print("{name} FAIL\\n")\n'
            "    }\n"
            "    r\n"
            "}\n"
        )
    return src


def all_combos():
    combos = []
    for root in ROOTS:
        for chain in CHAINS:
            for op in OPS:
                if skip_combo(root, op):
                    continue
                combos.append((root, chain, op))
    return combos


def write_witnesses(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for root, chain, op in sorted(all_combos(), key=lambda c: name_for(c[0], c[1], c[2])):
        name = name_for(root, chain, op)
        src = gen_source(root, chain, op, name)
        path = os.path.join(out_dir, f"{name}.sio")
        with open(path, "w") as f:
            f.write(src)
        names.append(name)
    return names


def compile_and_run(name, out_dir, seed):
    src_path = os.path.join(out_dir, f"{name}.sio")
    elf_path = os.path.join(out_dir, f"{name}.elf")
    try:
        cp = subprocess.run(
            [seed, src_path, elf_path],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=COMPILE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1, "COMPILE_ERROR", ""

    compile_rc = cp.returncode
    if compile_rc != 0 or not os.path.exists(elf_path):
        return compile_rc, -1, "COMPILE_ERROR", ""

    os.chmod(elf_path, 0o755)
    try:
        rp = subprocess.run(
            [elf_path],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return compile_rc, -1, "TIMEOUT", ""

    run_rc = rp.returncode
    stdout_text = rp.stdout.decode("utf-8", errors="replace")
    first_line = stdout_text.splitlines()[0] if stdout_text.splitlines() else ""

    if run_rc < 0:
        verdict = "CRASH"
    elif run_rc > 128:
        verdict = "CRASH"
    elif run_rc == 0 and first_line.endswith("PASS"):
        verdict = "PASS"
    elif first_line.endswith("FAIL"):
        verdict = "FAIL"
    elif run_rc == 0:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return compile_rc, run_rc, verdict, first_line


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="lean_lvalue_shapes_")
    seed = os.environ.get(
        "SOUNIO_LVALUE_SEED",
        os.path.join(REPO_ROOT, "bin", "souc-lean-single-x86_64"),
    )
    baseline = os.environ.get("SOUNIO_SHAPE_MATRIX_BASELINE") == "1"

    names = write_witnesses(out_dir)

    rows = []
    for name in names:
        compile_rc, run_rc, verdict, first_line = compile_and_run(name, out_dir, seed)
        rows.append((name, compile_rc, run_rc, verdict, first_line))

    rows.sort(key=lambda r: r[0])

    for name, compile_rc, run_rc, verdict, first_line in rows:
        print(f"{name}\t{compile_rc}\t{run_rc}\t{verdict}\t{first_line}")

    green = sum(1 for r in rows if r[3] == "PASS")
    total = len(rows)
    print(f"SHAPES: {green}/{total} green")

    if baseline:
        sys.exit(0)
    sys.exit(0 if green == total else 1)


if __name__ == "__main__":
    main()
