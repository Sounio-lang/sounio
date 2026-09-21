#!/usr/bin/env python3
"""Generate self-hosted/native/math_ffi_emitters.sio from the oracle.

The two kernels in self-hosted/native/math_ffi.sio (mf_pow, mf_tgamma) are a
transliteration of scripts/research/pow_tgamma_oracle.py.  They take every
constant they need as a .rodata BYTE OFFSET, so somebody has to register those
constants.  This script is that somebody.

Why generated and not hand-written:

  Every constant crosses into the Sounio as a 64-bit hex literal.  A hand-typed
  bit pattern is a transcription error in the one place where a reference test
  would CERTIFY the error rather than catch it -- the oracle and the emitter
  would be measured against each other while both carry the same typo.  Here
  the hex comes out of the oracle's own K dict via its own d2b, so the emitter
  cannot disagree with the model it was validated against.

  The oracle derives K from exact Fractions and 60-digit decimals, rounds once,
  and asserts the result against the design's published hex.  Importing it is
  therefore importing the audit, not just the numbers.

CONTIGUITY IS A CONTRACT.  mf_log2_dd reads atanh coefficient i from
atanh_off + 8*i, mf_exp2_core_dd reads expe_off + 8*i, mf_gamma_core reads
lz_off + 8*k and lzk_off + 8*k, mf_sinpi reads sinq_off + 8*i and
cosq_off + 8*i, mf_tgamma reads fact_off + 8*(i-1), and every dd pair is read
as (off, off+8).  The kernels do NOT get a per-element offset, so the generator
emits each table as consecutive data_section_add_f64 calls and captures ONLY
the first offset.  data_section_add_f64 aligns to 8 and appends 8 bytes, so
consecutive calls are exactly 8 apart -- that is what makes this legal, and it
is why nothing may be registered in between.

Usage:  python3 scripts/dev/gen_math_ffi_emitters.py [--check]
        --check exits 1 if the file on disk differs from what would be written.
"""

import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORACLE = os.path.join(ROOT, "scripts", "research", "pow_tgamma_oracle.py")
OUT = os.path.join(ROOT, "self-hosted", "native", "math_ffi_emitters.sio")


def load_oracle():
    spec = importlib.util.spec_from_file_location("pow_tgamma_oracle", ORACLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pow_tgamma_oracle"] = mod
    spec.loader.exec_module(mod)          # __main__-guarded: no measurement runs
    return mod


# --------------------------------------------------------------------------
# The rodata each kernel wants, IN THE ORDER ITS SIGNATURE NAMES IT.
#
#   ("s",  sounio_name, oracle_key)                 one double
#   ("dd", sounio_name, hi_key, lo_key)             two doubles, hi then lo
#   ("t",  sounio_name, [oracle_key, ...], note)    a contiguous table
#
# The sounio_name is what `<name>_off` is called in the generated body and is
# exactly the parameter name in math_ffi.sio, so the call site reads as a
# positional echo of the signature.
# --------------------------------------------------------------------------

def common_prefix():
    """The twelve mf_log2_dd / mf_exp2_core_dd want, passed straight through."""
    return [
        ("s", "split", "SPLIT", "SPLIT = 2^27+1"),
        ("s", "one", "ONE", "1.0"),
        ("s", "zero", "ZERO", "0.0"),
        ("s", "half", "HALF", "0.5"),
        ("s", "two", "TWO", "2.0"),
        ("s", "sqrt2", "SQRT2", "fl(sqrt 2)"),
        ("s", "p54", "P54", "2^54, denormal pre-scale"),
        ("t", "atanh", ["ATANH%d" % i for i in range(12)],
         "C2 coefficients 1/(2j+3), j = 1..12"),
        ("dd", "third", "THIRD_HI", "THIRD_LO", "1/3"),
        ("dd", "tl2e", "TL2E_HI", "TL2E_LO", "2*log2(e)"),
        ("dd", "ln2", "LN2_HI", "LN2_LO", "ln 2"),
        ("t", "expe", ["EXPE%d" % i for i in range(12)],
         "E coefficients 1/(j+2)!, j = 0..11"),
    ]


# pow does NOT register a NaN constant. Row 20 produces its NaN by dividing
# zero by zero, which raises the invalid flag C99 requires and returns the same
# real-indefinite pattern glibc's pow(-2.0, 0.5) returns; a constant would raise
# nothing and carry the other sign. Registering one anyway would leave a value
# in .rodata that nothing reads.
POW_RODATA = common_prefix() + [
    ("s", "inf", "INF", "+inf"),
    ("s", "ninf", "NINF", "-inf"),
    ("s", "nzero", "NZERO", "-0.0"),
    ("s", "rovf", "R_OVF", "1024.0, r_hi above this overflows"),
    ("s", "runf", "R_UNF", "-1080.0, below this the result is zero"),
    ("s", "ybig", "Y_BIG", "2^63, finding F5's guard"),
    ("s", "twom54", "TWOM54", "2^-54, mf_scale_2n's subnormal tail"),
]

TGAMMA_RODATA = common_prefix() + [
    ("s", "sqrt2pi", "SQRT2PI", "sqrt(2*pi), NOT sqrt(fl(2*pi))"),
    ("t", "lz", ["LZ%d" % k for k in range(14)],
     "Lanczos c_k, k = 0..13"),
    ("t", "lzk", ["LZK%d" % k for k in range(14)],
     "the integers 0.0 .. 13.0"),
    ("s", "c3375", "C3375", "3.375 = 27/8 = g - 1/2"),
    ("s", "log2e", "LOG2E", "log2(e), for the w_lo correction"),
    ("dd", "log2e_dd", "LOG2E_HI", "LOG2E_LO", "log2(e)"),
    ("dd", "pi", "PI_HI", "PI_LO", "pi"),
    ("t", "sinq", ["SINQ%d" % j for j in range(1, 10)],
     "Q coefficients (-1)^j/(2j+1)!, j = 1..9"),
    ("t", "cosq", ["COSQ%d" % j for j in range(1, 10)],
     "C coefficients (-1)^j/(2j)!, j = 1..9"),
    ("s", "inf", "INF", "+inf"),
    ("s", "ninf", "NINF", "-inf"),
    ("s", "nan", "NANV", "a quiet NaN"),
    ("s", "nzero", "NZERO", "-0.0"),
    ("s", "twom54", "TWOM54", "2^-54, mf_scale_2n's subnormal tail"),
    ("s", "gamovf", "GAMMA_OVF", "largest x with a finite Gamma"),
    ("s", "neg185", "NEG185", "-185.0, finding F9's cut-off"),
    ("s", "neghalf", "NEGHALF", "-0.5"),
    ("s", "twenty3", "TWENTY3", "23.0"),
    ("t", "fact", ["FACT%d" % i for i in range(23)],
     "Gamma(i) = (i-1)! for i = 1..23"),
]


def names_of(spec):
    return [item[1] for item in spec]


def hexlit(d2b, v):
    return "0x%016X" % d2b(v)


def emit_rodata(lines, spec, K, d2b):
    for item in spec:
        kind = item[0]
        if kind == "s":
            _, name, key, note = item
            lines.append("    c.rodata = data_section_add_f64(c.rodata, %s)"
                         % hexlit(d2b, K[key]))
            lines.append("    let %s_off = c.rodata.last_offset          // %s"
                         % (name, note))
        elif kind == "dd":
            _, name, hi, lo, note = item
            lines.append("    // %s as a dd: hi at %s_off, lo at %s_off + 8. CONTIGUOUS."
                         % (note, name, name))
            lines.append("    c.rodata = data_section_add_f64(c.rodata, %s)   // hi"
                         % hexlit(d2b, K[hi]))
            lines.append("    let %s_off = c.rodata.last_offset" % name)
            lines.append("    c.rodata = data_section_add_f64(c.rodata, %s)   // lo"
                         % hexlit(d2b, K[lo]))
        elif kind == "t":
            _, name, keys, note = item
            lines.append("    // %s: %d doubles, %s_off + 8*i. CONTIGUOUS -- only the"
                         % (note, len(keys), name))
            lines.append("    // first offset is captured; the kernel derives the rest.")
            for i, key in enumerate(keys):
                lines.append("    c.rodata = data_section_add_f64(c.rodata, %s)   // [%d]"
                             % (hexlit(d2b, K[key]), i))
                if i == 0:
                    lines.append("    let %s_off = c.rodata.last_offset" % name)
        else:
            raise AssertionError(kind)
        lines.append("")


def wrap_args(prefix, args, indent, width=96):
    """Lay out a call so no line runs past `width`."""
    out = []
    cur = prefix
    for i, a in enumerate(args):
        piece = a + ("," if i + 1 < len(args) else ")")
        if len(cur) + 1 + len(piece) > width and cur.strip() != prefix.strip():
            out.append(cur)
            cur = indent + piece
        else:
            cur = cur + (" " if cur.endswith(",") else "") + piece
    out.append(cur)
    return out


def emit_emitter(K, d2b, *, fname, kernel, spec, nargs, ntemps, frame,
                 header):
    """One `pub fn emit_builtin_<fname>` body.

    nargs   1 (tgamma) or 2 (pow) incoming f64 arguments
    ntemps  slots the kernel's temp block needs, contiguous below tb
    frame   bytes subtracted from rsp
    """
    # Slot layout, all rbp-relative and negative.  Argument slots first, then
    # the result slot, then the temp block.
    slots = {}
    nxt = 8
    slots["xs"] = nxt; nxt += 8
    if nargs == 2:
        slots["ys"] = nxt; nxt += 8
    slots["out"] = nxt; nxt += 8
    tb = nxt
    deepest = tb + 8 * (ntemps - 1)
    assert frame >= deepest, (frame, deepest)

    L = []
    L.extend(header)
    L.append("//")
    L.append("// Frame, all rbp-relative and negative:")
    L.append("//   [rbp-%d]  x, as bits then as a double" % slots["xs"])
    if nargs == 2:
        L.append("//   [rbp-%d] y" % slots["ys"])
    L.append("//   [rbp-%d] the result" % slots["out"])
    L.append("//   [rbp-%d] tb, the base of %s's %d-slot temp block, which runs DOWN"
             % (tb, kernel, ntemps))
    L.append("//            to [rbp-%d]. sub rsp,%d reserves %d bytes past that."
             % (deepest, frame, frame - deepest))
    L.append("//")
    L.append("// Calling shape: the f64 argument arrives in rdi as raw BITS, not in")
    if nargs == 2:
        L.append("// xmm0 -- and the second in rsi -- which is what every scalar builtin in")
        L.append("// this backend does (see emit_builtin_ffi_sqrt). The result goes back in")
    else:
        L.append("// xmm0, which is what every scalar builtin in this backend does (see")
        L.append("// emit_builtin_ffi_sqrt). The result goes back in")
    L.append("// rax the same way, via movq.")
    L.append("//")
    L.append("// GPRs the kernel clobbers: rax, rbx, rcx, rdx, rsi. rbx is not preserved,")
    L.append("// matching emit_builtin_exp, which clobbers it the same way.")
    L.append("pub fn emit_builtin_%s(nc: NativeCompiler) -> NativeCompiler with Mut, Panic, Div {"
             % fname)
    L.append("    var c = nc")
    L.append("")
    L.append("    // ---- .rodata ------------------------------------------------------")
    emit_rodata(L, spec, K, d2b)
    L.append("    // ---- prologue -----------------------------------------------------")
    L.append("    c.code = emit_push_rbp(c.code)")
    L.append("    c.code = emit_mov_rbp_rsp(c.code)")
    L.append("    c.code = emit_sub_rsp_imm32(c.code, %d)" % frame)
    L.append("")
    L.append("    c.code = emit_mov_reg_reg(c.code, 0, 7)              // rax = rdi = bits(x)")
    L.append("    c.code = emit_store_rax_rbp_disp32(c.code, 0 - %d)" % slots["xs"])
    if nargs == 2:
        L.append("    c.code = emit_mov_reg_reg(c.code, 0, 6)              // rax = rsi = bits(y)")
        L.append("    c.code = emit_store_rax_rbp_disp32(c.code, 0 - %d)" % slots["ys"])
    L.append("")
    L.append("    // ---- the kernel ---------------------------------------------------")
    args = ["c", "0 - %d" % slots["out"], "0 - %d" % slots["xs"]]
    if nargs == 2:
        args.append("0 - %d" % slots["ys"])
    args.extend("%s_off" % n for n in names_of(spec))
    args.append("0 - %d" % tb)
    L.extend(wrap_args("    c = %s(" % kernel, args,
                       " " * (len("    c = %s(" % kernel))))
    L.append("")
    L.append("    // ---- epilogue -----------------------------------------------------")
    L.append("    c.code = emit_movsd_xmm0_rbp_disp32(c.code, 0 - %d)" % slots["out"])
    L.append("    c.code = emit_movq_rax_xmm0(c.code)                  // movq rax, xmm0")
    L.append("    c.code = emit_add_rsp_imm32(c.code, %d)" % frame)
    L.append("    c.code = emit_pop_rbp(c.code)")
    L.append("    c.code = emit_ret(c.code)")
    L.append("    c")
    L.append("}")
    return L


PREAMBLE = """\
// self-hosted::native::math_ffi_emitters -- GENERATED, DO NOT EDIT.
//
// Regenerate with:  python3 scripts/dev/gen_math_ffi_emitters.py
// Source of truth:  scripts/research/pow_tgamma_oracle.py (dict K)
//
// The two kernels in native::math_ffi are transliterations of that oracle and
// take every constant as a .rodata byte offset. This file registers those
// constants and wraps each kernel in the scalar-builtin calling shape, so
// ffi_pow and ffi_tgamma become reachable builtin ids.
//
// Every hex literal below came out of the oracle's own K dict through its own
// d2b. None was typed. A hand-copied bit pattern is a transcription error in
// the one place a reference test would CERTIFY it rather than catch it: the
// oracle and the emitter would then be measured against each other while both
// carried the same typo.
//
// CONTIGUITY. data_section_add_f64 aligns to 8 and appends 8 bytes, so
// consecutive calls land exactly 8 apart. Every table below (atanh 12, expe 12,
// lz 14, lzk 14, sinq 9, cosq 9, fact 23) and every dd pair (third, tl2e, ln2,
// pi, log2e_dd) is registered as consecutive calls and only its FIRST offset is
// captured -- the kernels read element i at off + 8*i and are given no
// per-element offset. Registering anything between the elements of a table
// breaks it silently: the reads still assemble, link and run, and return
// whatever now sits at off + 8*i.

use native::encode::*
use native::elf::{data_section_add_f64}
use native::codegen_x86_linux::{NativeCompiler}
use native::math_ffi::{mf_pow, mf_tgamma}
"""

POW_HEADER = [
    "// pow(x, y) -- the ffi_pow builtin.",
    "//",
    "// Oracle-measured 0.76 ulp. This is an APPROXIMATION, unlike ffi_sqrt /",
    "// ffi_floor / ffi_ceil, which are single correctly-rounded instructions; it",
    "// carries an error budget and belongs behind its own reference vectors.",
]

TGAMMA_HEADER = [
    "// tgamma(x) -- the ffi_tgamma builtin.",
    "//",
    "// Oracle-measured 3.64 ulp against mpmath at 60 digits. Note finding F8: glibc's",
    "// own tgamma measures up to 5.16 ulp on the same bands and is 1 ulp off on the",
    "// exactly-representable factorials, so libm is NOT a valid reference for this.",
]


def build():
    oracle = load_oracle()
    K, d2b = oracle.K, oracle.d2b

    # Flatten the list-valued table entries into the K names the specs use.
    # ATANH%d / EXPE%d / SINQ%d / COSQ%d / LZ%d / LZK%d already exist; FACT does
    # not, so index it here rather than teaching the emitter about lists.
    for i, v in enumerate(K["FACT"]):
        K["FACT%d" % i] = v

    lines = [PREAMBLE]
    lines.append("")
    lines.extend(emit_emitter(
        K, d2b,
        fname="ffi_pow", kernel="mf_pow", spec=POW_RODATA,
        nargs=2, ntemps=72, frame=1024, header=POW_HEADER))
    lines.append("")
    lines.extend(emit_emitter(
        K, d2b,
        fname="ffi_tgamma", kernel="mf_tgamma", spec=TGAMMA_RODATA,
        nargs=1, ntemps=100, frame=1280, header=TGAMMA_HEADER))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file on disk is not what would be written")
    args = ap.parse_args()
    text = build()
    if args.check:
        have = open(OUT).read() if os.path.exists(OUT) else ""
        if have != text:
            print("STALE: %s differs from the generator's output" % OUT)
            return 1
        print("fresh: %s matches the generator" % OUT)
        return 0
    with open(OUT, "w") as fh:
        fh.write(text)
    print("wrote %s (%d lines)" % (OUT, text.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
