#!/usr/bin/env python3
"""Generate every fdlibm sin/cos artefact from scripts/research/fdlibm_sin_cos_oracle.py.

  self-hosted/native/math_fdlibm_trig.sio          Madaros sin/cos builtins (the instruction list)
  stdlib/math/pure.sio                             block between the GENERATED markers
  self-hosted/compiler/lean_single.sio             block between the GENERATED markers (prelude)
  tests/run-pass/native_sin_cos_fdlibm_vectors.sio bare sin/cos: the builtin of the engine that runs it
  tests/stdlib/math/test_sin_cos_fdlibm.sio        math::pure sin/cos

Why one generator. The three implementations must return the same bits, and the
tests must assert those bits. Written by hand, a constant typed twice or a
reference value copied once is exactly the error a reference test certifies
instead of catching. Here the emitter comes from the oracle's instruction list,
the Sounio source block is one text used twice, and every expected value is the
oracle interpreter's output -- which the oracle diffs against OpenLibm.

Constants in the Sounio block are exact integer mantissas divided by exact powers
of two, never decimal literals: lean_single's literal reader is not correctly
rounded (#1626 follow-up) and would silently change the kernels.

Usage:  python3 scripts/dev/gen_fdlibm_trig.py [--check]
"""
import argparse
import importlib.util
import math
import os
import random
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORACLE = os.path.join(ROOT, "scripts", "research", "fdlibm_sin_cos_oracle.py")
EMITTER = os.path.join(ROOT, "self-hosted", "native", "math_fdlibm_trig.sio")
PURE = os.path.join(ROOT, "stdlib", "math", "pure.sio")
LEAN = os.path.join(ROOT, "self-hosted", "compiler", "lean_single.sio")
TEST_NATIVE = os.path.join(ROOT, "tests", "run-pass", "native_sin_cos_fdlibm_vectors.sio")
TEST_PURE = os.path.join(ROOT, "tests", "stdlib", "math", "test_sin_cos_fdlibm.sio")
BEGIN = "BEGIN GENERATED fdlibm sin/cos"
END = "END GENERATED fdlibm sin/cos"


def load_oracle():
    spec = importlib.util.spec_from_file_location("fdlibm_sin_cos_oracle", ORACLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fdlibm_sin_cos_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


# oracle constant name -> Sounio function suffix
CONST_NAMES = [
    ("S1", "S1"), ("S2", "S2"), ("S3", "S3"), ("S4", "S4"), ("S5", "S5"), ("S6", "S6"),
    ("C1", "C1"), ("C2", "C2"), ("C3", "C3"), ("C4", "C4"), ("C5", "C5"), ("C6", "C6"),
    ("INVPIO2", "invpio2"), ("PIO2_1", "pio2_1"), ("PIO2_1T", "pio2_1t"), ("PIO2_2", "pio2_2"),
    ("PIO2_2T", "pio2_2t"), ("PIO2_3", "pio2_3"), ("PIO2_3T", "pio2_3t"),
    ("TWO24", "two24"), ("TWON24", "twon24"), ("P52X15", "p52x15"),
]

ALGORITHM = r"""
// v * 2^k, exact whenever v and the result are finite and every intermediate
// stays normal -- true at every call site below (|k| <= 1001, |v| >= 2^-200).
fn fdlibm_scalbn(v: f64, k: i64) -> f64 with Mut, Div, Panic {
    var r = v
    var n = k
    while n > 60 {
        r = r * ((1 << 60) as f64)
        n = n - 60
    }
    while n < 0 - 60 {
        r = r / ((1 << 60) as f64)
        n = n + 60
    }
    if n >= 0 { return r * ((1 << n) as f64) }
    r / ((1 << (0 - n)) as f64)
}

// floor for 0 <= v < 2^63; truncation is floor there.
fn fdlibm_floor_nonneg(v: f64) -> f64 {
    (v as i64) as f64
}

// IEEE negation, including the sign of a zero. Written as a multiplication
// because `-v` is lowered as a subtraction by Madaros, where -(+0.0) is +0.0.
fn fdlibm_neg(v: f64) -> f64 {
    v * (0.0 - 1.0)
}

struct FdlibmRemPio2 { n: i64, y0: f64, y1: f64 }

// k_sin.c
fn fdlibm_k_sin(x: f64, y: f64, iy: i64) -> f64 with Mut, Div, Panic {
    let z = x * x
    let w = z * z
    let r = (fdlibm_c_S2() + z * (fdlibm_c_S3() + z * fdlibm_c_S4())) + (z * w) * (fdlibm_c_S5() + z * fdlibm_c_S6())
    let v = z * x
    if iy == 0 { return x + v * (fdlibm_c_S1() + z * r) }
    x - (((z * (0.5 * y - v * r)) - y) - v * fdlibm_c_S1())
}

// k_cos.c
fn fdlibm_k_cos(x: f64, y: f64) -> f64 with Mut, Div, Panic {
    let z = x * x
    let w = z * z
    let r = (z * (fdlibm_c_C1() + z * (fdlibm_c_C2() + z * fdlibm_c_C3()))) + (w * w) * (fdlibm_c_C4() + z * (fdlibm_c_C5() + z * fdlibm_c_C6()))
    let hz = 0.5 * z
    let w1 = 1.0 - hz
    w1 + (((1.0 - w1) - hz) + (z * r - x * y))
}

// k_rem_pio2.c with prec = 1 (jk = jp = 4). x0..x2 are the 24-bit chunks tx[].
fn fdlibm_k_rem_pio2(x0: f64, x1: f64, x2: f64, e0: i64, nx: i64) -> FdlibmRemPio2 with Mut, Div, Panic {
    var xa: [f64; 3] = [0.0; 3]
    xa[0] = x0
    xa[1] = x1
    xa[2] = x2
    var f: [f64; 20] = [0.0; 20]
    var q: [f64; 20] = [0.0; 20]
    var fq: [f64; 20] = [0.0; 20]
    var iq: [i64; 20] = [0; 20]
    let jk: i64 = 4
    let jp: i64 = 4
    let jx = nx - 1
    var jv = (e0 - 3) / 24
    if jv < 0 { jv = 0 }
    var q0 = e0 - 24 * (jv + 1)

    // f[0] .. f[jx+jk] = ipio2[jv-jx] .. ipio2[jv+jk]
    var j = jv - jx
    let m = jx + jk
    var i: i64 = 0
    while i <= m {
        if j < 0 { f[i] = 0.0 } else { f[i] = fdlibm_ipio2(j) as f64 }
        i = i + 1
        j = j + 1
    }

    // q[0] .. q[jk]
    var fw = 0.0
    i = 0
    while i <= jk {
        fw = 0.0
        j = 0
        while j <= jx {
            fw = fw + xa[j] * f[jx + i - j]
            j = j + 1
        }
        q[i] = fw
        i = i + 1
    }

    var jz = jk
    var z = 0.0
    var n: i64 = 0
    var ih: i64 = 0
    while true {
        // distill q[] into iq[] reversingly
        i = 0
        j = jz
        z = q[jz]
        while j > 0 {
            fw = ((fdlibm_c_twon24() * z) as i64) as f64
            iq[i] = (z - fdlibm_c_two24() * fw) as i64
            z = q[j - 1] + fw
            i = i + 1
            j = j - 1
        }

        // n
        z = fdlibm_scalbn(z, q0)
        z = z - 8.0 * fdlibm_floor_nonneg(z * 0.125)
        n = z as i64
        z = z - (n as f64)
        ih = 0
        if q0 > 0 {
            i = iq[jz - 1] >> (24 - q0)
            n = n + i
            iq[jz - 1] = iq[jz - 1] - (i << (24 - q0))
            ih = iq[jz - 1] >> (23 - q0)
        } else if q0 == 0 {
            ih = iq[jz - 1] >> 23
        } else if z >= 0.5 {
            ih = 2
        }

        if ih > 0 {
            n = n + 1
            var carry: i64 = 0
            i = 0
            while i < jz {
                j = iq[i]
                if carry == 0 {
                    if j != 0 {
                        carry = 1
                        iq[i] = 0x1000000 - j
                    }
                } else {
                    iq[i] = 0xffffff - j
                }
                i = i + 1
            }
            if q0 == 1 {
                iq[jz - 1] = iq[jz - 1] & 0x7fffff
            } else if q0 == 2 {
                iq[jz - 1] = iq[jz - 1] & 0x3fffff
            }
            if ih == 2 {
                z = 1.0 - z
                if carry != 0 { z = z - fdlibm_scalbn(1.0, q0) }
            }
        }

        // recompute when every fraction chunk vanished
        var again: i64 = 0
        if z == 0.0 {
            j = 0
            i = jz - 1
            while i >= jk {
                j = j | iq[i]
                i = i - 1
            }
            if j == 0 {
                var k: i64 = 1
                while iq[jk - k] == 0 { k = k + 1 }
                i = jz + 1
                while i <= jz + k {
                    f[jx + i] = fdlibm_ipio2(jv + i) as f64
                    fw = 0.0
                    j = 0
                    while j <= jx {
                        fw = fw + xa[j] * f[jx + i - j]
                        j = j + 1
                    }
                    q[i] = fw
                    i = i + 1
                }
                jz = jz + k
                again = 1
            }
        }
        if again == 0 { break }
    }

    // chop off zero terms
    if z == 0.0 {
        jz = jz - 1
        q0 = q0 - 24
        while iq[jz] == 0 {
            jz = jz - 1
            q0 = q0 - 24
        }
    } else {
        z = fdlibm_scalbn(z, 0 - q0)
        if z >= fdlibm_c_two24() {
            fw = ((fdlibm_c_twon24() * z) as i64) as f64
            iq[jz] = (z - fdlibm_c_two24() * fw) as i64
            jz = jz + 1
            q0 = q0 + 24
            iq[jz] = fw as i64
        } else {
            iq[jz] = z as i64
        }
    }

    // integer chunks back to floating point
    fw = fdlibm_scalbn(1.0, q0)
    i = jz
    while i >= 0 {
        q[i] = fw * (iq[i] as f64)
        fw = fw * fdlibm_c_twon24()
        i = i - 1
    }

    // PIo2[0..jp] * q[jz..0]
    i = jz
    while i >= 0 {
        fw = 0.0
        var kk: i64 = 0
        while kk <= jp && kk <= jz - i {
            fw = fw + fdlibm_pio2_tab(kk) * q[i + kk]
            kk = kk + 1
        }
        fq[jz - i] = fw
        i = i - 1
    }

    // compress fq[] into y[] (prec 1)
    fw = 0.0
    i = jz
    while i >= 0 {
        fw = fw + fq[i]
        i = i - 1
    }
    var y0 = fw
    if ih != 0 { y0 = fdlibm_neg(fw) }
    fw = fq[0] - fw
    i = 1
    while i <= jz {
        fw = fw + fq[i]
        i = i + 1
    }
    var y1 = fw
    if ih != 0 { y1 = fdlibm_neg(fw) }
    FdlibmRemPio2 { n: n & 7, y0: y0, y1: y1 }
}

// e_rem_pio2.c. The caller has already handled |x| <= pi/4.
fn fdlibm_rem_pio2(x: f64) -> FdlibmRemPio2 with Mut, Div, Panic {
    let hx = f64_to_bits(x) >> 32
    let ix = hx & 0x7fffffff
    var medium: i64 = 0
    if ix <= 0x400f6a7a {
        if (ix & 0xfffff) == 0x921fb {
            medium = 1
        } else if ix <= 0x4002d97c {
            if hx > 0 {
                let z = x - fdlibm_c_pio2_1()
                let y0 = z - fdlibm_c_pio2_1t()
                let y1 = (z - y0) - fdlibm_c_pio2_1t()
                return FdlibmRemPio2 { n: 1, y0: y0, y1: y1 }
            }
            let z = x + fdlibm_c_pio2_1()
            let y0 = z + fdlibm_c_pio2_1t()
            let y1 = (z - y0) + fdlibm_c_pio2_1t()
            return FdlibmRemPio2 { n: 0 - 1, y0: y0, y1: y1 }
        } else {
            if hx > 0 {
                let z = x - 2.0 * fdlibm_c_pio2_1()
                let y0 = z - 2.0 * fdlibm_c_pio2_1t()
                let y1 = (z - y0) - 2.0 * fdlibm_c_pio2_1t()
                return FdlibmRemPio2 { n: 2, y0: y0, y1: y1 }
            }
            let z = x + 2.0 * fdlibm_c_pio2_1()
            let y0 = z + 2.0 * fdlibm_c_pio2_1t()
            let y1 = (z - y0) + 2.0 * fdlibm_c_pio2_1t()
            return FdlibmRemPio2 { n: 0 - 2, y0: y0, y1: y1 }
        }
    }
    if medium == 0 && ix <= 0x401c463b {
        if ix <= 0x4015fdbc {
            if ix == 0x4012d97c {
                medium = 1
            } else {
                if hx > 0 {
                    let z = x - 3.0 * fdlibm_c_pio2_1()
                    let y0 = z - 3.0 * fdlibm_c_pio2_1t()
                    let y1 = (z - y0) - 3.0 * fdlibm_c_pio2_1t()
                    return FdlibmRemPio2 { n: 3, y0: y0, y1: y1 }
                }
                let z = x + 3.0 * fdlibm_c_pio2_1()
                let y0 = z + 3.0 * fdlibm_c_pio2_1t()
                let y1 = (z - y0) + 3.0 * fdlibm_c_pio2_1t()
                return FdlibmRemPio2 { n: 0 - 3, y0: y0, y1: y1 }
            }
        } else {
            if ix == 0x401921fb {
                medium = 1
            } else {
                if hx > 0 {
                    let z = x - 4.0 * fdlibm_c_pio2_1()
                    let y0 = z - 4.0 * fdlibm_c_pio2_1t()
                    let y1 = (z - y0) - 4.0 * fdlibm_c_pio2_1t()
                    return FdlibmRemPio2 { n: 4, y0: y0, y1: y1 }
                }
                let z = x + 4.0 * fdlibm_c_pio2_1()
                let y0 = z + 4.0 * fdlibm_c_pio2_1t()
                let y1 = (z - y0) + 4.0 * fdlibm_c_pio2_1t()
                return FdlibmRemPio2 { n: 0 - 4, y0: y0, y1: y1 }
            }
        }
    }
    if medium == 1 || ix < 0x413921fb {
        // |x| ~< 2^20*(pi/2): n = nearest(x*2/pi), then one to three rounds of pi/2
        var fnv = x * fdlibm_c_invpio2() + fdlibm_c_p52x15()
        fnv = fnv - fdlibm_c_p52x15()
        let n = fnv as i64
        var r = x - fnv * fdlibm_c_pio2_1()
        var w = fnv * fdlibm_c_pio2_1t()
        let j = ix >> 20
        var y0 = r - w
        var i = j - ((f64_to_bits(y0) >> 52) & 0x7ff)
        if i > 16 {
            var t = r
            w = fnv * fdlibm_c_pio2_2()
            r = t - w
            w = fnv * fdlibm_c_pio2_2t() - ((t - r) - w)
            y0 = r - w
            i = j - ((f64_to_bits(y0) >> 52) & 0x7ff)
            if i > 49 {
                t = r
                w = fnv * fdlibm_c_pio2_3()
                r = t - w
                w = fnv * fdlibm_c_pio2_3t() - ((t - r) - w)
                y0 = r - w
            }
        }
        let y1 = (r - y0) - w
        return FdlibmRemPio2 { n: n, y0: y0, y1: y1 }
    }
    if ix >= 0x7ff00000 {
        let d = x - x
        return FdlibmRemPio2 { n: 0, y0: d, y1: d }
    }
    // large |x|: z = |x| * 2^-e0 lies in [2^23, 2^24); split it into 24-bit chunks.
    // (C does this with INSERT_WORDS; lowering the exponent field by e0 is exactly
    // this multiplication by a power of two.)
    let e0 = (ix >> 20) - 1046
    var ax = x
    if x < 0.0 { ax = fdlibm_neg(x) }
    var z = fdlibm_scalbn(ax, 0 - e0)
    let tx0 = (z as i64) as f64
    z = (z - tx0) * fdlibm_c_two24()
    let tx1 = (z as i64) as f64
    z = (z - tx1) * fdlibm_c_two24()
    let tx2 = z
    var nx: i64 = 3
    if tx2 == 0.0 {
        nx = 2
        if tx1 == 0.0 { nx = 1 }
    }
    let kr = fdlibm_k_rem_pio2(tx0, tx1, tx2, e0, nx)
    if hx < 0 {
        return FdlibmRemPio2 { n: 0 - kr.n, y0: fdlibm_neg(kr.y0), y1: fdlibm_neg(kr.y1) }
    }
    kr
}

// s_sin.c
fn fdlibm_sin(x: f64) -> f64 with Mut, Div, Panic {
    let ix = (f64_to_bits(x) >> 32) & 0x7fffffff
    if ix <= 0x3fe921fb {
        if ix < 0x3e500000 { return x }
        return fdlibm_k_sin(x, 0.0, 0)
    }
    if ix >= 0x7ff00000 { return x - x }
    let rp = fdlibm_rem_pio2(x)
    let quad = rp.n & 3
    if quad == 0 { return fdlibm_k_sin(rp.y0, rp.y1, 1) }
    if quad == 1 { return fdlibm_k_cos(rp.y0, rp.y1) }
    if quad == 2 { return fdlibm_neg(fdlibm_k_sin(rp.y0, rp.y1, 1)) }
    fdlibm_neg(fdlibm_k_cos(rp.y0, rp.y1))
}

// s_cos.c
fn fdlibm_cos(x: f64) -> f64 with Mut, Div, Panic {
    let ix = (f64_to_bits(x) >> 32) & 0x7fffffff
    if ix <= 0x3fe921fb {
        if ix < 0x3e46a09e { return 1.0 }
        return fdlibm_k_cos(x, 0.0)
    }
    if ix >= 0x7ff00000 { return x - x }
    let rp = fdlibm_rem_pio2(x)
    let quad = rp.n & 3
    if quad == 0 { return fdlibm_k_cos(rp.y0, rp.y1) }
    if quad == 1 { return fdlibm_neg(fdlibm_k_sin(rp.y0, rp.y1, 1)) }
    if quad == 2 { return fdlibm_neg(fdlibm_k_cos(rp.y0, rp.y1)) }
    fdlibm_k_sin(rp.y0, rp.y1, 1)
}
"""


def d2b(v):
    import struct
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def exact_expr(v):
    """v as (M as f64) scaled by exact powers of two; no decimal float literal."""
    assert v != 0.0 and v == v and abs(v) != float("inf")
    u = d2b(abs(v))
    e_bits = (u >> 52) & 0x7FF
    assert e_bits != 0, "subnormal constant not expected"
    mant = (u & ((1 << 52) - 1)) | (1 << 52)
    e = e_bits - 1075
    while mant % 2 == 0:
        mant //= 2
        e += 1
    assert float(Fraction(mant) * Fraction(2) ** e) == abs(v)
    parts = ["(%d as f64)" % mant]
    k = e
    while k > 62:
        parts.append(" * ((1 << 62) as f64)")
        k -= 62
    while k < -62:
        parts.append(" / ((1 << 62) as f64)")
        k += 62
    if k > 0:
        parts.append(" * ((1 << %d) as f64)" % k)
    elif k < 0:
        parts.append(" / ((1 << %d) as f64)" % -k)
    expr = "".join(parts)
    return "0.0 - (%s)" % expr if v < 0 else expr


def sounio_block(o, K):
    kd = dict(K)
    L = ["// constants: exact integer mantissa over exact powers of two (see the note above)"]
    for oname, sname in CONST_NAMES:
        L.append("fn fdlibm_c_%s() -> f64 { %s }" % (sname, exact_expr(kd[oname])))
    L.append("")
    L.append("// k_rem_pio2.c ipio2[]: 2/pi in 24-bit chunks")
    L.append("fn fdlibm_ipio2(i: i64) -> i64 with Panic {")
    L.append("    let t: [i64; 66] = [" + ", ".join("0x%06X" % v for v in o.IPIO2) + "]")
    L.append("    t[i]")
    L.append("}")
    L.append("")
    L.append("// k_rem_pio2.c PIo2[]: pi/2 in 24-bit pieces")
    L.append("fn fdlibm_pio2_tab(k: i64) -> f64 with Mut, Div, Panic {")
    for i in range(8):
        L.append("    if k == %d { return %s }" % (i, exact_expr(kd["PIO2TAB%d" % i])))
    L.append("    0.0")
    L.append("}")
    L.extend(ALGORITHM.rstrip("\n").split("\n"))
    return L


def prelude_block(o, K):
    import re
    src = sounio_block(o, K) + [
        "fn __native_sin_f64(x: f64) -> f64 with Mut, Div, Panic {",
        "    fdlibm_sin(x)",
        "}",
        "fn __native_cos_f64(x: f64) -> f64 with Mut, Div, Panic {",
        "    fdlibm_cos(x)",
        "}",
    ]
    out = []
    for line in src:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        line = re.sub(r"\bfdlibm_", "__native_fdlibm_", line)
        line = line.replace("FdlibmRemPio2", "NativeFdlibmRemPio2")
        assert '"' not in line and "\\" not in line, line
        # Long source lines (the 66-entry ipio2 literal is ~600 characters) are split
        # across several append_src_lit calls; only the last carries the newline, so
        # the prelude text is unchanged. No existing prelude literal exceeds ~130.
        width = 96
        pieces = [line[i:i + width] for i in range(0, len(line), width)]
        for k, piece in enumerate(pieces):
            nl = "\\n" if k == len(pieces) - 1 else ""
            out.append('    append_src_lit("%s%s")' % (piece, nl))
    return out


def replace_between(text, body):
    lines = text.split("\n")
    b = [i for i, l in enumerate(lines) if BEGIN in l]
    e = [i for i, l in enumerate(lines) if END in l]
    assert len(b) == 1 and len(e) == 1 and b[0] < e[0], "markers missing or duplicated"
    return "\n".join(lines[:b[0] + 1] + body + lines[e[0]:])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def fixture_inputs(o):
    xs = list(o.SPECIALS) + [0xFFF8000000000001]
    for hw in o.BRANCH_HIGH_WORDS:
        for sign in (0, 1 << 63):
            for u in (sign | (hw << 32), sign | (hw << 32) | 0xFFFFFFFF, sign | ((hw << 32) - 1)):
                xs.append(u)
    for k in range(1, 25):
        u = d2b(k * math.pi / 2)
        xs += [u, (1 << 63) | u, u + 1]
    for i, (u, _) in enumerate(o.WORST_CASES[:48]):
        xs.append(u)
        if i < 16:
            xs.append((1 << 63) | u)
    rng = random.Random(20260914)
    for _ in range(100):
        band = rng.random()
        if band < 0.3:
            x = rng.uniform(-8.0, 8.0)
        elif band < 0.6:
            x = rng.uniform(-2e6, 2e6)
        else:
            x = rng.choice((-1, 1)) * 2.0 ** rng.uniform(20, 1023)
        xs.append(d2b(x))
    seen = set()
    return [u & ((1 << 64) - 1) for u in xs if not (u in seen or seen.add(u))]


def i64lit(u):
    s = u - (1 << 64) if u >= (1 << 63) else u
    if s == -(1 << 63):
        return "((0 - 9223372036854775807) - 1)"
    return str(s) if s >= 0 else "(0 - %d)" % -s


def is_nan_u(u):
    return ((u >> 52) & 0x7FF) == 0x7FF and (u & ((1 << 52) - 1)) != 0


def x_expr(u):
    neg = 1 if (u >> 63) else 0
    e_bits = (u >> 52) & 0x7FF
    frac = u & ((1 << 52) - 1)
    if e_bits == 0x7FF:
        inf = "tv_scalbn(1.0, 1024)"
        if frac:
            return "(%s - %s)" % (inf, inf)
        return "(%s * (0.0 - 1.0))" % inf if neg else inf
    if e_bits == 0:
        mant, e = frac, -1074
    else:
        mant, e = frac | (1 << 52), e_bits - 1075
    return "tv_mk(%d, %d, %d)" % (mant, e, neg)


FIXTURE_HELPERS = """
fn tv_scalbn(v: f64, k: i64) -> f64 with Mut, Div, Panic {
    var r = v
    var n = k
    while n > 60 {
        r = r * ((1 << 60) as f64)
        n = n - 60
    }
    while n < 0 - 60 {
        r = r / ((1 << 60) as f64)
        n = n + 60
    }
    if n >= 0 { return r * ((1 << n) as f64) }
    r / ((1 << (0 - n)) as f64)
}

// mant * 2^e, built exactly; negation by multiplication keeps the sign of zero.
fn tv_mk(mant: i64, e: i64, neg: i64) -> f64 with Mut, Div, Panic {
    let v = tv_scalbn(mant as f64, e)
    if neg != 0 { return v * (0.0 - 1.0) }
    v
}

fn tv_is_nan_bits(b: i64) -> bool {
    let ex = (b >> 52) & 2047
    let m = b & 4503599627370495
    ex == 2047 && m != 0
}

// One row: the input must be the intended double, then sin and cos must be the
// oracle's bit patterns. A NaN result is asserted as NaN: IEEE 754 leaves a
// generated NaN's sign and payload unspecified.
fn tv_row(row: i64, x: f64, xb: i64, sb: i64, cb: i64, snan: i64, cnan: i64) -> i64 with IO, Mut, Div, Panic {
    if f64_to_bits(x) != xb { print("ROW "); print(row); print(" input\\n"); return 1 }
    var bad: i64 = 0
    let s = f64_to_bits(sin(x))
    let c = f64_to_bits(cos(x))
    if snan != 0 {
        if !tv_is_nan_bits(s) { print("ROW "); print(row); print(" sin-nan\\n"); bad = bad + 1 }
    } else if s != sb { print("ROW "); print(row); print(" sin\\n"); bad = bad + 1 }
    if cnan != 0 {
        if !tv_is_nan_bits(c) { print("ROW "); print(row); print(" cos-nan\\n"); bad = bad + 1 }
    } else if c != cb { print("ROW "); print(row); print(" cos\\n"); bad = bad + 1 }
    bad
}
"""


def render_fixture(o, K, index, prog, header, sentinel):
    rows = []
    for u in fixture_inputs(o):
        sb, _ = o.run(prog, K, index, "sin", u)
        cb, _ = o.run(prog, K, index, "cos", u)
        rows.append((u, sb, cb))
    L = list(header)
    L.append(FIXTURE_HELPERS)
    chunk = 60
    nchunks = 0
    for ci in range(0, len(rows), chunk):
        L.append("fn tv_rows_%d() -> i64 with IO, Mut, Div, Panic {" % nchunks)
        L.append("    var bad: i64 = 0")
        for ri, (xb, sb, cb) in enumerate(rows[ci:ci + chunk], start=ci):
            if is_nan_u(xb):
                L.append('    if !tv_is_nan_bits(f64_to_bits(sin(%s))) { print("ROW %d nan-in\\n"); bad = bad + 1 }'
                         % (x_expr(xb), ri))
                continue
            L.append("    bad = bad + tv_row(%d, %s, %s, %s, %s, %d, %d)" % (
                ri, x_expr(xb), i64lit(xb), i64lit(sb), i64lit(cb), int(is_nan_u(sb)), int(is_nan_u(cb))))
        L.append("    bad")
        L.append("}")
        L.append("")
        nchunks += 1
    L.append("fn main() -> i32 with IO, Mut, Div, Panic {")
    L.append("    var bad: i64 = 0")
    for k in range(nchunks):
        L.append("    bad = bad + tv_rows_%d()" % k)
    L.append('    if bad == 0 { print("%s rows=%d\\n"); return 0 }' % (sentinel, len(rows)))
    L.append('    print("FAILED rows: "); print(bad); print("\\n")')
    L.append("    1")
    L.append("}")
    return "\n".join(L) + "\n"


NATIVE_HEADER = [
    "//@ run-pass",
    "//@ expect-stdout-contains: NATIVE_SIN_COS_FDLIBM_OK",
    "// GENERATED by scripts/dev/gen_fdlibm_trig.py from scripts/research/fdlibm_sin_cos_oracle.py.",
    "// Do not hand-edit: regenerate.",
    "//",
    "// sin and cos with no definition in scope, so each engine answers with its own builtin:",
    "// Madaros with ids 18/19 (native::math_fdlibm_trig), lean_single with the",
    "// __native_sin_f64 / __native_cos_f64 prelude. Both are fdlibm and must return the",
    "// oracle's bit patterns exactly -- the oracle is diffed against OpenLibm, so this is",
    "// bit-identity with OpenLibm, not a tolerance. Rows: special values, every high-word",
    "// branch threshold of s_sin/s_cos/e_rem_pio2, multiples of pi/2, the worst cases for",
    "// reduction mod pi/2 (down to 2^-60.89), and seeded random doubles.",
    "//",
    "// Inputs are built from mantissa and exponent, never from decimal literals, and each",
    "// row first checks that the input is the intended double.",
]

PURE_HEADER = [
    "//@ run-pass",
    "//@ expect-stdout-contains: PURE_SIN_COS_FDLIBM_OK",
    "// GENERATED by scripts/dev/gen_fdlibm_trig.py from scripts/research/fdlibm_sin_cos_oracle.py.",
    "// Do not hand-edit: regenerate.",
    "//",
    "// math::pure's sin and cos -- the Sounio fdlibm source -- against the same bit patterns",
    "// as tests/run-pass/native_sin_cos_fdlibm_vectors.sio, under whichever engine runs it.",
    "",
    "use math::pure::*",
]


def build_outputs():
    o = load_oracle()
    K, index = o.build_constants(o.IPIO2)
    prog = o.build_program()
    outputs = {}
    outputs[EMITTER] = o.render_sio(prog, K, index)
    outputs[PURE] = replace_between(open(PURE).read(), sounio_block(o, K))
    outputs[LEAN] = replace_between(open(LEAN).read(), prelude_block(o, K))
    outputs[TEST_NATIVE] = render_fixture(o, K, index, prog, NATIVE_HEADER, "NATIVE_SIN_COS_FDLIBM_OK")
    outputs[TEST_PURE] = render_fixture(o, K, index, prog, PURE_HEADER, "PURE_SIN_COS_FDLIBM_OK")
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any generated file on disk differs from what would be written")
    args = ap.parse_args()
    outputs = build_outputs()
    stale = 0
    for path, text in outputs.items():
        rel = os.path.relpath(path, ROOT)
        have = open(path).read() if os.path.exists(path) else ""
        if args.check:
            if have != text:
                print("STALE: %s" % rel)
                stale += 1
            else:
                print("fresh: %s" % rel)
        elif have != text:
            with open(path, "w") as fh:
                fh.write(text)
            print("wrote %s" % rel)
        else:
            print("unchanged %s" % rel)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
