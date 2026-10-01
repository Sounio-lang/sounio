#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fdlibm_sin_cos_oracle.py -- the native sin/cos builtins as ONE instruction list.

Companion to docs/audit/NATIVE_SIN_COS_FDLIBM_DISPATCH_2026-09-14.md.

The Madaros backend has no libm: sin and cos are builtins whose bodies are
x86-64 instructions. This file holds those bodies once, as a list drawn from
the emitter's own primitive set -- two xmm registers, rbp-relative 8-byte
slots, rax/rbx/rcx/rdx, rip-relative constants and [rdx+rbx*8] loads for the
k_rem_pio2 arrays -- and everything else is derived from that list:

  run()            a Python interpreter, instruction by instruction
  render_c()       the same list as C; --against-openlibm compiles it next to
                   OpenLibm and diffs the two bit for bit over ~1M inputs
  render_sio()     the Sounio emitter, self-hosted/native/math_fdlibm_trig.sio,
                   written by scripts/dev/gen_fdlibm_trig.py

The algorithm is not designed here. It is OpenLibm's s_sin.c, s_cos.c,
e_rem_pio2.c, k_rem_pio2.c (prec 1), k_sin.c and k_cos.c, kept operation for
operation, so the property to check is bit-identity with that reference, not an
ulp bound. Measured: 0 differences over 1,009,578 inputs against OpenLibm
5fe3997 built with gcc 13.3 -O2 -ffp-contract=off.

usage:
  python3 scripts/research/fdlibm_sin_cos_oracle.py                   # counts
  python3 scripts/research/fdlibm_sin_cos_oracle.py --emit-c out.c
  python3 scripts/research/fdlibm_sin_cos_oracle.py --against-openlibm /path/to/openlibm
  python3 scripts/research/fdlibm_sin_cos_oracle.py --search-worst-cases 64   # needs mpmath
"""
import math
import os
import random
import re
import struct
import subprocess
import sys

MASK = (1 << 64) - 1


def d2b(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def b2d(u):
    return struct.unpack("<d", struct.pack("<Q", u & MASK))[0]


def s64(u):
    u &= MASK
    return u - (1 << 64) if u >> 63 else u


# ---------------------------------------------------------------------------
# constants: one contiguous rodata table, element i at k + 8*i
# ---------------------------------------------------------------------------
FDLIBM_DEC = [
    # k_sin.c
    ("S1", "-1.66666666666666324348e-01", 0xBFC5555555555549),
    ("S2", "8.33333333332248946124e-03", 0x3F8111111110F8A6),
    ("S3", "-1.98412698298579493134e-04", 0xBF2A01A019C161D5),
    ("S4", "2.75573137070700676789e-06", 0x3EC71DE357B1FE7D),
    ("S5", "-2.50507602534068634195e-08", 0xBE5AE5E68A2B9CEB),
    ("S6", "1.58969099521155010221e-10", 0x3DE5D93A5ACFD57C),
    # k_cos.c
    ("C1", "4.16666666666666019037e-02", 0x3FA555555555554C),
    ("C2", "-1.38888888888741095749e-03", 0xBF56C16C16C15177),
    ("C3", "2.48015872894767294178e-05", 0x3EFA01A019CB1590),
    ("C4", "-2.75573143513906633035e-07", 0xBE927E4F809C52AD),
    ("C5", "2.08757232129817482790e-09", 0x3E21EE9EBDB4B1C4),
    ("C6", "-1.13596475577881948265e-11", 0xBDA8FAE9BE8838D4),
    # e_rem_pio2.c
    ("INVPIO2", "6.36619772367581382433e-01", 0x3FE45F306DC9C883),
    ("PIO2_1", "1.57079632673412561417e+00", 0x3FF921FB54400000),
    ("PIO2_1T", "6.07710050650619224932e-11", 0x3DD0B4611A626331),
    ("PIO2_2", "6.07710050630396597660e-11", 0x3DD0B4611A600000),
    ("PIO2_2T", "2.02226624879595063154e-21", 0x3BA3198A2E037073),
    ("PIO2_3", "2.02226624871116645580e-21", 0x3BA3198A2E000000),
    ("PIO2_3T", "8.47842766036889956997e-32", 0x397B839A252049C1),
]
PIO2_TABLE_DEC = [
    "1.57079625129699707031e+00", "7.54978941586159635335e-08",
    "5.39030252995776476554e-15", "3.28200341580791294123e-22",
    "1.27065575308067607349e-29", "1.22933308981111328932e-36",
    "2.73370053816464559624e-44", "2.16741683877804819444e-51",
]


def load_ipio2(path):
    src = open(path).read()
    body = src[src.index("ipio2[] = {"):]
    body = body[:body.index("};")]
    vals = [int(v, 16) for v in re.findall(r"0x[0-9A-Fa-f]+", body)][:66]
    assert len(vals) == 66
    return vals


IPIO2 = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
    0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
    0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
    0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
    0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
    0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
    0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
]


# The 64 doubles, over all binary exponents, closest to a multiple of pi/2, from
# search_worst_cases() (continued fractions of 2^(E-52)*2/pi). Second column:
# -log2 |x mod pi/2|. The first is the published worst case 6381956970095103*2^797.
WORST_CASES = [
    (0x7506AC5B262CA1FF, 60.89),
    (0x4046C6CBC45DC8DE, 60.49),
    (0x7516AC5B262CA1FF, 59.89),
    (0x4056C6CBC45DC8DE, 59.49),
    (0x416B951F1572EBA5, 59.03),
    (0x482504CAC51F1EAF, 58.93),
    (0x7526AC5B262CA1FF, 58.89),
    (0x7DEE009C53148BE1, 58.78),
    (0x4066C6CBC45DC8DE, 58.49),
    (0x6404C96C11134D36, 58.48),
    (0x75BCFE482285F8ED, 58.14),
    (0x6A7DB41F3CB71D7B, 58.09),
    (0x4C4E7E44A78AC18C, 58.05),
    (0x417B951F1572EBA5, 58.03),
    (0x526C45CD11154DFD, 57.99),
    (0x483504CAC51F1EAF, 57.93),
    (0x66583009E2E9E2EB, 57.91),
    (0x4FCB2196364D750B, 57.90),
    (0x7536AC5B262CA1FF, 57.89),
    (0x7DFE009C53148BE1, 57.78),
    (0x4F569EAB0985179B, 57.53),
    (0x4076C6CBC45DC8DE, 57.49),
    (0x6414C96C11134D36, 57.48),
    (0x442782B7A20DF6D4, 57.44),
    (0x45966BD5424E5655, 57.44),
    (0x7AA4117573397D42, 57.15),
    (0x75CCFE482285F8ED, 57.14),
    (0x7FC61A3DB8C8D129, 57.10),
    (0x6A8DB41F3CB71D7B, 57.09),
    (0x62A8B28676CDCC5B, 57.06),
    (0x4C5E7E44A78AC18C, 57.05),
    (0x418B951F1572EBA5, 57.03),
    (0x527C45CD11154DFD, 56.99),
    (0x484504CAC51F1EAF, 56.93),
    (0x66683009E2E9E2EB, 56.91),
    (0x4FDB2196364D750B, 56.90),
    (0x7546AC5B262CA1FF, 56.89),
    (0x48E5AD5A62CB1CC9, 56.83),
    (0x6B0DFA8D18F2B3EE, 56.79),
    (0x7E0E009C53148BE1, 56.78),
    (0x4B50539B48D14C55, 56.73),
    (0x5E6B88CBB4E32576, 56.71),
    (0x597E3CA9B6C655CB, 56.70),
    (0x7196E8D778C94D66, 56.70),
    (0x4F669EAB0985179B, 56.53),
    (0x7B6E1987122B7E06, 56.53),
    (0x4086C6CBC45DC8DE, 56.49),
    (0x6424C96C11134D36, 56.48),
    (0x4C66DEB37DA81129, 56.47),
    (0x443782B7A20DF6D4, 56.44),
    (0x45A66BD5424E5655, 56.44),
    (0x79FFCE36F3EBFE43, 56.30),
    (0x6F6D5C8D09F26EBF, 56.27),
    (0x4DE33322FD48B212, 56.26),
    (0x44B99CAA5236FEEA, 56.22),
    (0x53944599AE031379, 56.21),
    (0x450E0664DBEDFEC5, 56.19),
    (0x7AB4117573397D42, 56.15),
    (0x5130809C95F020F7, 56.15),
    (0x75DCFE482285F8ED, 56.14),
    (0x7FD61A3DB8C8D129, 56.10),
    (0x6A9DB41F3CB71D7B, 56.09),
    (0x62B8B28676CDCC5B, 56.06),
    (0x419B951F1572EBA5, 56.03),
]


def build_constants(ipio2):
    K = []
    for nm, v in (("ZERO", 0.0), ("HALF", 0.5), ("ONE", 1.0), ("TWO", 2.0), ("THREE", 3.0),
                  ("FOUR", 4.0), ("EIGHT", 8.0), ("EIGHTH", 0.125), ("MONE", -1.0),
                  ("TWO24", 16777216.0), ("TWON24", 2.0 ** -24), ("P52X15", 1.5 * 2.0 ** 52)):
        K.append((nm, v))
    for nm, dec, hx in FDLIBM_DEC:
        v = float(dec)
        assert d2b(v) == hx, (nm, hex(d2b(v)), hex(hx))
        K.append((nm, v))
    for i, dec in enumerate(PIO2_TABLE_DEC):
        K.append(("PIO2TAB%d" % i, float(dec)))
    for i, v in enumerate(ipio2):
        K.append(("IPIO2_%d" % i, float(v)))
    index = {nm: i for i, (nm, _) in enumerate(K)}
    return K, index


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------
class Frame:
    def __init__(self):
        self.next = 16          # [rbp-8] holds the argument bits

    def s(self):
        n = self.next
        self.next += 8
        return n

    def arr(self, count):
        """element i lives at [rbp - base + 8*i]; base is the deepest slot."""
        base = self.next + 8 * (count - 1)
        self.next = base + 8
        return base


F = Frame()
X = 8
OUT = F.s()
IXS, QUAD = F.s(), F.s()
# kernel ports and locals
KX, KY, KOUT = F.s(), F.s(), F.s()
KZ, KW, KR, KV, KT1, KT2, KT3 = (F.s() for _ in range(7))
# rem_pio2 ports and locals
RX, RN, RY0, RY1 = (F.s() for _ in range(4))
HX, IX, MED, TIX, Z, T, T2, T3, FN, R, W, TT, J, I = (F.s() for _ in range(14))
E0, AX, ZL, TX0, TX1, TX2, NX = (F.s() for _ in range(7))
# k_rem_pio2 ports and locals
KN, KY0, KY1 = F.s(), F.s(), F.s()
JV, Q0, JX, MM, JZ, NN, IH, CARRY, KC, KK, II, JJ, TI, T2I, T3I = (F.s() for _ in range(15))
FW, ZK, TF, TF2 = (F.s() for _ in range(4))
XA = F.arr(3)
FA = F.arr(20)
QA = F.arr(20)
FQA = F.arr(20)
IQA = F.arr(20)
FRAME_SLOTS_END = F.next


# ---------------------------------------------------------------------------
# program builder
# ---------------------------------------------------------------------------
class Routine:
    def __init__(self, name):
        self.name = name
        self.ops = []
        self.nlab = 0

    def e(self, *op):
        self.ops.append(op)

    def lab(self):
        self.nlab += 1
        return "L%d" % self.nlab

    def place(self, L):
        self.e("label", L)

    # float sugar
    def fop(self, dst, a, op, b):
        self.e("ld0", a); self.e("ld1", b); self.e(op); self.e("st0", dst)

    def fopk(self, dst, a, op, k):
        self.e("ld0", a); self.e("ld1c", k); self.e(op); self.e("st0", dst)

    def fkop(self, dst, k, op, b):
        self.e("ld0c", k); self.e("ld1", b); self.e(op); self.e("st0", dst)

    def fkk(self, dst, k1, op, k2):
        self.e("ld0c", k1); self.e("ld1c", k2); self.e(op); self.e("st0", dst)

    def fset(self, dst, k):
        self.e("ld0c", k); self.e("st0", dst)

    def fmov(self, dst, src):
        self.e("ld0", src); self.e("st0", dst)

    # integer sugar
    def ld(self, reg, term):
        kind, v = term
        if kind == "s":
            self.e("load", reg, v)
        else:
            self.e("movi", reg, v)

    def iset(self, dst, v):
        self.e("movi", "rax", v); self.e("store", "rax", dst)

    def imov(self, dst, src):
        self.e("load", "rax", src); self.e("store", "rax", dst)

    def iadd(self, slot, v):
        self.e("load", "rax", slot); self.e("movi", "rbx", v); self.e("add", "rax", "rbx"); self.e("store", "rax", slot)

    def jump_unless_i(self, a, rel, b, L):
        """integer a REL b; jump to L when it is false. cmp rbx,rax sets flags of b - a."""
        self.ld("rax", a); self.ld("rbx", b); self.e("cmp_rbx_rax")
        cc = {"<": "setg", "<=": "setge", ">": "setl", ">=": "setle", "==": "sete", "!=": "setne"}[rel]
        self.e(cc); self.e("test_al"); self.e("jz", L)

    def jump_unless_f(self, a, rel, b, L):
        """float a REL b (operands never NaN at the call sites); jump to L when false."""
        self.e("ld0", a)
        if b[0] == "k":
            self.e("ld1c", b[1])
        else:
            self.e("ld1", b[1])
        self.e("ucomisd")
        cc = {"<": "setb", ">=": "setae", "==": "sete", ">": "seta"}[rel]
        self.e(cc); self.e("test_al"); self.e("jz", L)

    def idx(self, terms):
        """rax = sum of signed terms, left to right."""
        sign, t = terms[0]
        assert sign == 1
        self.ld("rax", t)
        for sign, t in terms[1:]:
            self.ld("rbx", t)
            self.e("add" if sign == 1 else "sub", "rax", "rbx")

    def _elem(self, arr, terms):
        self.idx(terms); self.e("mov_rbx_rax"); self.e("lea_rax_rbp", arr); self.e("mov_rdx_rax")

    def aget_f(self, arr, terms, dst):
        self._elem(arr, terms); self.e("load_idx"); self.e("movq_x0_rax"); self.e("st0", dst)

    def aset_f(self, arr, terms, src):
        self._elem(arr, terms); self.e("ld0", src); self.e("movq_rax_x0"); self.e("store_idx")

    def aget_i(self, arr, terms, dst):
        self._elem(arr, terms); self.e("load_idx"); self.e("store", "rax", dst)

    def aset_i(self, arr, terms, src):
        self._elem(arr, terms); self.e("load", "rax", src); self.e("store_idx")

    def pow2_mul(self, src, e_slot, dst, negate=False):
        """dst = src * 2^(+/-e): the power of two is built from its bit pattern (exact)."""
        if negate:
            self.e("movi", "rax", 1023); self.e("load", "rbx", e_slot); self.e("sub", "rax", "rbx")
        else:
            self.e("load", "rax", e_slot); self.e("movi", "rbx", 1023); self.e("add", "rax", "rbx")
        self.e("movi", "rcx", 52); self.e("shl_cl"); self.e("movq_x1_rax")
        self.e("ld0", src); self.e("mul"); self.e("st0", dst)

    def pow2_set(self, e_slot, dst):
        self.e("load", "rax", e_slot); self.e("movi", "rbx", 1023); self.e("add", "rax", "rbx")
        self.e("movi", "rcx", 52); self.e("shl_cl"); self.e("movq_x0_rax"); self.e("st0", dst)


S_ = lambda slot: ("s", slot)
C_ = lambda v: ("c", v)
P = lambda slot: (1, ("s", slot))
M_ = lambda slot: (-1, ("s", slot))
PC = lambda v: (1, ("c", v))
MC = lambda v: (-1, ("c", v))


def build_k_sin(iy):
    r = Routine("k_sin%d" % iy)
    r.fop(KZ, KX, "mul", KX)
    r.fop(KW, KZ, "mul", KZ)
    r.fopk(KT1, KZ, "mul", "S4"); r.fkop(KT1, "S3", "add", KT1)
    r.fop(KT1, KZ, "mul", KT1); r.fkop(KT1, "S2", "add", KT1)
    r.fopk(KT2, KZ, "mul", "S6"); r.fkop(KT2, "S5", "add", KT2)
    r.fop(KT3, KZ, "mul", KW); r.fop(KT3, KT3, "mul", KT2)
    r.fop(KR, KT1, "add", KT3)
    r.fop(KV, KZ, "mul", KX)
    if iy == 0:
        r.fop(KT1, KZ, "mul", KR); r.fkop(KT1, "S1", "add", KT1)
        r.fop(KT1, KV, "mul", KT1); r.fop(KOUT, KX, "add", KT1)
    else:
        r.fkop(KT1, "HALF", "mul", KY); r.fop(KT2, KV, "mul", KR)
        r.fop(KT1, KT1, "sub", KT2); r.fop(KT1, KZ, "mul", KT1)
        r.fop(KT1, KT1, "sub", KY); r.fopk(KT2, KV, "mul", "S1")
        r.fop(KT1, KT1, "sub", KT2); r.fop(KOUT, KX, "sub", KT1)
    return r


def build_k_cos():
    r = Routine("k_cos")
    r.fop(KZ, KX, "mul", KX); r.fop(KW, KZ, "mul", KZ)
    r.fopk(KT1, KZ, "mul", "C3"); r.fkop(KT1, "C2", "add", KT1); r.fop(KT1, KZ, "mul", KT1)
    r.fkop(KT1, "C1", "add", KT1); r.fop(KT1, KZ, "mul", KT1)
    r.fopk(KT2, KZ, "mul", "C6"); r.fkop(KT2, "C5", "add", KT2); r.fop(KT2, KZ, "mul", KT2)
    r.fkop(KT2, "C4", "add", KT2); r.fop(KT3, KW, "mul", KW); r.fop(KT3, KT3, "mul", KT2)
    r.fop(KR, KT1, "add", KT3)
    r.fkop(KV, "HALF", "mul", KZ)            # hz
    r.fkop(KW, "ONE", "sub", KV)             # w = 1 - hz
    r.fkop(KT1, "ONE", "sub", KW); r.fop(KT1, KT1, "sub", KV)
    r.fop(KT2, KZ, "mul", KR); r.fop(KT3, KX, "mul", KY); r.fop(KT2, KT2, "sub", KT3)
    r.fop(KT1, KT1, "add", KT2); r.fop(KOUT, KW, "add", KT1)
    return r


def small_case(r, mult, positive, nval, Lret):
    op = "sub" if positive else "add"
    if mult == 1:
        r.e("ld0", RX); r.e("ld1c", "PIO2_1"); r.e(op); r.e("st0", Z)
        r.e("ld0", Z); r.e("ld1c", "PIO2_1T"); r.e(op); r.e("st0", RY0)
        r.fop(T, Z, "sub", RY0); r.e("ld0", T); r.e("ld1c", "PIO2_1T"); r.e(op); r.e("st0", RY1)
    else:
        mk = {2: "TWO", 3: "THREE", 4: "FOUR"}[mult]
        r.fkk(T2, mk, "mul", "PIO2_1"); r.fop(Z, RX, op, T2)
        r.fkk(T2, mk, "mul", "PIO2_1T"); r.fop(RY0, Z, op, T2)
        r.fop(T, Z, "sub", RY0); r.fkk(T2, mk, "mul", "PIO2_1T"); r.fop(RY1, T, op, T2)
    r.iset(RN, nval)
    r.e("jmp", Lret)


def expo_diff(r, ys, js, out):
    r.e("ld0", ys); r.e("movq_rax_x0"); r.e("movi", "rcx", 52); r.e("shr_cl")
    r.e("movi", "rbx", 0x7FF); r.e("and"); r.e("mov", "rbx", "rax")
    r.e("load", "rax", js); r.e("sub", "rax", "rbx"); r.e("store", "rax", out)


def build_rem_pio2():
    r = Routine("rem_pio2")
    Lret, Lblk2, Lmedchk, Lmedium, Lnotmed = r.lab(), r.lab(), r.lab(), r.lab(), r.lab()
    r.e("ld0", RX); r.e("movq_rax_x0"); r.e("movi", "rcx", 32); r.e("sar_cl"); r.e("store", "rax", HX)
    r.e("movi", "rbx", 0x7FFFFFFF); r.e("and"); r.e("store", "rax", IX)
    r.iset(MED, 0)
    # |x| ~<= 5pi/4
    r.jump_unless_i(S_(IX), "<=", C_(0x400F6A7A), Lblk2)
    La, Lb, Lneg1, Lneg2 = r.lab(), r.lab(), r.lab(), r.lab()
    r.e("load", "rax", IX); r.e("movi", "rbx", 0xFFFFF); r.e("and"); r.e("store", "rax", TIX)
    r.jump_unless_i(S_(TIX), "==", C_(0x921FB), La)
    r.iset(MED, 1); r.e("jmp", Lblk2)
    r.place(La)
    r.jump_unless_i(S_(IX), "<=", C_(0x4002D97C), Lb)
    r.jump_unless_i(S_(HX), ">", C_(0), Lneg1)
    small_case(r, 1, True, 1, Lret)
    r.place(Lneg1); small_case(r, 1, False, -1, Lret)
    r.place(Lb)
    r.jump_unless_i(S_(HX), ">", C_(0), Lneg2)
    small_case(r, 2, True, 2, Lret)
    r.place(Lneg2); small_case(r, 2, False, -2, Lret)
    # |x| ~<= 9pi/4
    r.place(Lblk2)
    r.jump_unless_i(S_(MED), "==", C_(0), Lmedchk)
    r.jump_unless_i(S_(IX), "<=", C_(0x401C463B), Lmedchk)
    Lc, Lc3, Lc4, Lneg3, Lneg4 = r.lab(), r.lab(), r.lab(), r.lab(), r.lab()
    r.jump_unless_i(S_(IX), "<=", C_(0x4015FDBC), Lc)
    r.jump_unless_i(S_(IX), "==", C_(0x4012D97C), Lc3)
    r.iset(MED, 1); r.e("jmp", Lmedchk)
    r.place(Lc3)
    r.jump_unless_i(S_(HX), ">", C_(0), Lneg3)
    small_case(r, 3, True, 3, Lret)
    r.place(Lneg3); small_case(r, 3, False, -3, Lret)
    r.place(Lc)
    r.jump_unless_i(S_(IX), "==", C_(0x401921FB), Lc4)
    r.iset(MED, 1); r.e("jmp", Lmedchk)
    r.place(Lc4)
    r.jump_unless_i(S_(HX), ">", C_(0), Lneg4)
    small_case(r, 4, True, 4, Lret)
    r.place(Lneg4); small_case(r, 4, False, -4, Lret)
    # medium?
    r.place(Lmedchk)
    Lmt2 = r.lab()
    r.jump_unless_i(S_(MED), "==", C_(1), Lmt2)
    r.e("jmp", Lmedium)
    r.place(Lmt2)
    r.jump_unless_i(S_(IX), "<", C_(0x413921FB), Lnotmed)
    r.place(Lmedium)
    r.fopk(FN, RX, "mul", "INVPIO2"); r.fopk(FN, FN, "add", "P52X15"); r.fopk(FN, FN, "sub", "P52X15")
    r.e("ld0", FN); r.e("cvttsd2si"); r.e("store", "rax", RN)
    r.fopk(T, FN, "mul", "PIO2_1"); r.fop(R, RX, "sub", T)
    r.fopk(W, FN, "mul", "PIO2_1T")
    r.e("load", "rax", IX); r.e("movi", "rcx", 20); r.e("sar_cl"); r.e("store", "rax", J)
    r.fop(RY0, R, "sub", W)
    expo_diff(r, RY0, J, I)
    Ly1 = r.lab()
    r.jump_unless_i(S_(I), ">", C_(16), Ly1)
    r.fmov(TT, R)
    r.fopk(W, FN, "mul", "PIO2_2"); r.fop(R, TT, "sub", W)
    r.fop(T2, TT, "sub", R); r.fop(T2, T2, "sub", W); r.fopk(T3, FN, "mul", "PIO2_2T"); r.fop(W, T3, "sub", T2)
    r.fop(RY0, R, "sub", W)
    expo_diff(r, RY0, J, I)
    r.jump_unless_i(S_(I), ">", C_(49), Ly1)
    r.fmov(TT, R)
    r.fopk(W, FN, "mul", "PIO2_3"); r.fop(R, TT, "sub", W)
    r.fop(T2, TT, "sub", R); r.fop(T2, T2, "sub", W); r.fopk(T3, FN, "mul", "PIO2_3T"); r.fop(W, T3, "sub", T2)
    r.fop(RY0, R, "sub", W)
    r.place(Ly1)
    r.fop(T, R, "sub", RY0); r.fop(RY1, T, "sub", W)
    r.e("jmp", Lret)
    # inf / NaN
    r.place(Lnotmed)
    Lfinite = r.lab()
    r.jump_unless_i(S_(IX), ">=", C_(0x7FF00000), Lfinite)
    r.fop(RY0, RX, "sub", RX); r.fmov(RY1, RY0); r.iset(RN, 0); r.e("jmp", Lret)
    # large
    r.place(Lfinite)
    r.e("load", "rax", IX); r.e("movi", "rcx", 20); r.e("sar_cl"); r.e("movi", "rbx", 1046)
    r.e("sub", "rax", "rbx"); r.e("store", "rax", E0)
    Lpos = r.lab()
    r.fmov(AX, RX)
    r.jump_unless_f(RX, "<", ("k", "ZERO"), Lpos)
    r.fopk(AX, RX, "mul", "MONE")
    r.place(Lpos)
    r.pow2_mul(AX, E0, ZL, negate=True)
    for tx in (TX0, TX1):
        r.e("ld0", ZL); r.e("cvttsd2si"); r.e("cvtsi2sd"); r.e("st0", tx)
        r.fop(ZL, ZL, "sub", tx); r.fopk(ZL, ZL, "mul", "TWO24")
    r.fmov(TX2, ZL)
    Lnx = r.lab()
    r.iset(NX, 3)
    r.jump_unless_f(TX2, "==", ("k", "ZERO"), Lnx)
    r.iset(NX, 2)
    r.jump_unless_f(TX1, "==", ("k", "ZERO"), Lnx)
    r.iset(NX, 1)
    r.place(Lnx)
    r.e("call", "k_rem_pio2")
    Lkpos = r.lab()
    r.jump_unless_i(S_(HX), "<", C_(0), Lkpos)
    r.e("movi", "rax", 0); r.e("load", "rbx", KN); r.e("sub", "rax", "rbx"); r.e("store", "rax", RN)
    r.fopk(RY0, KY0, "mul", "MONE"); r.fopk(RY1, KY1, "mul", "MONE")
    r.e("jmp", Lret)
    r.place(Lkpos)
    r.imov(RN, KN); r.fmov(RY0, KY0); r.fmov(RY1, KY1)
    r.place(Lret)
    return r


def loop_head(r, a, rel, b):
    Ltop, Lend = r.lab(), r.lab()
    r.place(Ltop)
    r.jump_unless_i(a, rel, b, Lend)
    return Ltop, Lend


def loop_tail(r, Ltop, Lend):
    r.e("jmp", Ltop)
    r.place(Lend)


def build_k_rem_pio2():
    r = Routine("k_rem_pio2")
    for i, tx in enumerate((TX0, TX1, TX2)):
        r.aset_f(XA, [PC(i)], tx)
    r.e("load", "rax", NX); r.e("movi", "rbx", 1); r.e("sub", "rax", "rbx"); r.e("store", "rax", JX)
    r.e("load", "rax", E0); r.e("movi", "rbx", 3); r.e("sub", "rax", "rbx")
    r.e("movi", "rcx", 24); r.e("cqo_idiv"); r.e("store", "rax", JV)
    L = r.lab(); r.jump_unless_i(S_(JV), "<", C_(0), L); r.iset(JV, 0); r.place(L)
    r.e("load", "rax", JV); r.e("movi", "rbx", 1); r.e("add", "rax", "rbx"); r.e("movi", "rbx", 24)
    r.e("imul", "rax", "rbx"); r.e("mov", "rbx", "rax"); r.e("load", "rax", E0); r.e("sub", "rax", "rbx")
    r.e("store", "rax", Q0)
    r.e("load", "rax", JV); r.e("load", "rbx", JX); r.e("sub", "rax", "rbx"); r.e("store", "rax", JJ)
    r.e("load", "rax", JX); r.e("movi", "rbx", 4); r.e("add", "rax", "rbx"); r.e("store", "rax", MM)

    # f[0..m]
    r.iset(II, 0)
    Lt, Le = loop_head(r, S_(II), "<=", S_(MM))
    Ltab, Lsto = r.lab(), r.lab()
    r.jump_unless_i(S_(JJ), "<", C_(0), Ltab)
    r.fset(TF, "ZERO"); r.e("jmp", Lsto)
    r.place(Ltab); r.e("load", "rax", JJ); r.e("ldtab", "IPIO2_0"); r.e("st0", TF)
    r.place(Lsto); r.aset_f(FA, [P(II)], TF)
    r.iadd(II, 1); r.iadd(JJ, 1)
    loop_tail(r, Lt, Le)

    # q[0..jk]
    r.iset(II, 0)
    Lt, Le = loop_head(r, S_(II), "<=", C_(4))
    r.fset(FW, "ZERO"); r.iset(JJ, 0)
    Lt2, Le2 = loop_head(r, S_(JJ), "<=", S_(JX))
    r.aget_f(XA, [P(JJ)], TF); r.aget_f(FA, [P(JX), P(II), M_(JJ)], TF2)
    r.fop(TF, TF, "mul", TF2); r.fop(FW, FW, "add", TF)
    r.iadd(JJ, 1)
    loop_tail(r, Lt2, Le2)
    r.aset_f(QA, [P(II)], FW)
    r.iadd(II, 1)
    loop_tail(r, Lt, Le)

    r.iset(JZ, 4)
    Lrec = r.lab()
    r.place(Lrec)
    # distill q[] into iq[]
    r.iset(II, 0); r.imov(JJ, JZ)
    r.aget_f(QA, [P(JZ)], ZK)
    Lt, Le = loop_head(r, S_(JJ), ">", C_(0))
    r.fkop(FW, "TWON24", "mul", ZK); r.e("ld0", FW); r.e("cvttsd2si"); r.e("cvtsi2sd"); r.e("st0", FW)
    r.fkop(TF, "TWO24", "mul", FW); r.fop(TF, ZK, "sub", TF); r.e("ld0", TF); r.e("cvttsd2si"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(II)], TI)
    r.aget_f(QA, [P(JJ), MC(1)], TF); r.fop(ZK, TF, "add", FW)
    r.iadd(II, 1); r.iadd(JJ, -1)
    loop_tail(r, Lt, Le)

    # n
    r.pow2_mul(ZK, Q0, ZK)
    r.fopk(TF, ZK, "mul", "EIGHTH"); r.e("ld0", TF); r.e("cvttsd2si"); r.e("cvtsi2sd"); r.e("st0", TF)
    r.fkop(TF, "EIGHT", "mul", TF); r.fop(ZK, ZK, "sub", TF)
    r.e("ld0", ZK); r.e("cvttsd2si"); r.e("store", "rax", NN)
    r.e("load", "rax", NN); r.e("cvtsi2sd"); r.e("st0", TF); r.fop(ZK, ZK, "sub", TF)
    r.iset(IH, 0)
    Lq0zero, Lq0neg, Lihdone = r.lab(), r.lab(), r.lab()
    r.jump_unless_i(S_(Q0), ">", C_(0), Lq0zero)
    # rcx = 24 - q0
    r.e("load", "rbx", Q0); r.e("movi", "rcx", 24); r.e("sub", "rcx", "rbx")
    r.aget_i(IQA, [P(JZ), MC(1)], TI)
    r.e("load", "rax", TI); r.e("sar_cl"); r.e("store", "rax", TI)
    r.e("load", "rax", NN); r.e("load", "rbx", TI); r.e("add", "rax", "rbx"); r.e("store", "rax", NN)
    r.e("load", "rax", TI); r.e("shl_cl"); r.e("store", "rax", T2I)
    r.aget_i(IQA, [P(JZ), MC(1)], T3I)
    r.e("load", "rax", T3I); r.e("load", "rbx", T2I); r.e("sub", "rax", "rbx"); r.e("store", "rax", T3I)
    r.aset_i(IQA, [P(JZ), MC(1)], T3I)
    r.e("load", "rbx", Q0); r.e("movi", "rcx", 23); r.e("sub", "rcx", "rbx")
    r.e("load", "rax", T3I); r.e("sar_cl"); r.e("store", "rax", IH)
    r.e("jmp", Lihdone)
    r.place(Lq0zero)
    r.jump_unless_i(S_(Q0), "==", C_(0), Lq0neg)
    r.aget_i(IQA, [P(JZ), MC(1)], TI)
    r.e("load", "rax", TI); r.e("movi", "rcx", 23); r.e("sar_cl"); r.e("store", "rax", IH)
    r.e("jmp", Lihdone)
    r.place(Lq0neg)
    r.jump_unless_f(ZK, ">=", ("k", "HALF"), Lihdone)
    r.iset(IH, 2)
    r.place(Lihdone)

    # ih > 0: q > 0.5
    Lihend = r.lab()
    r.jump_unless_i(S_(IH), ">", C_(0), Lihend)
    r.iadd(NN, 1)
    r.iset(CARRY, 0); r.iset(II, 0)
    Lt, Le = loop_head(r, S_(II), "<", S_(JZ))
    r.aget_i(IQA, [P(II)], JJ)
    Lcar1, Lnext = r.lab(), r.lab()
    r.jump_unless_i(S_(CARRY), "==", C_(0), Lcar1)
    r.jump_unless_i(S_(JJ), "!=", C_(0), Lnext)
    r.iset(CARRY, 1)
    r.e("movi", "rax", 0x1000000); r.e("load", "rbx", JJ); r.e("sub", "rax", "rbx"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(II)], TI); r.e("jmp", Lnext)
    r.place(Lcar1)
    r.e("movi", "rax", 0xFFFFFF); r.e("load", "rbx", JJ); r.e("sub", "rax", "rbx"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(II)], TI)
    r.place(Lnext)
    r.iadd(II, 1)
    loop_tail(r, Lt, Le)
    Lm1, Lm2 = r.lab(), r.lab()
    r.jump_unless_i(S_(Q0), "==", C_(1), Lm1)
    r.aget_i(IQA, [P(JZ), MC(1)], TI)
    r.e("load", "rax", TI); r.e("movi", "rbx", 0x7FFFFF); r.e("and"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(JZ), MC(1)], TI)
    r.e("jmp", Lm2)
    r.place(Lm1)
    r.jump_unless_i(S_(Q0), "==", C_(2), Lm2)
    r.aget_i(IQA, [P(JZ), MC(1)], TI)
    r.e("load", "rax", TI); r.e("movi", "rbx", 0x3FFFFF); r.e("and"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(JZ), MC(1)], TI)
    r.place(Lm2)
    r.jump_unless_i(S_(IH), "==", C_(2), Lihend)
    r.fkop(ZK, "ONE", "sub", ZK)
    r.jump_unless_i(S_(CARRY), "!=", C_(0), Lihend)
    r.pow2_set(Q0, TF)
    r.fop(ZK, ZK, "sub", TF)
    r.place(Lihend)

    # recomputation
    Lnorec = r.lab()
    r.jump_unless_f(ZK, "==", ("k", "ZERO"), Lnorec)
    r.iset(JJ, 0)
    r.e("load", "rax", JZ); r.e("movi", "rbx", 1); r.e("sub", "rax", "rbx"); r.e("store", "rax", II)
    Lt, Le = loop_head(r, S_(II), ">=", C_(4))
    r.aget_i(IQA, [P(II)], TI)
    r.e("load", "rax", JJ); r.e("load", "rbx", TI); r.e("or"); r.e("store", "rax", JJ)
    r.iadd(II, -1)
    loop_tail(r, Lt, Le)
    r.jump_unless_i(S_(JJ), "==", C_(0), Lnorec)
    r.iset(KC, 1)
    Lk, Lke = r.lab(), r.lab()
    r.place(Lk)
    r.aget_i(IQA, [PC(4), M_(KC)], TI)
    r.jump_unless_i(S_(TI), "==", C_(0), Lke)
    r.iadd(KC, 1)
    r.e("jmp", Lk)
    r.place(Lke)
    r.e("load", "rax", JZ); r.e("movi", "rbx", 1); r.e("add", "rax", "rbx"); r.e("store", "rax", II)
    r.e("load", "rax", JZ); r.e("load", "rbx", KC); r.e("add", "rax", "rbx"); r.e("store", "rax", T2I)
    Lt, Le = loop_head(r, S_(II), "<=", S_(T2I))
    r.e("load", "rax", JV); r.e("load", "rbx", II); r.e("add", "rax", "rbx"); r.e("ldtab", "IPIO2_0"); r.e("st0", TF)
    r.aset_f(FA, [P(JX), P(II)], TF)
    r.fset(FW, "ZERO"); r.iset(JJ, 0)
    Lt2, Le2 = loop_head(r, S_(JJ), "<=", S_(JX))
    r.aget_f(XA, [P(JJ)], TF); r.aget_f(FA, [P(JX), P(II), M_(JJ)], TF2)
    r.fop(TF, TF, "mul", TF2); r.fop(FW, FW, "add", TF)
    r.iadd(JJ, 1)
    loop_tail(r, Lt2, Le2)
    r.aset_f(QA, [P(II)], FW)
    r.iadd(II, 1)
    loop_tail(r, Lt, Le)
    r.e("load", "rax", JZ); r.e("load", "rbx", KC); r.e("add", "rax", "rbx"); r.e("store", "rax", JZ)
    r.e("jmp", Lrec)
    r.place(Lnorec)

    # chop off zero terms
    Lnz, Lchop = r.lab(), r.lab()
    r.jump_unless_f(ZK, "==", ("k", "ZERO"), Lnz)
    r.iadd(JZ, -1); r.iadd(Q0, -24)
    Lw, Lwe = r.lab(), r.lab()
    r.place(Lw)
    r.aget_i(IQA, [P(JZ)], TI)
    r.jump_unless_i(S_(TI), "==", C_(0), Lwe)
    r.iadd(JZ, -1); r.iadd(Q0, -24)
    r.e("jmp", Lw)
    r.place(Lwe)
    r.e("jmp", Lchop)
    r.place(Lnz)
    r.pow2_mul(ZK, Q0, ZK, negate=True)
    Lsmall = r.lab()
    r.jump_unless_f(ZK, ">=", ("k", "TWO24"), Lsmall)
    r.fkop(FW, "TWON24", "mul", ZK); r.e("ld0", FW); r.e("cvttsd2si"); r.e("cvtsi2sd"); r.e("st0", FW)
    r.fkop(TF, "TWO24", "mul", FW); r.fop(TF, ZK, "sub", TF); r.e("ld0", TF); r.e("cvttsd2si"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(JZ)], TI)
    r.iadd(JZ, 1); r.iadd(Q0, 24)
    r.e("ld0", FW); r.e("cvttsd2si"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(JZ)], TI)
    r.e("jmp", Lchop)
    r.place(Lsmall)
    r.e("ld0", ZK); r.e("cvttsd2si"); r.e("store", "rax", TI)
    r.aset_i(IQA, [P(JZ)], TI)
    r.place(Lchop)

    # integer chunks back to floating point
    r.pow2_set(Q0, FW)
    r.imov(II, JZ)
    Lt, Le = loop_head(r, S_(II), ">=", C_(0))
    r.aget_i(IQA, [P(II)], TI)
    r.e("load", "rax", TI); r.e("cvtsi2sd"); r.e("st0", TF)
    r.fop(TF, FW, "mul", TF); r.aset_f(QA, [P(II)], TF)
    r.fopk(FW, FW, "mul", "TWON24")
    r.iadd(II, -1)
    loop_tail(r, Lt, Le)

    # PIo2[0..jp] * q[jz..0]
    r.imov(II, JZ)
    Lt, Le = loop_head(r, S_(II), ">=", C_(0))
    r.fset(FW, "ZERO"); r.iset(KK, 0)
    r.e("load", "rax", JZ); r.e("load", "rbx", II); r.e("sub", "rax", "rbx"); r.e("store", "rax", T2I)
    Lt2, Le2 = loop_head(r, S_(KK), "<=", C_(4))
    r.jump_unless_i(S_(KK), "<=", S_(T2I), Le2)
    r.e("load", "rax", KK); r.e("ldtab", "PIO2TAB0"); r.e("st0", TF)
    r.aget_f(QA, [P(II), P(KK)], TF2)
    r.fop(TF, TF, "mul", TF2); r.fop(FW, FW, "add", TF)
    r.iadd(KK, 1)
    loop_tail(r, Lt2, Le2)
    r.aset_f(FQA, [P(JZ), M_(II)], FW)
    r.iadd(II, -1)
    loop_tail(r, Lt, Le)

    # compress fq[] into y[] (prec 1)
    r.fset(FW, "ZERO"); r.imov(II, JZ)
    Lt, Le = loop_head(r, S_(II), ">=", C_(0))
    r.aget_f(FQA, [P(II)], TF); r.fop(FW, FW, "add", TF)
    r.iadd(II, -1)
    loop_tail(r, Lt, Le)
    r.fmov(KY0, FW)
    L0 = r.lab()
    r.jump_unless_i(S_(IH), "!=", C_(0), L0)
    r.fopk(KY0, FW, "mul", "MONE")
    r.place(L0)
    r.aget_f(FQA, [PC(0)], TF); r.fop(FW, TF, "sub", FW)
    r.iset(II, 1)
    Lt, Le = loop_head(r, S_(II), "<=", S_(JZ))
    r.aget_f(FQA, [P(II)], TF); r.fop(FW, FW, "add", TF)
    r.iadd(II, 1)
    loop_tail(r, Lt, Le)
    r.fmov(KY1, FW)
    L1 = r.lab()
    r.jump_unless_i(S_(IH), "!=", C_(0), L1)
    r.fopk(KY1, FW, "mul", "MONE")
    r.place(L1)
    r.e("load", "rax", NN); r.e("movi", "rbx", 7); r.e("and"); r.e("store", "rax", KN)
    return r


def build_top(which):
    r = Routine(which)
    Ldone = r.lab()
    r.e("ld0", X); r.e("movq_rax_x0"); r.e("movi", "rcx", 32); r.e("shr_cl")
    r.e("movi", "rbx", 0x7FFFFFFF); r.e("and"); r.e("store", "rax", IXS)
    Lbig, Lk = r.lab(), r.lab()
    r.jump_unless_i(S_(IXS), "<=", C_(0x3FE921FB), Lbig)
    if which == "sin":
        r.jump_unless_i(S_(IXS), "<", C_(0x3E500000), Lk)
        r.fmov(OUT, X); r.e("jmp", Ldone)
        r.place(Lk)
        r.fmov(KX, X); r.fset(KY, "ZERO"); r.e("call", "k_sin0"); r.fmov(OUT, KOUT); r.e("jmp", Ldone)
    else:
        r.jump_unless_i(S_(IXS), "<", C_(0x3E46A09E), Lk)
        r.fset(OUT, "ONE"); r.e("jmp", Ldone)
        r.place(Lk)
        r.fmov(KX, X); r.fset(KY, "ZERO"); r.e("call", "k_cos"); r.fmov(OUT, KOUT); r.e("jmp", Ldone)
    r.place(Lbig)
    Lfin = r.lab()
    r.jump_unless_i(S_(IXS), ">=", C_(0x7FF00000), Lfin)
    r.fop(OUT, X, "sub", X); r.e("jmp", Ldone)
    r.place(Lfin)
    r.fmov(RX, X); r.e("call", "rem_pio2")
    r.e("load", "rax", RN); r.e("movi", "rbx", 3); r.e("and"); r.e("store", "rax", QUAD)
    r.fmov(KX, RY0); r.fmov(KY, RY1)
    L1, L2, L3 = r.lab(), r.lab(), r.lab()
    # quadrant -> (kernel, negate)
    table = {"sin": [("k_sin1", False), ("k_cos", False), ("k_sin1", True), ("k_cos", True)],
             "cos": [("k_cos", False), ("k_sin1", True), ("k_cos", True), ("k_sin1", False)]}[which]
    nexts = [L1, L2, L3, None]
    for q, (kern, neg) in enumerate(table):
        if nexts[q] is not None:
            r.jump_unless_i(S_(QUAD), "==", C_(q), nexts[q])
        r.e("call", kern)
        if neg:
            r.fopk(OUT, KOUT, "mul", "MONE")
        else:
            r.fmov(OUT, KOUT)
        if nexts[q] is not None:
            r.e("jmp", Ldone)
            r.place(nexts[q])
    r.place(Ldone)
    return r


def build_program():
    routines = [build_k_sin(0), build_k_sin(1), build_k_cos(), build_k_rem_pio2(),
                build_rem_pio2(), build_top("sin"), build_top("cos")]
    return {r.name: r for r in routines}


# ---------------------------------------------------------------------------
# C rendering (verification at scale)
# ---------------------------------------------------------------------------
def render_c(prog, K, index):
    out = []
    out.append("#include <stdint.h>\n#include <string.h>\n#include <stdio.h>\n#include <math.h>")
    out.append("static const double KT[%d] = {%s};" % (len(K), ", ".join(v.hex() for _, v in K)))
    nslots = FRAME_SLOTS_END // 8 + 8
    uid = [0]

    def emit_routine(name, lines, prefix):
        for op in prog[name].ops:
            o = op[0]
            a = op[1:]
            if o == "label":
                lines.append("%s_%s: ;" % (prefix, a[0]))
            elif o in ("jz", "jnz", "jmp"):
                cond = {"jz": "if (zf) ", "jnz": "if (!zf) ", "jmp": ""}[o]
                lines.append("%sgoto %s_%s;" % (cond, prefix, a[0]))
            elif o == "call":
                uid[0] += 1
                emit_routine(a[0], lines, "%s_c%d" % (prefix, uid[0]))
            else:
                lines.append(c_op(o, a, index))

    for which in ("sin", "cos"):
        out.append("double fdt_%s(double xin) {" % which)
        out.append("  uint64_t mem[%d]; double x0 = 0, x1 = 0; uint64_t rax = 0, rbx = 0, rcx = 0, rdx = 0;" % nslots)
        out.append("  int zf = 0, cf = 0, al = 0; int64_t ca = 0, cb = 0; uint64_t ua = 0, ub = 0;")
        out.append("  memset(mem, 0, sizeof mem); memcpy(&mem[1], &xin, 8);")
        body = []
        emit_routine(which, body, "t")
        out.extend("  " + l for l in body)
        out.append("  memcpy(&x0, &mem[%d], 8); return x0;\n}" % (OUT // 8))
    out.append("""int main(void) {
  char line[64]; uint64_t u;
  while (fgets(line, sizeof line, stdin)) {
    if (sscanf(line, "%lx", &u) != 1) continue;
    double x; memcpy(&x, &u, 8);
    double s = fdt_sin(x), c = fdt_cos(x); uint64_t sb, cb2; memcpy(&sb, &s, 8); memcpy(&cb2, &c, 8);
    printf("%016lx %016lx %016lx\\n", u, sb, cb2);
  }
  return 0;
}""")
    return "\n".join(out) + "\n"


def c_op(o, a, index):
    reg = lambda r: r
    if o == "ld0": return "memcpy(&x0, &mem[%d], 8);" % (a[0] // 8)
    if o == "ld1": return "memcpy(&x1, &mem[%d], 8);" % (a[0] // 8)
    if o == "st0": return "memcpy(&mem[%d], &x0, 8);" % (a[0] // 8)
    if o == "st1": return "memcpy(&mem[%d], &x1, 8);" % (a[0] // 8)
    if o == "ld0c": return "x0 = KT[%d];" % index[a[0]]
    if o == "ld1c": return "x1 = KT[%d];" % index[a[0]]
    if o == "add":
        if len(a) == 0: return "x0 = x0 + x1;"
        return "%s = %s + %s;" % (a[0], a[0], a[1])
    if o == "sub":
        if len(a) == 0: return "x0 = x0 - x1;"
        return "%s = %s - %s;" % (a[0], a[0], a[1])
    if o == "mul": return "x0 = x0 * x1;"
    if o == "div": return "x0 = x0 / x1;"
    if o == "ucomisd": return "{ int un = isnan(x0) || isnan(x1); zf = un || x0 == x1; cf = un || x0 < x1; }"
    if o == "cvttsd2si": return "rax = (uint64_t)(int64_t)x0;"
    if o == "cvtsi2sd": return "x0 = (double)(int64_t)rax;"
    if o == "movq_rax_x0": return "memcpy(&rax, &x0, 8);"
    if o == "movq_x0_rax": return "memcpy(&x0, &rax, 8);"
    if o == "movq_x1_rax": return "memcpy(&x1, &rax, 8);"
    if o == "load": return "%s = mem[%d];" % (a[0], a[1] // 8)
    if o == "store": return "mem[%d] = %s;" % (a[1] // 8, a[0])
    if o == "movi": return "%s = (uint64_t)(int64_t)(%dLL);" % (a[0], a[1])
    if o == "mov": return "%s = %s;" % (a[0], a[1])
    if o == "and": return "rax &= rbx;"
    if o == "or": return "rax |= rbx;"
    if o == "xor": return "rax ^= rbx;"
    if o == "shl_cl": return "rax <<= (rcx & 63);"
    if o == "shr_cl": return "rax >>= (rcx & 63);"
    if o == "sar_cl": return "rax = (uint64_t)((int64_t)rax >> (rcx & 63));"
    if o == "imul": return "%s = (uint64_t)((int64_t)%s * (int64_t)%s);" % (a[0], a[0], a[1])
    if o == "cqo_idiv": return "{ int64_t n_ = (int64_t)rax, d_ = (int64_t)rcx; rax = (uint64_t)(n_ / d_); rdx = (uint64_t)(n_ % d_); }"
    if o == "cmp_rbx_rax": return "ca = (int64_t)rbx; cb = (int64_t)rax; ua = rbx; ub = rax; zf = (rbx == rax); cf = (rbx < rax);"
    if o == "setl": return "al = ca < cb;"
    if o == "setg": return "al = ca > cb;"
    if o == "setle": return "al = ca <= cb;"
    if o == "setge": return "al = ca >= cb;"
    if o == "sete": return "al = zf;"
    if o == "setne": return "al = !zf;"
    if o == "setb": return "al = cf;"
    if o == "setae": return "al = !cf;"
    if o == "seta": return "al = !cf && !zf;"
    if o == "test_al": return "zf = (al == 0);"
    if o == "test_rax": return "zf = (rax == 0);"
    if o == "lea_rax_rbp": return "rax = (uint64_t)(int64_t)(-%d);" % a[0]
    if o == "mov_rdx_rax": return "rdx = rax;"
    if o == "mov_rbx_rax": return "rbx = rax;"
    if o == "load_idx": return "rax = mem[(-((int64_t)rdx + 8 * (int64_t)rbx)) / 8];"
    if o == "store_idx": return "mem[(-((int64_t)rdx + 8 * (int64_t)rbx)) / 8] = rax;"
    if o == "ldtab": return "rbx = rax; x0 = KT[%d + (int64_t)rbx]; memcpy(&rax, &x0, 8);" % index[a[0]]
    raise ValueError("no C rendering for %r" % (o,))


# ---------------------------------------------------------------------------
# Python interpreter: runs the instruction list itself
# ---------------------------------------------------------------------------
def run(prog, K, index, which, xbits):
    """Execute routine `which` ("sin" or "cos") on the double whose bits are xbits.

    Returns (result_bits, instructions_executed). Memory is addressed like the
    frame: slot n lives at address -n, arrays at -base + 8*i.
    """
    mem = {-8: xbits & MASK}
    reg = {"rax": 0, "rbx": 0, "rcx": 0, "rdx": 0}
    st = {"x0": 0.0, "x1": 0.0, "zf": 0, "cf": 0, "al": 0, "ca": 0, "cb": 0}
    executed = [0]

    def fget(addr):
        return b2d(mem.get(addr, 0))

    def exec_routine(name):
        ops = prog[name].ops
        labels = {op[1]: i for i, op in enumerate(ops) if op[0] == "label"}
        pc = 0
        while pc < len(ops):
            op = ops[pc]
            o, a = op[0], op[1:]
            pc += 1
            if o == "label":
                continue
            if o == "call":
                exec_routine(a[0])
                continue
            executed[0] += 1
            if o in ("jz", "jnz", "jmp"):
                if o == "jmp" or (o == "jz" and st["zf"]) or (o == "jnz" and not st["zf"]):
                    pc = labels[a[0]]
            elif o == "ld0": st["x0"] = fget(-a[0])
            elif o == "ld1": st["x1"] = fget(-a[0])
            elif o == "st0": mem[-a[0]] = d2b(st["x0"])
            elif o == "st1": mem[-a[0]] = d2b(st["x1"])
            elif o == "ld0c": st["x0"] = K[index[a[0]]][1]
            elif o == "ld1c": st["x1"] = K[index[a[0]]][1]
            elif o in ("add", "sub") and a:
                v = reg[a[0]] + reg[a[1]] if o == "add" else reg[a[0]] - reg[a[1]]
                reg[a[0]] = v & MASK
            elif o == "add": st["x0"] = st["x0"] + st["x1"]
            elif o == "sub": st["x0"] = st["x0"] - st["x1"]
            elif o == "mul": st["x0"] = st["x0"] * st["x1"]
            elif o == "div": st["x0"] = st["x0"] / st["x1"]
            elif o == "ucomisd":
                un = st["x0"] != st["x0"] or st["x1"] != st["x1"]
                st["zf"] = 1 if (un or st["x0"] == st["x1"]) else 0
                st["cf"] = 1 if (un or st["x0"] < st["x1"]) else 0
            elif o == "cvttsd2si": reg["rax"] = int(math.trunc(st["x0"])) & MASK
            elif o == "cvtsi2sd": st["x0"] = float(s64(reg["rax"]))
            elif o == "movq_rax_x0": reg["rax"] = d2b(st["x0"])
            elif o == "movq_x0_rax": st["x0"] = b2d(reg["rax"])
            elif o == "movq_x1_rax": st["x1"] = b2d(reg["rax"])
            elif o == "load": reg[a[0]] = mem.get(-a[1], 0)
            elif o == "store": mem[-a[1]] = reg[a[0]] & MASK
            elif o == "movi": reg[a[0]] = a[1] & MASK
            elif o == "mov": reg[a[0]] = reg[a[1]]
            elif o == "and": reg["rax"] &= reg["rbx"]
            elif o == "or": reg["rax"] |= reg["rbx"]
            elif o == "xor": reg["rax"] ^= reg["rbx"]
            elif o == "shl_cl": reg["rax"] = (reg["rax"] << (reg["rcx"] & 63)) & MASK
            elif o == "shr_cl": reg["rax"] = reg["rax"] >> (reg["rcx"] & 63)
            elif o == "sar_cl": reg["rax"] = (s64(reg["rax"]) >> (reg["rcx"] & 63)) & MASK
            elif o == "imul": reg[a[0]] = (s64(reg[a[0]]) * s64(reg[a[1]])) & MASK
            elif o == "cqo_idiv":
                n_, d_ = s64(reg["rax"]), s64(reg["rcx"])
                q = abs(n_) // abs(d_)
                q = q if (n_ >= 0) == (d_ > 0) else -q
                reg["rax"], reg["rdx"] = q & MASK, (n_ - q * d_) & MASK
            elif o == "cmp_rbx_rax":
                st["ca"], st["cb"] = s64(reg["rbx"]), s64(reg["rax"])
                st["zf"] = 1 if reg["rbx"] == reg["rax"] else 0
                st["cf"] = 1 if reg["rbx"] < reg["rax"] else 0
            elif o == "setl": st["al"] = int(st["ca"] < st["cb"])
            elif o == "setg": st["al"] = int(st["ca"] > st["cb"])
            elif o == "setle": st["al"] = int(st["ca"] <= st["cb"])
            elif o == "setge": st["al"] = int(st["ca"] >= st["cb"])
            elif o == "sete": st["al"] = st["zf"]
            elif o == "setne": st["al"] = 1 - st["zf"]
            elif o == "setb": st["al"] = st["cf"]
            elif o == "setae": st["al"] = 1 - st["cf"]
            elif o == "seta": st["al"] = int(not st["cf"] and not st["zf"])
            elif o == "test_al": st["zf"] = 1 if st["al"] == 0 else 0
            elif o == "test_rax": st["zf"] = 1 if reg["rax"] == 0 else 0
            elif o == "lea_rax_rbp": reg["rax"] = (-a[0]) & MASK
            elif o == "mov_rdx_rax": reg["rdx"] = reg["rax"]
            elif o == "mov_rbx_rax": reg["rbx"] = reg["rax"]
            elif o == "load_idx": reg["rax"] = mem.get(s64(reg["rdx"]) + 8 * s64(reg["rbx"]), 0)
            elif o == "store_idx": mem[s64(reg["rdx"]) + 8 * s64(reg["rbx"])] = reg["rax"]
            elif o == "ldtab":
                reg["rbx"] = reg["rax"]
                st["x0"] = K[index[a[0]] + s64(reg["rbx"])][1]
                reg["rax"] = d2b(st["x0"])
            else:
                raise ValueError("interpreter has no %r" % (o,))

    exec_routine(which)
    return mem.get(-OUT, 0), executed[0]


# ---------------------------------------------------------------------------
# Sounio rendering: the emitter itself
# ---------------------------------------------------------------------------
SIO_REG = {"rax": 0, "rcx": 1, "rdx": 2, "rbx": 3}

SIO_PREAMBLE = """\
// self-hosted::native::math_fdlibm_trig -- GENERATED, DO NOT EDIT.
//
// Regenerate with:  python3 scripts/dev/gen_fdlibm_trig.py
// Source of truth:  scripts/research/fdlibm_sin_cos_oracle.py (the instruction list)
//
// The native sin and cos builtins: fdlibm (OpenLibm s_sin.c, s_cos.c,
// e_rem_pio2.c, k_rem_pio2.c with prec 1, k_sin.c, k_cos.c), operation for
// operation, spelled in this backend's primitive set -- two xmm registers,
// rbp-relative 8-byte slots, rax/rbx/rcx/rdx, rip-relative constants and
// [rdx+rbx*8] loads for the k_rem_pio2 arrays.
//
// Every line below is printed from one instruction list. The same list is run
// by a Python interpreter and, rendered as C, diffed bit for bit against
// OpenLibm; nothing here was transliterated by hand, so the bytes cannot drift
// from the model that was measured.
//
// CONTIGUITY. Every constant is registered as one run of consecutive
// data_section_add_f64 calls and only the first offset, k, is captured:
// constant i is read at k + 8*i, and the ipio2 / PIo2 tables are indexed at run
// time from inside that run. Registering anything between two of these calls
// breaks every read after it without any diagnostic.
//
// GPRs clobbered: rax, rbx, rcx, rdx.

use native::encode::*
use native::elf::{data_section_add_f64}
use native::reloc::{add_rip_reloc, patch_u32_le}
use native::codegen_x86_linux::{NativeCompiler}

// xmm0 <- .rodata[off]. The displacement is written as zero and the real
// offset registered as a relocation: .rodata is not placed yet.
fn fdt_ld0c(nc: NativeCompiler, off: i64) -> NativeCompiler with Mut, Panic, Div {
    var c = nc
    let pos = c.code.len
    c.code = emit_movsd_xmm0_rip_disp32(c.code, 0)
    c.relocs = add_rip_reloc(c.relocs, pos + 4, off)
    c
}

// xmm1 <- .rodata[off]. Same relocation contract as fdt_ld0c.
fn fdt_ld1c(nc: NativeCompiler, off: i64) -> NativeCompiler with Mut, Panic, Div {
    var c = nc
    let pos = c.code.len
    c.code = emit_movsd_xmm1_rip_disp32(c.code, 0)
    c.relocs = add_rip_reloc(c.relocs, pos + 4, off)
    c
}

// xmm0 <- .rodata[off + 8*rax]. On entry rax holds the index; clobbers rax, rbx.
fn fdt_ld0_table(nc: NativeCompiler, off: i64) -> NativeCompiler with Mut, Panic, Div {
    var c = nc
    c.code = emit_mov_rbx_rax(c.code)
    let pos = c.code.len
    c.code = emit_lea_rax_rip_disp32(c.code, 0)
    c.relocs = add_rip_reloc(c.relocs, pos + 3, off)
    c.code = emit_mov_rax_mem_rax_rbx8(c.code)
    c.code = emit_movq_xmm0_rax(c.code)
    c
}
"""

ROUTINE_DOC = {
    "k_sin0": "k_sin.c, iy == 0: x + v*(S1 + z*r). Reads KX, writes KOUT.",
    "k_sin1": "k_sin.c, iy == 1: x - ((z*(y/2 - v*r) - y) - v*S1). Reads KX, KY, writes KOUT.",
    "k_cos": "k_cos.c: w + (((1 - w) - hz) + (z*r - x*y)). Reads KX, KY, writes KOUT.",
    "k_rem_pio2": "k_rem_pio2.c, prec 1 (jk = jp = 4): tx0..tx2, e0, nx -> n, y0, y1.",
    "rem_pio2": "e_rem_pio2.c: x -> n, y0, y1 (small multiples, medium, inf/NaN, large).",
    "sin": "s_sin.c.",
    "cos": "s_cos.c.",
}


def sio_slot(n):
    return "0 - %d" % n


def sio_op(o, a, index):
    if o == "ld0": return ["c.code = emit_movsd_xmm0_rbp_disp32(c.code, %s)" % sio_slot(a[0])]
    if o == "ld1": return ["c.code = emit_movsd_xmm1_rbp_disp32(c.code, %s)" % sio_slot(a[0])]
    if o == "st0": return ["c.code = emit_movsd_rbp_disp32_xmm0(c.code, %s)" % sio_slot(a[0])]
    if o == "st1": return ["c.code = emit_movsd_rbp_disp32_xmm1(c.code, %s)" % sio_slot(a[0])]
    if o == "ld0c": return ["c = fdt_ld0c(c, k + %d)   // %s" % (8 * index[a[0]], a[0])]
    if o == "ld1c": return ["c = fdt_ld1c(c, k + %d)   // %s" % (8 * index[a[0]], a[0])]
    if o in ("add", "sub") and a:
        return ["c.code = emit_%s_reg_reg(c.code, %d, %d)" % (o, SIO_REG[a[0]], SIO_REG[a[1]])]
    if o in ("add", "sub", "mul", "div"): return ["c.code = emit_%ssd_xmm0_xmm1(c.code)" % o]
    if o == "ucomisd": return ["c.code = emit_ucomisd_xmm0_xmm1(c.code)"]
    if o == "cvttsd2si": return ["c.code = emit_cvttsd2si_rax_xmm0(c.code)"]
    if o == "cvtsi2sd": return ["c.code = emit_cvtsi2sd_xmm0_rax(c.code)"]
    if o == "movq_rax_x0": return ["c.code = emit_movq_rax_xmm0(c.code)"]
    if o == "movq_x0_rax": return ["c.code = emit_movq_xmm0_rax(c.code)"]
    if o == "movq_x1_rax": return ["c.code = emit_movq_xmm1_rax(c.code)"]
    if o == "load": return ["c.code = emit_load_rbp_disp32_%s(c.code, %s)" % (a[0], sio_slot(a[1]))]
    if o == "store":
        if a[0] == "rax":
            return ["c.code = emit_store_rax_rbp_disp32(c.code, %s)" % sio_slot(a[1])]
        return ["c.code = emit_store_reg_rbp_disp32(c.code, %s, %d)" % (sio_slot(a[1]), SIO_REG[a[0]])]
    if o == "movi":
        v = a[1]
        lit = ("0x%X" % v if v > 9999 else str(v)) if v >= 0 else "0 - %d" % -v
        return ["c.code = emit_mov_reg_imm(c.code, %d, %s)" % (SIO_REG[a[0]], lit)]
    if o == "mov": return ["c.code = emit_mov_reg_reg(c.code, %d, %d)" % (SIO_REG[a[0]], SIO_REG[a[1]])]
    if o in ("and", "or", "xor"): return ["c.code = emit_%s_rax_rbx(c.code)" % o]
    if o in ("shl_cl", "shr_cl", "sar_cl"): return ["c.code = emit_%s_rax_cl(c.code)" % o[:3]]
    if o == "imul": return ["c.code = emit_imul_reg_reg(c.code, %d, %d)" % (SIO_REG[a[0]], SIO_REG[a[1]])]
    if o == "cqo_idiv": return ["c.code = emit_cqo(c.code)", "c.code = emit_idiv_rcx(c.code)"]
    if o == "cmp_rbx_rax": return ["c.code = emit_cmp_rbx_rax(c.code)"]
    if o in ("setl", "setg", "setle", "setge", "sete", "setne", "setb", "setae", "seta"):
        return ["c.code = emit_%s_al(c.code)" % o]
    if o == "test_al": return ["c.code = emit_test_al_al(c.code)"]
    if o == "test_rax": return ["c.code = emit_test_rax_rax(c.code)"]
    if o == "lea_rax_rbp": return ["c.code = emit_lea_rax_rbp_disp32(c.code, %s)" % sio_slot(a[0])]
    if o == "mov_rdx_rax": return ["c.code = emit_mov_rdx_rax(c.code)"]
    if o == "mov_rbx_rax": return ["c.code = emit_mov_rbx_rax(c.code)"]
    if o == "load_idx": return ["c.code = emit_load_rax_mem_rdx_rbx8(c.code)"]
    if o == "store_idx": return ["c.code = emit_store_rax_mem_rdx_rbx8(c.code)"]
    if o == "ldtab": return ["c = fdt_ld0_table(c, k + %d)   // %s[rax]" % (8 * index[a[0]], a[0].rstrip("0").rstrip("_"))]
    raise ValueError("no Sounio rendering for %r" % (o,))


FRAME_BYTES = 1280


def render_sio(prog, K, index):
    assert FRAME_SLOTS_END <= FRAME_BYTES
    L = [SIO_PREAMBLE]
    for name, r in prog.items():
        referenced = {op[1] for op in r.ops if op[0] in ("jz", "jnz", "jmp")}
        L.append("// %s" % ROUTINE_DOC[name])
        L.append("fn fdt_%s(nc: NativeCompiler, k: i64) -> NativeCompiler with Mut, Panic, Div {" % name)
        L.append("    var c = nc")
        jumps = []
        placed = set()
        for op in r.ops:
            o, a = op[0], op[1:]
            if o == "label":
                if a[0] in referenced:
                    L.append("    let lab_%s = c.code.len" % a[0])
                placed.add(a[0])
            elif o in ("jz", "jnz", "jmp"):
                jn = len(jumps)
                jumps.append((jn, o, a[0]))
                L.append("    let j%d = c.code.len" % jn)
                L.append("    c.code = emit_%s_rel32(c.code)" % o)
            elif o == "call":
                L.append("    c = fdt_%s(c, k)" % a[0])
            else:
                L.extend("    " + s for s in sio_op(o, a, index))
        for jn, o, lab in jumps:
            assert lab in placed, (name, lab)
            if o == "jmp":
                L.append("    c.code = patch_u32_le(c.code, j%d + 1, lab_%s - (j%d + 5))" % (jn, lab, jn))
            else:
                L.append("    c.code = patch_u32_le(c.code, j%d + 2, lab_%s - (j%d + 6))" % (jn, lab, jn))
        L.append("    c")
        L.append("}")
        L.append("")
    for which in ("sin", "cos"):
        L.append("// %s(x) -- the native %s builtin. The f64 argument arrives in rdi as raw bits" % (which, which))
        L.append("// and the result leaves in rax the same way, like every scalar builtin here.")
        L.append("// Frame: [rbp-8] x, [rbp-%d] the result, %d bytes in all." % (OUT, FRAME_BYTES))
        L.append("pub fn emit_builtin_fdlibm_%s(nc: NativeCompiler) -> NativeCompiler with Mut, Panic, Div {" % which)
        L.append("    var c = nc")
        for i, (nm, v) in enumerate(K):
            L.append("    c.rodata = data_section_add_f64(c.rodata, 0x%016X)   // [%d] %s" % (d2b(v), i, nm))
            if i == 0:
                L.append("    let k = c.rodata.last_offset")
        L.append("    c.code = emit_push_rbp(c.code)")
        L.append("    c.code = emit_mov_rbp_rsp(c.code)")
        L.append("    c.code = emit_sub_rsp_imm32(c.code, %d)" % FRAME_BYTES)
        L.append("    c.code = emit_mov_reg_reg(c.code, 0, 7)              // rax = rdi = bits(x)")
        L.append("    c.code = emit_store_rax_rbp_disp32(c.code, 0 - 8)")
        L.append("    c = fdt_%s(c, k)" % which)
        L.append("    c.code = emit_movsd_xmm0_rbp_disp32(c.code, %s)" % sio_slot(OUT))
        L.append("    c.code = emit_movq_rax_xmm0(c.code)")
        L.append("    c.code = emit_add_rsp_imm32(c.code, %d)" % FRAME_BYTES)
        L.append("    c.code = emit_pop_rbp(c.code)")
        L.append("    c.code = emit_ret(c.code)")
        L.append("    c")
        L.append("}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------
def static_counts(prog):
    def count(name, seen=()):
        n = 0
        for op in prog[name].ops:
            if op[0] == "label":
                continue
            if op[0] == "call":
                n += count(op[1])
            else:
                n += 1
        return n
    return {nm: count(nm) for nm in prog}



def rip_loads(prog, name):
    n = 0
    for op in prog[name].ops:
        if op[0] == "call":
            n += rip_loads(prog, op[1])
        elif op[0] in ("ld0c", "ld1c", "ldtab"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
BRANCH_HIGH_WORDS = (0x3E46A09E, 0x3E500000, 0x3FE921FB, 0x3FE921FC, 0x3FF921FB, 0x4002D97C,
                     0x400921FB, 0x400F6A7A, 0x4012D97C, 0x4015FDBC, 0x401921FB, 0x401C463B,
                     0x413921FB, 0x7FEFFFFF)
SPECIALS = (0, 1 << 63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000,
            1, (1 << 63) | 1, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x7FEFFFFFFFFFFFFF,
            0xFFEFFFFFFFFFFFFF)


def reference_corpus(nrandom=200000, sweep=400, multiples=400, seed=20260914):
    """specials, every high-word branch threshold +-1, multiples of pi/2 +-2, the
    worst cases, random doubles in four bands, and a sweep of every exponent 20..1023."""
    xs = list(SPECIALS)
    for hw in BRANCH_HIGH_WORDS:
        for low in (0, 1, 0x54442D18, 0xFFFFFFFF):
            for sign in (0, 1 << 63):
                u = sign | (hw << 32) | low
                xs += [u, (u - 1) & MASK, u + 1]
    for k in range(1, multiples + 1):
        u = d2b(k * math.pi / 2)
        for d in (-2, -1, 0, 1, 2):
            xs += [u + d, (1 << 63) | (u + d)]
    for u, _ in WORST_CASES:
        xs += [u, (1 << 63) | u]
    rng = random.Random(seed)
    for _ in range(nrandom):
        band = rng.random()
        if band < 0.25:
            x = rng.uniform(-8.0, 8.0)
        elif band < 0.5:
            x = rng.uniform(-2e6, 2e6)
        elif band < 0.8:
            x = rng.choice((-1, 1)) * 2.0 ** rng.uniform(20, 1023)
        else:
            x = b2d(rng.getrandbits(64))
        xs.append(d2b(x))
    for E in range(20, 1024):
        for _ in range(sweep):
            m = rng.getrandbits(52) | (1 << 52)
            u = d2b(m * 2.0 ** (E - 52))
            xs += [u, (1 << 63) | u]
    return [u & MASK for u in xs]


def search_worst_cases(top):
    """Per binary exponent, the double closest to a multiple of pi/2 (needs mpmath)."""
    from mpmath import mp, mpf, pi, floor, nint, log
    lo, hi = 2 ** 52, 2 ** 53
    found = []
    for E in range(-1, 1024):
        mp.prec = 1200 + max(0, E)
        alpha = mpf(2) ** (E - 52) * 2 / pi
        a = alpha
        p0, q0, p1, q1 = 0, 1, 1, 0
        cands = set()
        for _ in range(200):
            ai = int(floor(a))
            frac = a - ai
            p2, q2 = ai * p1 + p0, ai * q1 + q0
            for t in range(1, min(ai, 64) + 1):
                q = t * q1 + q0
                for mult in (1, 2, 3):
                    m = q * mult
                    if lo <= m < hi:
                        cands.add(m)
            for mult in range(1, 64):
                m = q2 * mult
                if m >= hi:
                    break
                if m >= lo:
                    cands.add(m)
            if q2 >= hi or frac == 0:
                break
            p0, q0, p1, q1 = p1, q1, p2, q2
            a = 1 / frac
        best = None
        for m in cands:
            d = abs(m * alpha - nint(m * alpha))
            if best is None or d < best[0]:
                best = (d, m)
        if best is None:
            continue
        d, m = best
        bits = float(-log(d * pi / 2, 2))
        found.append((bits, d2b(float(mpf(m) * mpf(2) ** (E - 52)))))
    found.sort(reverse=True)
    return [(u, b) for b, u in found[:top]]


# ---------------------------------------------------------------------------
# bit-identity against OpenLibm
# ---------------------------------------------------------------------------
DRIVER_C = r"""#include <stdio.h>
#include <stdint.h>
#include <string.h>
extern double sin(double); extern double cos(double);
int main(void) { char line[64]; uint64_t u;
  while (fgets(line, sizeof line, stdin)) { if (sscanf(line, "%lx", &u) != 1) continue;
    double x, s, c; uint64_t sb, cb; memcpy(&x, &u, 8); s = sin(x); c = cos(x);
    memcpy(&sb, &s, 8); memcpy(&cb, &c, 8); printf("%016lx %016lx %016lx\n", u, sb, cb); }
  return 0; }
"""
OPENLIBM_SOURCES = ("s_sin.c", "s_cos.c", "k_sin.c", "k_cos.c", "k_rem_pio2.c", "e_rem_pio2.c",
                    "s_scalbn.c", "s_floor.c")


def against_openlibm(olm, work, corpus):
    os.makedirs(work, exist_ok=True)
    assert load_ipio2(os.path.join(olm, "src", "k_rem_pio2.c")) == IPIO2, "ipio2 table differs from OpenLibm"
    for nm, dec, _ in FDLIBM_DEC:
        text = "".join(open(os.path.join(olm, "src", f)).read() for f in ("k_sin.c", "k_cos.c", "e_rem_pio2.c"))
        m = re.search(r"\b%s\s*=\s*([-+]?[0-9.]+e[-+]?[0-9]+)" % re.escape(nm.lower() if nm.startswith("PIO2") or nm == "INVPIO2" else nm), text)
        assert m and float(m.group(1)) == float(dec), "constant %s differs from OpenLibm" % nm
    K, index = build_constants(IPIO2)
    drv, ref, prog_c, prog_bin = (os.path.join(work, f) for f in ("drv.c", "openlibm_ref", "oracle_prog.c", "oracle_prog"))
    open(drv, "w").write(DRIVER_C)
    flags = ["-O2", "-fno-builtin", "-ffp-contract=off", "-fno-fast-math", "-std=gnu99",
             "-I" + os.path.join(olm, "include"), "-I" + os.path.join(olm, "src"), "-I" + os.path.join(olm, "amd64")]
    subprocess.run(["gcc"] + flags + ["-o", ref, drv] + [os.path.join(olm, "src", f) for f in OPENLIBM_SOURCES], check=True)
    open(prog_c, "w").write(render_c(build_program(), K, index))
    subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-fno-fast-math", "-o", prog_bin, prog_c, "-lm"], check=True)
    inp = "\n".join("%016x" % u for u in corpus) + "\n"
    a = subprocess.run([ref], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
    b = subprocess.run([prog_bin], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
    mism = [(x, y) for x, y in zip(a, b) if x != y]
    ok = len(a) == len(b) == len(corpus) and not mism
    print("against OpenLibm: %d inputs, %d results each side, %d differing" % (len(corpus), len(a), len(mism) + abs(len(a) - len(b))))
    for x, y in mism[:5]:
        print("  openlibm %s\n  oracle   %s" % (x, y))
    return ok


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="fdlibm sin/cos instruction list for the native emitter")
    ap.add_argument("--emit-c", metavar="OUT")
    ap.add_argument("--against-openlibm", metavar="DIR")
    ap.add_argument("--workdir", default=os.path.join("/tmp", "fdlibm_sin_cos_oracle"))
    ap.add_argument("--random", type=int, default=200000)
    ap.add_argument("--sweep", type=int, default=400)
    ap.add_argument("--search-worst-cases", type=int, metavar="N")
    args = ap.parse_args(argv[1:])

    K, index = build_constants(IPIO2)
    prog = build_program()
    print("constants %d, frame %d slots (%d bytes of %d)" % (len(K), FRAME_SLOTS_END // 8, FRAME_SLOTS_END, FRAME_BYTES))
    for nm, n in static_counts(prog).items():
        print("  %-11s %5d instructions with calls inlined" % (nm, n))
    for nm in ("sin", "cos"):
        print("  %s: %d rip-relative loads" % (nm, rip_loads(prog, nm)))
    worst_n = 0
    for u, _ in WORST_CASES[:8]:
        for which in ("sin", "cos"):
            worst_n = max(worst_n, run(prog, K, index, which, u)[1])
    print("  executed on the worst cases: up to %d instructions per call" % worst_n)

    if args.search_worst_cases:
        for u, b in search_worst_cases(args.search_worst_cases):
            print("0x%016X %.2f" % (u, b))
    if args.emit_c:
        open(args.emit_c, "w").write(render_c(prog, K, index))
        print("wrote", args.emit_c)
    if args.against_openlibm:
        corpus = reference_corpus(nrandom=args.random, sweep=args.sweep)
        return 0 if against_openlibm(args.against_openlibm, args.workdir, corpus) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
