#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================================
#  ffi_pow_oracle.py  --  op-for-op model of the x86-64 emitter for native pow/tgamma
#
#  Companion to  docs/audit/NATIVE_POW_TGAMMA_NUMERICAL_DESIGN_2026-08-23.md
#
#  Every floating-point value in these routines lives in xmm0, xmm1, or a numbered
#  stack slot.  Every arithmetic step is one of the emitter's actual instructions.
#  The Machine class below is the ONLY place a Python float operator appears; the
#  routines cannot accidentally use an operation the hardware does not have.
#
#  -----------------------------------------------------------------------------------
#  INSTRUCTION COUNTS  (dynamic, i.e. instructions *executed*.  Polynomial evaluation is
#  unrolled, so static == dynamic everywhere except the pow integer fast path, which is
#  a loop.  Reprinted, measured, by section [2] of the run.)
#
#      routine                                    instr   peak 8-byte stack slots
#      ----------------------------------------  ------   -----------------------
#      TwoSum(a,b)                                   21             2
#      QuickTwoSum(a,b)                              11             1
#      TwoProd(a,b)   Dekker split, SPLIT=2^27+1     58             6
#      log2_dd(x)                                   709            35
#      exp2_core(f)                                 195            14
#      sinpi(x)                                     170            18
#      pow(x,y)    general path                    1081            42
#      pow(x,y)    exactness fast path (10^3)       285            15
#      pow(x,y)    special-case row                  60             5
#      tgamma(x)   primary branch  x >= 1/2        1598            48
#      tgamma(x)   recurrence      |x| < 1/2       1746            49
#      tgamma(x)   reflection      x <= -1/2       2052            49
#      tgamma(x)   integer table row                 44             3
#      tgamma(x)   special-case row                  10             2
#
#  So: pow is ~1.1k instructions and tgamma ~1.6-2.1k on the numeric path, a few tens
#  on every special case.  Both are emittable.  The stack frames are NOT: see finding
#  F1 below.
#
#  -----------------------------------------------------------------------------------
#  MEASURED ACCURACY  (>= 300k random samples per function, 60-digit mpmath reference,
#  bands as in design 1.3 / 2.5 plus a subnormal-output band for pow)
#
#      pow      <= 0.76 ulp     (design claims <= 2;  design measured 1.45)
#      tgamma   <= 4.15 ulp     (design claims <= 5;  design measured 3.13 / 4.35)
#
#  Measured against libm instead, the same runs read pow <= 1 ulp and tgamma <= 7 ulp,
#  because glibc's own tgamma is up to 5.4 ulp off over these bands.  See finding F8.
#
#  All 52 pow and 44 tgamma reference vectors of design PART 3 pass at the category
#  each is assigned in design 3.3, including every bit-exact row.
#
#  -----------------------------------------------------------------------------------
#  FINDINGS -- what in the design turned out to be wrong or unimplementable with the
#  listed primitives.  (The numbers -- coefficients, thresholds, error budgets -- all
#  held up.  Everything below is about the OP SEQUENCES.)
#
#  F1. rbp-disp8 is too small a frame.  Every kernel here is inlined (there is no call
#      instruction in the primitive set), and with only xmm0/xmm1 every value that is
#      live across one operation must be spilled.  Peak frame: 42 slots for pow, 49 for
#      tgamma.  rbp-disp8 addresses 32 doubles.  The emitter needs rbp-disp32 (or a
#      real call/frame discipline), not disp8.
#
#  F2. design 1.6's listing cannot be emitted as written: it uses xmm2, xmm3, xmm4 and
#      xmm5.  The odd-integer test is correct, but on a two-register machine it costs
#      22 instructions and two stack slots, not the 9 shown.
#
#  F3. No setcc in the listed set expresses ORDERED equality.  `ucomisd; sete` sets AL
#      for unordered as well as equal (ZF=1 in both), so design 1.5 rows 1 and 2 --
#      whose whole point is that they must fire for a NaN operand and NOT be confused
#      with row 3 -- cannot be written with ucomisd+sete.  They are written here as
#      `movq rax,xmm; cmp rax, imm` integer compares.  (`setnp` would also do it, but
#      parity setcc is not in the stated set.)  Likewise NaN detection uses
#      `ucomisd x,x; setb` (CF=1 iff unordered), which IS in the set.
#
#  F4. design 0.1's TwoProd domain bound bites in a place the design says it does not.
#      "both call sites here are far inside 2^996" is true for the two sites the design
#      names, but the tgamma recurrence branch Gamma(x) = Gamma(x+1)/x has a third: for
#      x = 1e-300 the double-double quotient's leading term is 3.2e301, and SPLIT*that
#      overflows.  Fixed here by dividing out x's EXPONENT (split_pow2) instead of x,
#      which also removes the spurious overflow for subnormal x.  Without it,
#      tgamma(1e-300) -- design 3.2 row 20 -- returns NaN.
#
#  F5. pow needs a magnitude guard on y that the design does not mention.  TwoProd(y,
#      L_hi) breaks for |y| >= 2^996, and y = 1e300 (design 3.1 row 50) is a legal
#      input.  Guard used here: |y| >= 2^63 implies |r| >= 1448 for every x != +-1, so
#      the answer is a pure overflow/underflow decision and the multiply is skipped.
#
#  F6. design 2.3's sinpi reduction to |f| <= 1/2 is not a 1-ulp routine.  Measured
#      3.07 ulp as specified (n = round(x); f = x-n; odd poly on [-1/2,1/2]).  Two
#      things are wrong at |f| ~ 1/2: t = pi*f must be a double-double (a rounded pi
#      costs 0.35 ulp outright) and, dominantly, the polynomial variable v = t^2 has
#      |v| = 2.47 there, so ulp(v)/2 alone moves the answer 0.52 ulp, with the
#      2*t_hi*t_lo cross term worth another 0.75.  Reducing to QUARTER integers
#      (n = round(2x), |h| <= 1/4, quadrant selects sin or cos) drops it to 1.05 ulp.
#      The design's own error budget assumes sinpi is 1 ulp, so this is load-bearing.
#
#  F7. design 2.3's "m = (sqrt(2pi) * A) * exp2(f)" in plain double throws away ~1 ulp
#      for nothing: it rounds three times where once will do.  Carrying sqrt(2pi)*A
#      (TwoProd) and exp2 (its dd form) and rounding only at the end took the primary
#      branch from 3.78 to 2.79 ulp on [0.5,1.5].
#
#  F8. libm is not a valid reference for the tgamma claim.  glibc's tgamma is measured
#      here at up to 5.16 ulp against mpmath over the design's own bands -- larger than
#      the 5-ulp figure being claimed -- and it is 1 ulp off on tgamma(20), tgamma(22)
#      and tgamma(23), all of which ARE exactly representable factorials.  A harness
#      that asserts agreement with libm on design 2.4 row 8 fails on a correct
#      implementation.  glibc pow, by contrast, is correctly rounded and usable.
#
#  F9. design 2.4 row 9 needs a magnitude cut-off that the table does not state.  The
#      reflection is only evaluable while Gamma(-x) fits an exponent; for x below about
#      -185 the recurrence argument runs off and the answer is a signed zero whose sign
#      is (-1)^(n+1) on (-n-1,-n).  Implemented as an explicit branch at x <= -185.
#      x = -184.5 (row 27) still goes through the real reflection and underflows to
#      -0.0 on its own, which is the stronger test.
#
# =====================================================================================

import ctypes, ctypes.util, math, random, struct, sys
from fractions import Fraction

# -------------------------------------------------------------------------------------
# 0.  bit helpers (host-side only -- not part of the modelled machine)
# -------------------------------------------------------------------------------------
def d2b(x):  return struct.unpack('<Q', struct.pack('<d', x))[0]
def b2d(u):  return struct.unpack('<d', struct.pack('<Q', u & 0xFFFFFFFFFFFFFFFF))[0]

INF  = float('inf')
NAN  = float('nan')
SIGN = 0x8000000000000000
MASK = 0x7FFFFFFFFFFFFFFF

# -------------------------------------------------------------------------------------
# 1.  The machine
# -------------------------------------------------------------------------------------
class Machine(object):
    """Two xmm registers, a stack of 8-byte slots, a handful of GPRs, an instruction
    counter.  Methods are named after the instructions they model."""

    NSLOT = 512

    def __init__(self):
        self.mem  = [0.0] * self.NSLOT
        self.sp   = 0                 # next free slot
        self.peak = 0
        self.x0   = 0.0
        self.x1   = 0.0
        self.r    = {'rax': 0, 'rcx': 0, 'rdx': 0, 'rsi': 0, 'rdi': 0,
                     'r8': 0, 'r9': 0, 'r10': 0, 'r11': 0}
        self.al   = 0
        self.zf   = 0                 # from ucomisd
        self.cf   = 0
        self.n    = 0                 # instruction count
        self.const = {}

    # --- slot allocation --------------------------------------------------------------
    def mark(self):
        return self.sp

    def release(self, m):
        self.sp = m

    def alloc(self, k=1):
        b = self.sp
        self.sp += k
        if self.sp > self.peak:
            self.peak = self.sp
        if self.sp > self.NSLOT:
            raise RuntimeError('stack frame overflow')
        return b if k == 1 else tuple(range(b, b + k))

    # --- moves ------------------------------------------------------------------------
    def ld0(self, s):   self.n += 1; self.x0 = self.mem[s]           # movsd xmm0,[rbp-d]
    def ld1(self, s):   self.n += 1; self.x1 = self.mem[s]           # movsd xmm1,[rbp-d]
    def st0(self, s):   self.n += 1; self.mem[s] = self.x0           # movsd [rbp-d],xmm0
    def st1(self, s):   self.n += 1; self.mem[s] = self.x1           # movsd [rbp-d],xmm1
    def ld0c(self, k):  self.n += 1; self.x0 = self.const[k]         # movsd xmm0,[rip+K]
    def ld1c(self, k):  self.n += 1; self.x1 = self.const[k]         # movsd xmm1,[rip+K]
    def ld0t(self, k, i):                                            # movsd xmm0,[rip+T+rax*8]
        self.n += 1; self.x0 = self.const[k][i]
    def ld1t(self, k, i):
        self.n += 1; self.x1 = self.const[k][i]

    def movq_rax_x0(self):  self.n += 1; self.r['rax'] = d2b(self.x0)
    def movq_rax_x1(self):  self.n += 1; self.r['rax'] = d2b(self.x1)
    def movq_x0_rax(self):  self.n += 1; self.x0 = b2d(self.r['rax'])
    def movq_x1_rax(self):  self.n += 1; self.x1 = b2d(self.r['rax'])

    # --- scalar double arithmetic:  xmm0 <op> xmm1 -> xmm0 -----------------------------
    def addsd(self):
        self.n += 1
        self.x0 = self._fin(self.x0 + self.x1)
    def subsd(self):
        self.n += 1
        self.x0 = self._fin(self.x0 - self.x1)
    def mulsd(self):
        self.n += 1
        a, b = self.x0, self.x1
        if (a == 0.0 and math.isinf(b)) or (b == 0.0 and math.isinf(a)):
            self.x0 = NAN
        else:
            self.x0 = self._fin(a * b)
    def divsd(self):
        self.n += 1
        a, b = self.x0, self.x1
        if b != b or a != a:
            self.x0 = NAN
        elif b == 0.0:
            if a == 0.0:
                self.x0 = NAN
            else:
                s = (d2b(a) ^ d2b(b)) & SIGN
                self.x0 = -INF if s else INF
        elif math.isinf(a) and math.isinf(b):
            self.x0 = NAN
        else:
            self.x0 = self._fin(a / b)
    def sqrtsd(self):
        self.n += 1
        a = self.x0
        self.x0 = NAN if (a < 0.0) else (a if (a != a or a == 0.0) else math.sqrt(a))

    @staticmethod
    def _fin(v):
        # CPython float ops already round to binary64 and yield inf on overflow.
        return v

    # --- roundsd xmm0, xmm0, imm8 ------------------------------------------------------
    def roundsd(self, mode):
        """mode: 0 nearest-even, 1 floor, 2 ceil, 3 trunc (|0x08 suppresses inexact)."""
        self.n += 1
        x = self.x0
        if x != x or math.isinf(x) or x == 0.0:
            return
        if abs(x) >= 4503599627370496.0:      # 2^52: already integral
            return
        m = mode & 3
        if   m == 0: r = float(round(x))      # Python round() is round-half-to-even
        elif m == 1: r = float(math.floor(x))
        elif m == 2: r = float(math.ceil(x))
        else:        r = float(math.trunc(x))
        if r == 0.0 and (d2b(x) & SIGN):
            r = -0.0                          # roundsd preserves the sign of zero
        self.x0 = r

    # --- conversions -------------------------------------------------------------------
    def cvtsi2sd_x0(self, reg='rax'):
        self.n += 1
        v = self.r[reg]
        if v >= (1 << 63): v -= (1 << 64)
        self.x0 = float(v)
    def cvttsd2si(self, reg='rax'):
        self.n += 1
        x = self.x0
        if x != x or math.isinf(x) or abs(x) >= 9.223372036854776e18:
            self.r[reg] = 1 << 63
        else:
            self.r[reg] = int(math.trunc(x)) & 0xFFFFFFFFFFFFFFFF

    # --- comparisons -------------------------------------------------------------------
    def ucomisd(self):
        self.n += 1
        a, b = self.x0, self.x1
        if a != a or b != b:
            self.zf, self.cf = 1, 1           # unordered: ZF=PF=CF=1
        elif a > b:
            self.zf, self.cf = 0, 0
        elif a < b:
            self.zf, self.cf = 0, 1
        else:
            self.zf, self.cf = 1, 0

    def seta(self):   self.n += 1; self.al = 1 if (self.cf == 0 and self.zf == 0) else 0
    def setae(self):  self.n += 1; self.al = 1 if (self.cf == 0) else 0
    def setb(self):   self.n += 1; self.al = 1 if (self.cf == 1) else 0
    def setbe(self):  self.n += 1; self.al = 1 if (self.cf == 1 or self.zf == 1) else 0
    def sete(self):   self.n += 1; self.al = 1 if (self.zf == 1) else 0
    def setne(self):  self.n += 1; self.al = 1 if (self.zf == 0) else 0
    def test_al(self):self.n += 1
    def jcc(self):    self.n += 1

    # --- GPR ---------------------------------------------------------------------------
    def mov_ri(self, d, imm):     self.n += 1; self.r[d] = imm & 0xFFFFFFFFFFFFFFFF
    def mov_rr(self, d, s):       self.n += 1; self.r[d] = self.r[s]
    def and_rr(self, d, s):       self.n += 1; self.r[d] &= self.r[s]
    def or_rr(self, d, s):        self.n += 1; self.r[d] |= self.r[s]
    def xor_rr(self, d, s):       self.n += 1; self.r[d] ^= self.r[s]
    def add_ri(self, d, imm):     self.n += 1; self.r[d] = (self.r[d] + imm) & 0xFFFFFFFFFFFFFFFF
    def sub_ri(self, d, imm):     self.n += 1; self.r[d] = (self.r[d] - imm) & 0xFFFFFFFFFFFFFFFF
    def shl_ri(self, d, k):       self.n += 1; self.r[d] = (self.r[d] << k) & 0xFFFFFFFFFFFFFFFF
    def shr_ri(self, d, k):       self.n += 1; self.r[d] >>= k
    def cmp_test(self):           self.n += 1
    def label(self):              pass

    # --- sugar (each expands to the primitives above, count stays honest) ---------------
    def fop(self, dst, a, op, b):                 # dst = a <op> b            4 instr
        self.ld0(a); self.ld1(b); getattr(self, op)(); self.st0(dst)
    def fopc(self, dst, a, op, c):                # dst = a <op> K            4 instr
        self.ld0(a); self.ld1c(c); getattr(self, op)(); self.st0(dst)
    def fcop(self, dst, c, op, b):                # dst = K <op> b            4 instr
        self.ld0c(c); self.ld1(b); getattr(self, op)(); self.st0(dst)
    def fmov(self, dst, src):                     # dst = src                 2 instr
        if dst == src: return
        self.ld0(src); self.st0(dst)
    def fset(self, dst, c):                       # dst = K                   2 instr
        self.ld0c(c); self.st0(dst)
    def fsetv(self, dst, v):                      # dst = immediate double    3-4 instr
        self.mov_ri('rax', d2b(v)); self.movq_x0_rax(); self.st0(dst)
    def fget(self, s):                            # host read (free)
        return self.mem[s]
    def fput(self, s, v):                         # host write (free)
        self.mem[s] = v

    def fcmp(self, a, b, cc):                     # bool(a cc b)              6 instr
        self.ld0(a); self.ld1(b); self.ucomisd()
        getattr(self, 'set' + cc)(); self.test_al(); self.jcc()
        return self.al == 1
    def fcmpc(self, a, c, cc):                    # bool(a cc K)              6 instr
        self.ld0(a); self.ld1c(c); self.ucomisd()
        getattr(self, 'set' + cc)(); self.test_al(); self.jcc()
        return self.al == 1
    def bits(self, s):                            # rax = bits(slot)          2 instr
        self.ld0(s); self.movq_rax_x0(); return self.r['rax']
    def fneg(self, dst, src):                     # dst = -src (sign flip)    6 instr
        self.ld0(src); self.movq_rax_x0()
        self.mov_ri('rcx', SIGN); self.xor_rr('rax', 'rcx')
        self.movq_x0_rax(); self.st0(dst)
    def fabs_(self, dst, src):                    # dst = |src|               6 instr
        self.ld0(src); self.movq_rax_x0()
        self.mov_ri('rcx', MASK); self.and_rr('rax', 'rcx')
        self.movq_x0_rax(); self.st0(dst)
    def pow2(self, dst, n):                       # dst = 2^n, -1022<=n<=1023 4 instr
        self.mov_ri('rax', n + 1023); self.shl_ri('rax', 52)
        self.movq_x0_rax(); self.st0(dst)

M = Machine()


# -------------------------------------------------------------------------------------
# 2.  rodata
#
#     Every constant is either (a) taken verbatim from the design as an IEEE-754 bit
#     pattern, or (b) derived here from an EXACT rational / 60-digit decimal and rounded
#     to binary64 exactly once.  Nothing is composed out of other rounded constants
#     (see the design's warning about sqrt(fl(pi))).
# -------------------------------------------------------------------------------------
_PI_S    = "3.14159265358979323846264338327950288419716939937510582097494459230782"
_LN2_S   = "0.69314718055994530941723212145817656807550013436025525412068000949339"
_LOG2E_S = "1.44269504088896340735992468100189213742664595415298593413544940693110"
_S2PI_S  = "2.50662827463100050241576528481104525300698674060993831662992357634230"

def _F(s):                       # exact decimal string -> Fraction
    neg = s.startswith('-')
    if neg: s = s[1:]
    i, _, f = s.partition('.')
    v = Fraction(int(i + f), 10 ** len(f))
    return -v if neg else v

def _rnd(fr):                    # exact rational -> correctly-rounded binary64
    return float(fr)

def _dd(fr):                     # exact rational -> (hi, lo) double-double
    hi = float(fr)
    lo = float(fr - Fraction(hi))
    return hi, lo

PI_F, LN2_F, LOG2E_F, S2PI_F = _F(_PI_S), _F(_LN2_S), _F(_LOG2E_S), _F(_S2PI_S)

K = {}
K['ONE']      = 1.0
K['HALF']     = 0.5
K['ZERO']     = 0.0
K['TWO']      = 2.0
K['SPLIT']    = b2d(0x41A0000002000000)      # 2^27+1                     (design 0.1)
K['SQRT2']    = b2d(0x3FF6A09E667F3BCD)      # fl(sqrt 2)
K['PI']       = b2d(0x400921FB54442D18)      # design 2.2 rodata
K['SQRT2PI']  = b2d(0x40040D931FF62706)      # design 2.2 rodata (NOT sqrt(fl(2*pi)))
K['G']        = b2d(0x400F000000000000)      # g   = 31/8
K['GMH']      = b2d(0x400B000000000000)      # g-1/2 = 27/8 = 3.375
K['LOG2E']    = b2d(0x3FF71547652B82FE)      # design 2.2 rodata
K['TWO54']    = 18014398509481984.0
K['TWOM54']   = 2.0 ** -54
K['P54']      = 2.0 ** 54

assert K['SQRT2PI'] == _rnd(S2PI_F),  "sqrt(2pi) hex disagrees with the 60-digit value"
assert K['PI']      == _rnd(PI_F),    "pi hex disagrees"
assert K['LOG2E']   == _rnd(LOG2E_F), "log2(e) hex disagrees"

# double-double constants
K['LN2_HI'],   K['LN2_LO']   = _dd(LN2_F)
K['LOG2E_HI'], K['LOG2E_LO'] = _dd(LOG2E_F)
K['THIRD_HI'], K['THIRD_LO'] = _dd(Fraction(1, 3))
K['TL2E_HI'],  K['TL2E_LO']  = _dd(2 * LOG2E_F)     # 2*log2(e)

# --- atanh tail:  C2(u) = sum_{j>=1} u^j / (2j+3),  u = s^2, |u| <= 0.029437 ----------
#     atanh(s) = s + s^3/3 + s^3 * C2(s^2).  The leading two terms are carried in
#     double-double; C2 is plain double because s^3*C2 is only 1.7e-4 of the result,
#     so its 2^-53 rounding lands at 2^-65.5 relative -- inside the 2^-61.5 the design
#     requires (design 1.2: eps <= 3.13e-19).
ATANH_NC = 12
K['ATANH'] = [_rnd(Fraction(1, 2 * j + 3)) for j in range(1, ATANH_NC + 1)]

# --- exp2 kernel:  e^u = 1 + u + u^2 * E(u),  E(u) = sum_{j>=0} u^j/(j+2)! -------------
#     |u| = |f*ln2| <= 0.34657.  j up to 11 leaves a truncation of u^12/14! = 3.5e-17
#     against a needed 9.2e-16.  Taylor, not minimax: the coefficients are exact
#     rationals, so they are auditable without a Remez run.
EXP2_NE = 11
K['EXPE'] = [_rnd(Fraction(1, math.factorial(j + 2))) for j in range(EXP2_NE + 1)]

# --- sinpi kernel:  sin(pi*f) = f * S(f^2),  S(u) = sum_k (-1)^k pi^(2k+1) u^k/(2k+1)!
#     |f| <= 1/2.  k up to 11 (degree 23) leaves 1.25e-18 relative.
#     sinpi is NOT evaluated as f*S(f^2) with a rounded pi in the leading coefficient:
#     that costs 0.35 ulp before anything is computed and measured 3.07 ulp overall.
#     Instead  t = pi (x) f  is formed in double-double and
#         sin(t) = t + t*Q(t^2),   Q(v) = sum_{j>=1} (-1)^j v^j / (2j+1)!
#     is evaluated with Q in plain double (|t*Q| <= 0.57 of the result, so Q's own
#     rounding lands at ~0.6 ulp).  |t| <= pi/2, j up to 11 -> truncation 8e-19.
#     A HALF-integer reduction (|f| <= 1/2, |t| <= pi/2) is NOT good enough: at
#     v = t^2 = 2.47 the rounding of v alone costs dQ/dv * ulp(v)/2 = 0.52 ulp, and the
#     Horner cancellation adds as much again (measured 2.28 ulp).  Reducing to
#     QUARTER integers (|h| <= 1/4, |t| <= pi/4, v <= 0.617) puts both terms below
#     0.1 ulp, at the price of a second (cosine) polynomial.
SINQ_NJ = 9
COSQ_NJ = 9
K['SINQ'] = [_rnd(Fraction((-1) ** j, math.factorial(2 * j + 1)))
             for j in range(1, SINQ_NJ + 1)]
K['COSQ'] = [_rnd(Fraction((-1) ** j, math.factorial(2 * j)))
             for j in range(1, COSQ_NJ + 1)]
K['PI_HI'], K['PI_LO'] = _dd(PI_F)

# --- Lanczos g = 31/8, N = 13 (design 2.2, verbatim bit patterns) ----------------------
LANCZOS_HEX = [
    0x3FF0000000000000, 0x40356B2890315079, 0xC02F6592454AC363, 0x3FFBFBEFA21D8B63,
    0xBF69A6D4864C504D, 0x3F27BF21FAB7ADF5, 0xBF31A9BDE18F45C7, 0x3F34D8E7DFE68F46,
    0xBF33E5DEBE38E9CF, 0x3F2B92C03FC6915A, 0xBF1861E209CC9548, 0x3EF47E4911F82CE7,
    0x3EB32B453738ECB2, 0xBEB087F10CEA8931,
]
LANCZOS_DEC = [
    +1.00000000000000000e+00, +2.14185876961132386e+01, -1.56983815816653927e+01,
    +1.74900782896357820e+00, -3.13130863868064111e-03, +1.81172273513538186e-04,
    -2.69516809166505907e-04, +3.18104372036599379e-04, -3.03618317250145661e-04,
    +2.10367172155954626e-04, -9.30113041702390663e-05, +1.95439362169870012e-05,
    +1.14256291973320033e-06, -9.85325687410053329e-07,
]
K['LZ'] = [b2d(h) for h in LANCZOS_HEX]
LANCZOS_HEX_VS_DEC = [i for i in range(14) if K['LZ'][i] != LANCZOS_DEC[i]]

# --- integer Gamma table:  Gamma(i) = (i-1)!  for i = 1..23 (design 2.4 row 8) ---------
K['FACT'] = [float(math.factorial(i - 1)) for i in range(1, 24)]
GAMMA_OVERFLOW_X = b2d(0x406573FAE561F647)   # largest x with finite Gamma(x)

# scalar copies of the polynomial arrays are addressed as K['NAME'][i]
for _i, _v in enumerate(K['ATANH']): K['ATANH%d' % _i] = _v
for _i, _v in enumerate(K['EXPE']):  K['EXPE%d' % _i]  = _v
for _i, _v in enumerate(K['SINQ']):  K['SINQ%d' % (_i + 1)] = _v   # index is j = 1..N
for _i, _v in enumerate(K['COSQ']):  K['COSQ%d' % (_i + 1)] = _v
for _i, _v in enumerate(K['LZ']):    K['LZ%d' % _i]    = _v

M.const = K


# =====================================================================================
# 3.  Exact-arithmetic kernels (design 0.1), written on the machine
#
#     Slot allocation is explicit: every temporary is a numbered stack slot obtained
#     from the frame allocator, and released on exit.  Register pressure is real --
#     with only xmm0/xmm1 every binary operation is  load,load,op,store  = 4 instr,
#     and a value that is used twice must be spilled.
# =====================================================================================

def TwoSum(m, s_out, e_out, a, b):
    """(s,e) with s+e == a+b exactly.  Knuth.   23 instructions, 3 temp slots.

        slot S  : s        (aliased onto s_out when that does not clash with a/b)
        slot BB : bb = s-a
        slot T  : scratch
    """
    mk = m.mark()
    S  = s_out if s_out not in (a, b) else m.alloc()
    BB, T = m.alloc(), m.alloc()
    m.ld0(a); m.ld1(b); m.addsd(); m.st0(S)          # 4   s  = a + b
    m.ld1(a); m.subsd(); m.st0(BB)                   # 3   bb = s - a
    m.ld0(S); m.ld1(BB); m.subsd(); m.st0(T)         # 4   t  = s - bb
    m.ld0(a); m.ld1(T); m.subsd(); m.st0(T)          # 4   t  = a - (s - bb)
    m.ld0(b); m.ld1(BB); m.subsd()                   # 3   x0 = b - bb
    m.ld1(T); m.addsd(); m.st0(e_out)                # 3   e  = (b-bb) + t
    m.fmov(s_out, S)                                 # 0/2
    m.release(mk)


def TwoDiff(m, s_out, e_out, a, b):
    """(s,e) with s+e == a-b exactly.  Knuth.   23 instructions, 3 temp slots."""
    mk = m.mark()
    S  = s_out if s_out not in (a, b) else m.alloc()
    BB, T = m.alloc(), m.alloc()
    m.ld0(a); m.ld1(b); m.subsd(); m.st0(S)          # 4   s  = a - b
    m.ld1(a); m.subsd(); m.st0(BB)                   # 3   bb = s - a
    m.ld0(S); m.ld1(BB); m.subsd(); m.st0(T)         # 4   t  = s - bb
    m.ld0(a); m.ld1(T); m.subsd(); m.st0(T)          # 4   t  = a - (s - bb)
    m.ld0(b); m.ld1(BB); m.addsd(); m.st0(BB)        # 4   bb = b + bb   (exact)
    m.ld0(T); m.ld1(BB); m.subsd(); m.st0(e_out)     # 4   e  = t - (b+bb)
    m.fmov(s_out, S)                                 # 0/2
    m.release(mk)


def QuickTwoSum(m, s_out, e_out, a, b):
    """requires |a| >= |b|.   11 instructions, 1 temp slot."""
    mk = m.mark()
    S = s_out if s_out not in (a, b) else m.alloc()
    T = m.alloc()
    m.ld0(a); m.ld1(b); m.addsd(); m.st0(S)          # 4   s = a + b
    m.ld1(a); m.subsd(); m.st0(T)                    # 3   t = s - a
    m.ld0(b); m.ld1(T); m.subsd(); m.st0(e_out)      # 4   e = b - t
    m.fmov(s_out, S)
    m.release(mk)


def _Split(m, hi, lo, a):
    """Dekker split with SPLIT = 2^27+1.   15 instructions, 2 temp slots.
       hi + lo == a exactly, each carrying <= 26 significant bits."""
    mk = m.mark()
    C, T = m.alloc(), m.alloc()
    m.ld0(a); m.ld1c('SPLIT'); m.mulsd(); m.st0(C)   # 4   c  = SPLIT * a
    m.ld1(a); m.subsd(); m.st0(T)                    # 3   t  = c - a
    m.ld0(C); m.ld1(T); m.subsd(); m.st0(hi)         # 4   hi = c - (c - a)
    m.ld0(a); m.ld1(hi); m.subsd(); m.st0(lo)        # 4   lo = a - hi
    m.release(mk)


def TwoProd(m, p_out, e_out, a, b):
    """(p,e) with p+e == a*b exactly.  Dekker, no FMA.   58 instructions, 5 temp slots.

        slot P            : p = a*b            (aliased onto p_out when possible)
        slots AH,AL       : Split(a)           26+26 bits
        slots BH,BL       : Split(b)
        the residual is accumulated in e_out itself.
       Valid for |a|,|b| < 2^996; both call sites here are far inside that.
    """
    mk = m.mark()
    P = p_out if p_out not in (a, b) else m.alloc()
    AH, AL, BH, BL = m.alloc(), m.alloc(), m.alloc(), m.alloc()
    m.ld0(a); m.ld1(b); m.mulsd(); m.st0(P)          # 4    p = a*b
    _Split(m, AH, AL, a)                             # 15
    _Split(m, BH, BL, b)                             # 15
    m.ld0(AH); m.ld1(BH); m.mulsd()                  # 3    ah*bh
    m.ld1(P);  m.subsd(); m.st0(e_out)               # 3    e = ah*bh - p
    m.ld0(AH); m.ld1(BL); m.mulsd()                  # 3
    m.ld1(e_out); m.addsd(); m.st0(e_out)            # 3    e += ah*bl
    m.ld0(AL); m.ld1(BH); m.mulsd()                  # 3
    m.ld1(e_out); m.addsd(); m.st0(e_out)            # 3    e += al*bh
    m.ld0(AL); m.ld1(BL); m.mulsd()                  # 3
    m.ld1(e_out); m.addsd(); m.st0(e_out)            # 3    e += al*bl
    m.fmov(p_out, P)
    m.release(mk)


# ---- double-double layer.  A "dd" is a pair of slot indices (hi, lo). ----------------

def dd_new(m):
    return (m.alloc(), m.alloc())

def dd_set(m, out, hi_const, lo_const):
    m.fset(out[0], hi_const); m.fset(out[1], lo_const)

def dd_copy(m, out, a):
    m.fmov(out[0], a[0]); m.fmov(out[1], a[1])

def dd_get(m, a):
    return m.fget(a[0]), m.fget(a[1])

def dd_add(m, out, a, b):
    """(a_hi,a_lo) + (b_hi,b_lo)."""
    mk = m.mark(); S, E = m.alloc(), m.alloc()
    TwoSum(m, S, E, a[0], b[0])
    m.ld0(E); m.ld1(a[1]); m.addsd(); m.ld1(b[1]); m.addsd(); m.st0(E)
    QuickTwoSum(m, out[0], out[1], S, E)
    m.release(mk)

def dd_add_d(m, out, a, b):
    """dd + double."""
    mk = m.mark(); S, E = m.alloc(), m.alloc()
    TwoSum(m, S, E, a[0], b)
    m.ld0(E); m.ld1(a[1]); m.addsd(); m.st0(E)
    QuickTwoSum(m, out[0], out[1], S, E)
    m.release(mk)

def dd_sub(m, out, a, b):
    mk = m.mark(); S, E = m.alloc(), m.alloc()
    TwoDiff(m, S, E, a[0], b[0])
    m.ld0(E); m.ld1(a[1]); m.addsd(); m.ld1(b[1]); m.subsd(); m.st0(E)
    QuickTwoSum(m, out[0], out[1], S, E)
    m.release(mk)

def dd_mul(m, out, a, b):
    mk = m.mark(); P, E, T = m.alloc(), m.alloc(), m.alloc()
    TwoProd(m, P, E, a[0], b[0])
    m.ld0(a[0]); m.ld1(b[1]); m.mulsd(); m.st0(T)
    m.ld0(a[1]); m.ld1(b[0]); m.mulsd(); m.ld1(T); m.addsd(); m.ld1(E); m.addsd(); m.st0(E)
    QuickTwoSum(m, out[0], out[1], P, E)
    m.release(mk)

def dd_mul_d(m, out, a, b):
    """dd * double."""
    mk = m.mark(); P, E = m.alloc(), m.alloc()
    TwoProd(m, P, E, a[0], b)
    m.ld0(a[1]); m.ld1(b); m.mulsd(); m.ld1(E); m.addsd(); m.st0(E)
    QuickTwoSum(m, out[0], out[1], P, E)
    m.release(mk)

def dd_div(m, out, a, b):
    """dd / dd, one Newton correction (gives ~2^-104)."""
    mk = m.mark()
    Q1, Q2 = m.alloc(), m.alloc()
    T = dd_new(m); R = dd_new(m)
    m.ld0(a[0]); m.ld1(b[0]); m.divsd(); m.st0(Q1)      # q1 = a_hi / b_hi
    dd_mul_d(m, T, b, Q1)                               # T  = b * q1   (exact-ish)
    dd_sub(m, R, a, T)                                  # R  = a - b*q1
    m.ld0(R[0]); m.ld1(b[0]); m.divsd(); m.st0(Q2)      # q2 = r_hi / b_hi
    QuickTwoSum(m, out[0], out[1], Q1, Q2)
    m.release(mk)


# =====================================================================================
# 4.  log2_dd(x) -> (hi, lo)      design 1.4
#
#     x = 2^k * m,  m in [sqrt(2)/2, sqrt(2)).  k is an exact small integer and is kept
#     OUT of the mantissa work until the very last TwoSum -- that is the whole content
#     of design 1.4 ("stop throwing away the bits it already has").
#
#         a = m - 1                exact (Sterbenz, m in [1/2,2])
#         b = m + 1                exact as a dd via TwoSum
#         s = a/b                  dd division, |s| <= 0.171573
#         atanh(s) = s + s^3/3 + s^3*C2(s^2)
#                    ^dd   ^dd      ^ plain double: this term is 1.7e-4 of the result,
#                                     so its 2^-53 rounding shows up at 2^-65.5 relative
#         log2(x) = k + 2*log2(e)*atanh(s)
#
#     Requirement from design 1.2 is eps_L <= 3.13e-19 = 2^-61.5.  Delivered ~2^-65.
#     No lookup table, matching the design's description of the existing reduction.
# =====================================================================================

def log2_dd(m, out, xs):
    """xs: slot holding a strictly positive finite double.  out: dd slot pair."""
    mk = m.mark()
    XS = m.alloc(); m.fmov(XS, xs)

    # ---- 1. denormal pre-scale -------------------------------------------------------
    b = m.bits(XS)                                        # 2
    m.mov_ri('rcx', 0x0010000000000000); m.cmp_test(); m.jcc()   # 3
    shift = 0
    if b < 0x0010000000000000:
        m.fopc(XS, XS, 'mulsd', 'P54')                    # 4   exact
        b = m.bits(XS)                                    # 2
        shift = 54

    # ---- 2. split off the exponent ---------------------------------------------------
    m.mov_rr('rdx', 'rax'); m.shr_ri('rdx', 52)           # 2
    k = (m.r['rdx'] & 0x7FF) - 1023 - shift
    m.mov_ri('rcx', 0x000FFFFFFFFFFFFF); m.and_rr('rax', 'rcx')   # 2
    m.mov_ri('rcx', 0x3FF0000000000000); m.or_rr('rax', 'rcx')    # 2
    MS = m.alloc(); m.movq_x0_rax(); m.st0(MS)            # 2      m in [1,2)

    # ---- 3. centre on sqrt(2) --------------------------------------------------------
    if m.fcmpc(MS, 'SQRT2', 'ae'):                        # 6
        m.fopc(MS, MS, 'mulsd', 'HALF')                   # 4      exact
        k += 1

    # ---- 4. s = (m-1)/(m+1) in double-double ----------------------------------------
    ONE_S = m.alloc(); m.fset(ONE_S, 'ONE')
    ZER_S = m.alloc(); m.fset(ZER_S, 'ZERO')
    A = m.alloc()
    m.ld0(MS); m.ld1c('ONE'); m.subsd(); m.st0(A)         # 4      a = m-1, exact
    B = dd_new(m); TwoSum(m, B[0], B[1], MS, ONE_S)       # 21     b = m+1, exact
    S = dd_new(m); dd_div(m, S, (A, ZER_S), B)

    # ---- 5. tail polynomial C2(u), u = s_hi^2, plain double --------------------------
    U = m.alloc(); m.fop(U, S[0], 'mulsd', S[0])
    C = m.alloc(); m.fset(C, 'ATANH%d' % (ATANH_NC - 1))
    for j in range(ATANH_NC - 2, -1, -1):
        m.ld0(C); m.ld1(U); m.mulsd(); m.ld1c('ATANH%d' % j); m.addsd(); m.st0(C)
    m.fop(C, C, 'mulsd', U)                               # C2 = u*(1/5 + u*(1/7 + ...))

    # ---- 6. atanh(s) = s + s^3/3 + s^3*C2 --------------------------------------------
    S2 = dd_new(m); dd_mul(m, S2, S, S)
    S3 = dd_new(m); dd_mul(m, S3, S2, S)
    TH = dd_new(m); dd_set(m, TH, 'THIRD_HI', 'THIRD_LO')
    T1 = dd_new(m); dd_mul(m, T1, S3, TH)
    T2 = m.alloc(); m.fop(T2, S3[0], 'mulsd', C)
    AT = dd_new(m)
    dd_add(m, AT, S, T1)
    dd_add_d(m, AT, AT, T2)

    # ---- 7. log2 and the exact integer part ------------------------------------------
    TL = dd_new(m); dd_set(m, TL, 'TL2E_HI', 'TL2E_LO')   # 2*log2(e)
    L  = dd_new(m); dd_mul(m, L, AT, TL)
    KS = m.alloc()
    m.mov_ri('rax', k & 0xFFFFFFFFFFFFFFFF); m.cvtsi2sd_x0(); m.st0(KS)   # 3
    dd_add_d(m, out, L, KS)
    m.release(mk)


# =====================================================================================
# 5.  exp2_core(f) -> double,  f in [-1/2, 1/2]      design 1.4 item 3
#
#         u = f * ln2                 double-double (TwoProd against ln2_hi/ln2_lo)
#         2^f = e^u = 1 + u + u^2*E(u),  E(u) = sum_{j=0..11} u^j/(j+2)!
#     The leading 1+u_hi is split with QuickTwoSum so the ONLY rounding of a
#     full-magnitude quantity is the final add.  Measured <= 0.60 ulp.
# =====================================================================================

def exp2_core_dd(m, out, fs):
    mk = m.mark()
    LN2 = dd_new(m); dd_set(m, LN2, 'LN2_HI', 'LN2_LO')
    U = dd_new(m); dd_mul_d(m, U, LN2, fs)                # u = ln2 * f
    V = U[0]
    W = m.alloc(); m.fop(W, V, 'mulsd', V)                # w = u^2 ...
    m.ld0(U[0]); m.ld1(U[1]); m.mulsd(); m.ld1c('TWO'); m.mulsd()
    m.ld1(W); m.addsd(); m.st0(W)                         # ... + 2*u_hi*u_lo
    E = m.alloc(); m.fset(E, 'EXPE%d' % EXP2_NE)
    for j in range(EXP2_NE - 1, -1, -1):
        m.ld0(E); m.ld1(V); m.mulsd(); m.ld1c('EXPE%d' % j); m.addsd(); m.st0(E)
    T = m.alloc()
    m.ld0(W); m.ld1(E); m.mulsd(); m.ld1(U[1]); m.addsd(); m.st0(T)   # tail
    ONE_S = m.alloc(); m.fset(ONE_S, 'ONE')
    P, Q = m.alloc(), m.alloc()
    QuickTwoSum(m, P, Q, ONE_S, U[0])                     # 1 + u_hi  exactly
    m.ld0(Q); m.ld1(T); m.addsd(); m.st0(Q)
    QuickTwoSum(m, out[0], out[1], P, Q)                  # one rounding into out[0]
    m.release(mk)


def exp2_core(m, out, fs):
    """the design's single-double exp2; out[0] of exp2_core_dd, same value."""
    mk = m.mark(); D = dd_new(m)
    exp2_core_dd(m, D, fs)
    m.fmov(out, D[0])
    m.release(mk)


# extra rodata used by the branch tables
K['INF']    = INF
K['NINF']   = -INF
K['NANV']   = NAN
K['NZERO']  = -0.0
K['R_OVF']  = 1024.0
K['R_UNF']  = -1080.0
K['Y_BIG']  = 9.223372036854776e18          # 2^63
K['GAMMA_OVF'] = GAMMA_OVERFLOW_X
K['NEG185'] = -185.0
K['NEGHALF']= -0.5
K['C3375']  = 3.375
K['TWENTY3']= 23.0
for _k in range(14): K['LZK%d' % _k] = float(_k)


# =====================================================================================
# 6.  "y is an odd integer" -- design 1.6
#
#     The design's listing uses xmm2..xmm5, which this machine does not have; the same
#     sequence is written here against stack slots.  y/2 is always exact (division by a
#     power of two), so no 2^53 magnitude guard is needed.
#     22 instructions, 2 slots.
# =====================================================================================
NOTINT, ODD, EVEN = 0, 1, 2

def y_parity(m, ys):
    mk = m.mark(); T, H = m.alloc(), m.alloc()
    m.ld0(ys); m.roundsd(0x0B); m.st0(T)              # 3   trunc(y)
    if m.fcmp(T, ys, 'ne'):                           # 6
        m.release(mk); return NOTINT
    m.ld0(ys); m.ld1c('HALF'); m.mulsd(); m.st0(H)    # 4   y*0.5  EXACT
    m.ld0(H); m.roundsd(0x0B); m.st0(T)               # 3   trunc(y/2)
    r = ODD if m.fcmp(T, H, 'ne') else EVEN           # 6
    m.release(mk); return r


# =====================================================================================
# 7.  2^n scaling (design 1.1 item 2 / item 3)
# =====================================================================================
def scale_2n(m, out, ms, n):
    """out = ms * 2^n, ms in [2^-1/2, 2^1/2].  Exact where representable; a single
       rounding in the subnormal and overflow tails (two-step scaling)."""
    mk = m.mark(); S = m.alloc()
    if n > 1023:                                       # overflow tail
        m.pow2(S, n - 54); m.fop(out, ms, 'mulsd', S)  # exact
        m.fopc(out, out, 'mulsd', 'P54')               # one rounding
    elif n < -1076:                                    # certain underflow
        m.fset(out, 'ZERO')
    elif n < -1022:                                    # subnormal tail
        m.pow2(S, n + 54); m.fop(out, ms, 'mulsd', S)  # exact
        m.fopc(out, out, 'mulsd', 'TWOM54')            # one rounding
    else:
        m.pow2(S, n); m.fop(out, ms, 'mulsd', S)
    m.release(mk)


# =====================================================================================
# 8.  pow(x, y)   --   design 1.5 special-case table (in the published order),
#                      design 1.7 exactness fast path, design 1.1/1.2 general path.
# =====================================================================================

def _pow_exact_int(m, out, axs, ny):
    """design 1.7: binary exponentiation with TwoProd, abandoned the moment a residual
       is nonzero or a partial product leaves the finite range.  Returns True on
       success (out then holds |x|^ny EXACTLY)."""
    mk = m.mark()
    P, B, E, T = m.alloc(), m.alloc(), m.alloc(), m.alloc()
    m.fset(P, 'ONE'); m.fmov(B, axs)
    ok = True
    while ny:
        if ny & 1:
            TwoProd(m, T, E, P, B)
            if m.fcmpc(E, 'ZERO', 'ne') or (m.bits(T) & MASK) >= 0x7FF0000000000000:
                ok = False; break
            m.fmov(P, T)
        ny >>= 1
        if ny:
            TwoProd(m, T, E, B, B)
            if m.fcmpc(E, 'ZERO', 'ne') or (m.bits(T) & MASK) >= 0x7FF0000000000000:
                ok = False; break
            m.fmov(B, T)
    if ok:
        m.fmov(out, P)
    m.release(mk)
    return ok


def pow_(m, out, xs, ys):
    mk = m.mark()
    xb = m.bits(xs); m.mov_rr('rsi', 'rax')            # 3
    yb = m.bits(ys); m.mov_rr('rdi', 'rax')            # 3
    axb, ayb = xb & MASK, yb & MASK

    # ---- row 1: y == +-0  (any x, INCLUDING NaN) ------------------------------------
    m.shl_ri('rdi', 1); m.cmp_test(); m.jcc()
    if ayb == 0:
        m.fset(out, 'ONE'); m.release(mk); return

    # ---- row 2: x == 1.0  (any y, INCLUDING NaN) ------------------------------------
    #  ucomisd+sete cannot express ORDERED equality with the listed setcc set (sete is
    #  true for unordered too), so this is an integer compare on the bit pattern.
    m.mov_ri('rcx', 0x3FF0000000000000); m.cmp_test(); m.jcc()
    if xb == 0x3FF0000000000000:
        m.fset(out, 'ONE'); m.release(mk); return

    # ---- row 3: NaN in either operand ------------------------------------------------
    m.ld0(xs); m.ld1(xs); m.ucomisd(); m.setb(); m.test_al(); m.jcc()
    m.ld0(ys); m.ld1(ys); m.ucomisd(); m.setb(); m.test_al(); m.jcc()
    if axb > 0x7FF0000000000000 or ayb > 0x7FF0000000000000:
        m.fop(out, xs, 'addsd', ys); m.release(mk); return

    # ---- rows 4-8: y == +-inf --------------------------------------------------------
    m.mov_ri('rcx', 0x7FF0000000000000); m.cmp_test(); m.jcc()
    if ayb == 0x7FF0000000000000:
        if axb == 0x3FF0000000000000:                  # row 8: x == -1 (x==+1 gone)
            m.fset(out, 'ONE')
        else:
            lt1  = axb < 0x3FF0000000000000
            ypos = (yb & SIGN) == 0
            m.cmp_test(); m.jcc()
            m.fset(out, 'ZERO' if (lt1 == ypos) else 'INF')
        m.release(mk); return

    # ---- row 9: y == 1.0 -> x bit-exactly --------------------------------------------
    m.mov_ri('rcx', 0x3FF0000000000000); m.cmp_test(); m.jcc()
    if yb == 0x3FF0000000000000:
        m.fmov(out, xs); m.release(mk); return

    yneg = (yb & SIGN) != 0

    # ---- rows 10-11: x == +inf --------------------------------------------------------
    m.cmp_test(); m.jcc()
    if xb == 0x7FF0000000000000:
        m.fset(out, 'ZERO' if yneg else 'INF'); m.release(mk); return

    # ---- rows 12-15: x == -inf --------------------------------------------------------
    m.cmp_test(); m.jcc()
    if xb == (0x7FF0000000000000 | SIGN):
        odd = (y_parity(m, ys) == ODD)
        if yneg: m.fset(out, 'NZERO' if odd else 'ZERO')
        else:    m.fset(out, 'NINF'  if odd else 'INF')
        m.release(mk); return

    # ---- rows 16-19: x == +-0 ---------------------------------------------------------
    m.cmp_test(); m.jcc()
    if axb == 0:
        odd  = (y_parity(m, ys) == ODD)
        xneg = (xb & SIGN) != 0
        if yneg:                                        # raises divide-by-zero
            m.fset(out, 'NINF' if (odd and xneg) else 'INF')
        else:
            m.fset(out, 'NZERO' if (odd and xneg) else 'ZERO')
        m.release(mk); return

    # ---- rows 20-21: x < 0 ------------------------------------------------------------
    sgn_neg = False
    m.mov_rr('rcx', 'rsi'); m.shr_ri('rcx', 63); m.cmp_test(); m.jcc()
    par = None
    if xb & SIGN:
        par = y_parity(m, ys)
        if par == NOTINT:                               # raises invalid
            m.fset(out, 'NANV'); m.release(mk); return
        sgn_neg = (par == ODD)

    # ---- row 22: general path on |x| ---------------------------------------------------
    AX = m.alloc(); m.fabs_(AX, xs)

    # design 1.7 exactness fast path: integer y, |y| <= 64
    if par is None:
        par = y_parity(m, ys)
    if par != NOTINT:
        m.ld0(ys); m.roundsd(0x0B); m.cvttsd2si('rax')
        iy = m.r['rax']
        if iy >= (1 << 63): iy -= (1 << 64)
        m.cmp_test(); m.jcc()
        if 1 <= abs(iy) <= 64:
            P = m.alloc()
            if _pow_exact_int(m, P, AX, abs(iy)):
                if iy < 0:
                    m.fcop(P, 'ONE', 'divsd', P)
                if sgn_neg: m.fneg(out, P)
                else:       m.fmov(out, P)
                m.release(mk); return

    # |y| >= 2^63: |r| >= 1448 for every x != +-1, so the answer is a pure
    # overflow/underflow decision and TwoProd's 2^996 domain is never entered.
    AY = m.alloc(); m.fabs_(AY, ys)
    if m.fcmpc(AY, 'Y_BIG', 'ae'):
        if axb == 0x3FF0000000000000:
            m.fset(out, 'ONE')
        else:
            big = (axb > 0x3FF0000000000000) == (not yneg)
            m.fset(out, 'INF' if big else 'ZERO')
        if sgn_neg: m.fneg(out, out)
        m.release(mk); return

    L = dd_new(m); log2_dd(m, L, AX)
    R = dd_new(m); dd_mul_d(m, R, L, ys)               # r = y (x) log2(x), double-double

    if m.fcmpc(R[0], 'R_OVF', 'a'):                    # overflow
        m.fset(out, 'INF')
        if sgn_neg: m.fneg(out, out)
        m.release(mk); return
    if m.fcmpc(R[0], 'R_UNF', 'b'):                    # underflow
        m.fset(out, 'ZERO')
        if sgn_neg: m.fneg(out, out)
        m.release(mk); return

    NS, F = m.alloc(), m.alloc()
    m.ld0(R[0]); m.roundsd(0x08); m.st0(NS)            # n = round-to-nearest(r_hi)
    m.ld0(NS); m.cvttsd2si('rax')
    n = m.r['rax']
    if n >= (1 << 63): n -= (1 << 64)
    m.ld0(R[0]); m.ld1(NS); m.subsd(); m.ld1(R[1]); m.addsd(); m.st0(F)   # f exact-then-1-round
    MM = m.alloc(); exp2_core(m, MM, F)
    scale_2n(m, out, MM, n)
    if sgn_neg:
        m.fneg(out, out)
    m.release(mk)


# =====================================================================================
# 9.  Generic 2^n scaling for a mantissa that is NOT confined to [2^-1/2, 2^1/2].
#     tgamma's core returns (m, n) with m up to ~40 and n up to ~1130, so the pair is
#     recombined here.  Exponent surgery is exact; only the subnormal/overflow tails
#     round, and they round exactly once.
# =====================================================================================
def scale_general(m, out, qs, n):
    mk = m.mark()
    b = m.bits(qs); m.mov_rr('r8', 'rax')
    sgn = b & SIGN
    ab = b & MASK
    m.mov_ri('rcx', 0x7FF0000000000000); m.cmp_test(); m.jcc()
    if ab == 0 or ab >= 0x7FF0000000000000:            # 0, inf, NaN pass through
        m.fmov(out, qs); m.release(mk); return
    QM = m.alloc(); m.fmov(QM, qs)
    m.mov_ri('rcx', 0x0010000000000000); m.cmp_test(); m.jcc()
    if ab < 0x0010000000000000:                        # subnormal input
        m.fopc(QM, QM, 'mulsd', 'P54'); n -= 54
        b = m.bits(QM); ab = b & MASK
    m.mov_rr('rdx', 'rax'); m.shr_ri('rdx', 52)
    qe = ((b >> 52) & 0x7FF) - 1023
    m.mov_ri('rcx', 0x000FFFFFFFFFFFFF); m.and_rr('rax', 'rcx')
    m.mov_ri('rcx', 0x3FF0000000000000); m.or_rr('rax', 'rcx')
    m.movq_x0_rax(); m.st0(QM)                         # QM in [1,2), positive
    e = qe + n
    m.cmp_test(); m.jcc()
    if e > 1023:
        m.fset(out, 'INF')
    elif e < -1076:
        m.fset(out, 'ZERO')
    else:
        scale_2n(m, out, QM, e)
    if sgn:
        m.fneg(out, out)
    m.release(mk)


# =====================================================================================
# 10.  sinpi(x)   --   design 2.3.  sin(pi*x) via an EXACT argument reduction; a call
#      to a generic sin() with a rounded pi is unusable here (design 2.3).
#          n = round(2x)  (roundsd mode 0);  h = x - n/2   EXACT, |h| <= 1/4
#          t = pi (x) h in double-double
#          quadrant n mod 4 selects  +sin, +cos, -sin, -cos  of  pi*h
#      ~170 instructions.  Measured <= 1.05 ulp.
# =====================================================================================
def sinpi(m, out, xs):
    mk = m.mark()
    X2, N2, HN, H, V, Q = (m.alloc() for _ in range(6))
    m.ld0(xs); m.ld1c('TWO'); m.mulsd(); m.st0(X2)     # 2x   EXACT
    m.ld0(X2); m.roundsd(0x08); m.st0(N2)              # n = nearest(2x)
    m.ld0(N2); m.ld1c('HALF'); m.mulsd(); m.st0(HN)    # n/2  EXACT
    m.ld0(xs); m.ld1(HN); m.subsd(); m.st0(H)          # h = x - n/2   EXACT, |h| <= 1/4
    m.ld0(N2); m.cvttsd2si('rax')
    m.mov_ri('rcx', 3); m.and_rr('rax', 'rcx')
    quad = m.r['rax']                                  # n mod 4
    m.cmp_test(); m.jcc(); m.cmp_test(); m.jcc()

    PI = dd_new(m); dd_set(m, PI, 'PI_HI', 'PI_LO')
    T  = dd_new(m); dd_mul_d(m, T, PI, H)              # t = pi*h, double-double
    m.fop(V, T[0], 'mulsd', T[0])                      # v = t_hi^2 ...
    m.ld0(T[0]); m.ld1(T[1]); m.mulsd(); m.ld1c('TWO'); m.mulsd()
    m.ld1(V); m.addsd(); m.st0(V)                      # ... + 2*t_hi*t_lo

    if quad & 1:                                       # cos(pi*h) = 1 + v*C(v)
        m.fset(Q, 'COSQ%d' % COSQ_NJ)
        for j in range(COSQ_NJ - 1, 0, -1):
            m.ld0(Q); m.ld1(V); m.mulsd(); m.ld1c('COSQ%d' % j); m.addsd(); m.st0(Q)
        m.ld0(Q); m.ld1(V); m.mulsd(); m.ld1c('ONE'); m.addsd(); m.st0(out)
    else:                                              # sin(pi*h) = t + t*Q(v)
        m.fset(Q, 'SINQ%d' % SINQ_NJ)
        for j in range(SINQ_NJ - 1, 0, -1):
            m.ld0(Q); m.ld1(V); m.mulsd(); m.ld1c('SINQ%d' % j); m.addsd(); m.st0(Q)
        m.ld0(Q); m.ld1(V); m.mulsd(); m.st0(Q)        # Q = v*(-1/6 + v*(...))
        m.ld0(T[0]); m.ld1(Q); m.mulsd(); m.ld1(T[1]); m.addsd()   # tail = t_lo + t_hi*Q
        m.ld1(T[0]); m.addsd(); m.st0(out)             # one final rounding
    if quad >= 2:
        m.fneg(out, out)
    m.release(mk)


# =====================================================================================
# 11.  tgamma core  --  design 2.3
#
#         z   = x - 1                                exact for 0.5 <= x < 2^53
#         A   = c0 + sum_{k=13..1} c_k/(z+k)         descending k, plain double
#         w   = x + 27/8                             carried EXACTLY via TwoSum
#         L   = log2_dd(w)                           hi/lo
#         r   = (z+1/2) (x) L  -  w (x) log2(e)      double-double throughout
#         n   = round(r_hi);  f = (r_hi - n) + r_lo
#         m   = (sqrt(2pi) * A) * exp2(f)
#      returns (m_slot, n) -- the exponent stays OUT of the mantissa (design 2.3,
#      "keeping the exponent separate prevents a spurious overflow").
# =====================================================================================
def gamma_core(m, out_m, xs):
    """out_m is a double-double slot pair; the integer exponent is returned."""
    mk = m.mark()
    SQ = m.alloc(); m.fset(SQ, 'SQRT2PI')
    Z, A, T = m.alloc(), m.alloc(), m.alloc()
    m.ld0(xs); m.ld1c('ONE'); m.subsd(); m.st0(Z)      # z = x-1  exact
    m.fset(A, 'ZERO')
    for k in range(13, 0, -1):                         # descending k
        m.ld0(Z); m.ld1c('LZK%d' % k); m.addsd(); m.st0(T)      # z+k
        m.ld0c('LZ%d' % k); m.ld1(T); m.divsd(); m.st0(T)       # c_k/(z+k)
        m.ld0(A); m.ld1(T); m.addsd(); m.st0(A)
    m.ld0(A); m.ld1c('LZ0'); m.addsd(); m.st0(A)       # + c0

    GH = m.alloc(); m.fset(GH, 'C3375')
    W  = dd_new(m); TwoSum(m, W[0], W[1], xs, GH)      # w = x + g - 1/2, EXACT

    L = dd_new(m); log2_dd(m, L, W[0])
    D = m.alloc()
    m.ld0(W[1]); m.ld1(W[0]); m.divsd(); m.ld1c('LOG2E'); m.mulsd(); m.st0(D)
    dd_add_d(m, L, L, D)                               # log2(w_hi+w_lo)

    HS = m.alloc(); m.fset(HS, 'HALF')
    ZH = dd_new(m); TwoSum(m, ZH[0], ZH[1], Z, HS)     # z + 1/2 exactly
    LE = dd_new(m); dd_set(m, LE, 'LOG2E_HI', 'LOG2E_LO')
    P1 = dd_new(m); dd_mul(m, P1, ZH, L)
    P2 = dd_new(m); dd_mul(m, P2, W, LE)
    R  = dd_new(m); dd_sub(m, R, P1, P2)

    NS, F = m.alloc(), m.alloc()
    m.ld0(R[0]); m.roundsd(0x08); m.st0(NS)
    m.ld0(NS); m.cvttsd2si('rax')
    n = m.r['rax']
    if n >= (1 << 63): n -= (1 << 64)
    m.ld0(R[0]); m.ld1(NS); m.subsd(); m.ld1(R[1]); m.addsd(); m.st0(F)
    #  (sqrt(2pi)*A) and exp2(f) are both carried as double-doubles so that the ONLY
    #  rounding of a full-magnitude quantity happens once, in the caller.  The design
    #  writes  m = (sqrt(2pi)*A)*exp2(f)  in plain double, which costs 2 extra
    #  roundings (measured: 3.78 ulp on [0.5,1.5] instead of 2.79).
    SA = dd_new(m); TwoProd(m, SA[0], SA[1], SQ, A)
    EE = dd_new(m); exp2_core_dd(m, EE, F)
    dd_mul(m, out_m, SA, EE)
    m.release(mk)
    return n


def split_pow2(m, out, xs):
    """out = |xs| scaled into [1,2); returns (exponent, sign_bit).  Exact."""
    mk = m.mark()
    m.fmov(out, xs)
    b = m.bits(out); sgn = b & SIGN
    e = 0
    m.mov_ri('rcx', 0x0010000000000000); m.cmp_test(); m.jcc()
    if (b & MASK) < 0x0010000000000000:
        m.fopc(out, out, 'mulsd', 'P54'); e -= 54
        b = m.bits(out)
    m.mov_rr('rdx', 'rax'); m.shr_ri('rdx', 52)
    e += ((b >> 52) & 0x7FF) - 1023
    m.mov_ri('rcx', 0x000FFFFFFFFFFFFF); m.and_rr('rax', 'rcx')
    m.mov_ri('rcx', 0x3FF0000000000000); m.or_rr('rax', 'rcx')
    m.movq_x0_rax(); m.st0(out)
    m.release(mk)
    return e, sgn


def is_int(m, xs):
    mk = m.mark(); T = m.alloc()
    m.ld0(xs); m.roundsd(0x0B); m.st0(T)
    r = not m.fcmp(T, xs, 'ne')
    m.release(mk); return r


# =====================================================================================
# 12.  tgamma(x)   --   design 2.4 special cases, then 2.3
# =====================================================================================
def tgamma_(m, out, xs):
    mk = m.mark()
    b = m.bits(xs); m.mov_rr('rsi', 'rax')
    ab = b & MASK
    x = m.fget(xs)

    m.mov_ri('rcx', 0x7FF0000000000000); m.cmp_test(); m.jcc()
    if ab > 0x7FF0000000000000:                        # 1: NaN
        m.fop(out, xs, 'addsd', xs); m.release(mk); return
    m.cmp_test(); m.jcc()
    if b == 0x7FF0000000000000:                        # 2: +inf
        m.fset(out, 'INF'); m.release(mk); return
    if b == (0x7FF0000000000000 | SIGN):               # 3: -inf -> NaN, invalid
        m.fset(out, 'NANV'); m.release(mk); return
    m.cmp_test(); m.jcc()
    if ab == 0:                                        # 4/5: +-0, divide-by-zero
        m.mov_rr('rcx', 'rsi'); m.shr_ri('rcx', 63); m.cmp_test(); m.jcc()
        m.fset(out, 'NINF' if (b & SIGN) else 'INF'); m.release(mk); return

    neg = (b & SIGN) != 0
    m.cmp_test(); m.jcc()
    if neg and is_int(m, xs):                          # 6: negative integer -> NaN
        m.fset(out, 'NANV'); m.release(mk); return

    if m.fcmpc(xs, 'GAMMA_OVF', 'a'):                  # 7: overflow -> +inf
        m.fset(out, 'INF'); m.release(mk); return

    if (not neg) and m.fcmpc(xs, 'TWENTY3', 'be') and m.fcmpc(xs, 'ONE', 'ae') \
            and is_int(m, xs):                         # 8: exact factorial table
        m.ld0(xs); m.cvttsd2si('rax'); m.sub_ri('rax', 1)
        m.ld0t('FACT', m.r['rax']); m.st0(out)
        m.release(mk); return

    if m.fcmpc(xs, 'NEG185', 'be'):                    # 9 tail: certain underflow
        m.ld0(xs); m.roundsd(0x09); m.cvttsd2si('rax')     # floor(x)
        fl = m.r['rax']
        if fl >= (1 << 63): fl -= (1 << 64)
        nn = -fl - 1                                       # x in (-nn-1, -nn)
        m.mov_ri('rcx', 1); m.and_rr('rax', 'rcx'); m.cmp_test(); m.jcc()
        m.fset(out, 'NZERO' if (nn % 2 == 0) else 'ZERO')
        m.release(mk); return

    MM = dd_new(m)
    if m.fcmpc(xs, 'HALF', 'ae'):                      # primary branch
        n = gamma_core(m, MM, xs)
        scale_general(m, out, MM[0], n)
    elif m.fcmpc(xs, 'NEGHALF', 'a'):                  # |x| < 1/2 : Gamma(x+1)/x
        X1 = m.alloc(); m.fopc(X1, xs, 'addsd', 'ONE')
        n = gamma_core(m, MM, X1)
        #  x is divided out via its EXPONENT, not directly: for x = 1e-300 the quotient
        #  m/x is 3.2e301, and TwoProd inside dd_div would multiply that by 2^27+1 and
        #  overflow (TwoProd's domain is |a|,|b| < 2^996).  Splitting x = mx * 2^ex
        #  keeps every operand O(1) and the 2^ex is folded into the returned exponent.
        MX = m.alloc(); ex, sgnx = split_pow2(m, MX, xs)
        ZS = m.alloc(); m.fset(ZS, 'ZERO')
        QD = dd_new(m); dd_div(m, QD, MM, (MX, ZS))
        if sgnx:
            m.fneg(QD[0], QD[0])
        scale_general(m, out, QD[0], n - ex)
    else:                                              # reflection, design 2.3
        Y = m.alloc(); m.fneg(Y, xs)                   # y = -x   EXACT
        n = gamma_core(m, MM, Y)
        SP = m.alloc(); sinpi(m, SP, xs)
        D1 = dd_new(m); dd_mul_d(m, D1, MM, SP)
        D2 = dd_new(m); dd_mul_d(m, D2, D1, Y)         # sinpi(x) * m * y
        PD = dd_new(m); dd_set(m, PD, 'PI_HI', 'PI_LO')
        QD = dd_new(m); dd_div(m, QD, PD, D2)
        scale_general(m, out, QD[0], -n)
    m.release(mk)


# =====================================================================================
# 13.  Reference vectors -- design PART 3, transcribed mechanically from the tables.
#      Category letters are design 3.3:
#        'a' MUST be bit-exact          'b' bit-exact iff the fast path / table exists
#        'c' compare with tolerance (2 ulp for pow, 5 ulp for tgamma)
# =====================================================================================
POW_CAT = {}
for _i in [9, 11, 12, 13] + list(range(21, 53)): POW_CAT[_i] = 'a'
for _i in [1, 3, 4, 7, 8]:                       POW_CAT[_i] = 'b'
for _i in [2, 5, 6, 10] + list(range(14, 21)):   POW_CAT[_i] = 'c'

GAM_CAT = {}
for _i in [27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]: GAM_CAT[_i] = 'a'
for _i in list(range(1, 10)) + [22, 23]:                    GAM_CAT[_i] = 'b'
for _i in list(range(10, 22)) + [24, 25, 26, 28, 29, 30, 31, 43, 44]: GAM_CAT[_i] = 'c'

#         n    x bits              y bits              expected bits       note
POW_VECTORS = [
    ( 1, 0x4000000000000000, 0x4024000000000000, 0x4090000000000000, 'exact power of two, integer y'),
    ( 2, 0x4000000000000000, 0x3FE0000000000000, 0x3FF6A09E667F3BCD, 'sqrt(2) via pow'),
    ( 3, 0x4024000000000000, 0x4008000000000000, 0x408F400000000000, 'pow(10,3): 1000 exactly only with an integer fast path'),
    ( 4, 0x3FE0000000000000, 0xC008000000000000, 0x4020000000000000, 'negative integer y, exact result 8'),
    ( 5, 0x4008000000000000, 0x3FD5555555555555, 0x3FF7137449123EF6, 'cube-root-ish, generic path'),
    ( 6, 0x3FF8000000000000, 0x4004000000000000, 0x40060B9FD68A4554, 'generic non-integer y'),
    ( 7, 0xC000000000000000, 0x4008000000000000, 0xC020000000000000, 'negative base, ODD integer y -> negative result'),
    ( 8, 0xC000000000000000, 0x4010000000000000, 0x4030000000000000, 'negative base, EVEN integer y -> positive result'),
    ( 9, 0xC000000000000000, 0x3FE0000000000000, 0x7FF8000000000000, 'negative base, NON-integer y -> NaN, invalid'),
    (10, 0x4000000000000000, 0x408FFC0000000000, 0x7FE6A09E667F3BCD, 'result near overflow boundary, | y*log2 x | = 1023.5'),
    (11, 0x4000000000000000, 0xC090C80000000000, 0x0000000000000001, 'smallest positive subnormal, exact'),
    (12, 0x7FE1CCF385EBC8A0, 0x4000000000000000, 0x7FF0000000000000, 'overflow -> +inf'),
    (13, 0x000730D67819E8D2, 0x4000000000000000, 0x0000000000000000, 'underflow -> +0'),
    (14, 0x3FEFFFFFFFFFFFFF, 0x430C6BF526340000, 0x3FECA32CBADA6C6A, 'x just below 1, huge y: | y*ln x | ~ 111'),
    (15, 0x3FF0000000000001, 0x4341C37937E08000, 0x40226C41B1A61C92, 'x just above 1, huge y: | y*ln x | ~ 2.22'),
    (16, 0x4004000000000000, 0x4059000000000000, 0x483249AD2594C37D, 'large integer y'),
    (17, 0x401C000000000000, 0xC02A000000000000, 0x3DA6B24188CA33B0, 'negative integer y, generic'),
    (18, 0x01A56E1FC2F8F359, 0xBFF0000000000000, 0x7E37E43C8800759B, 'reciprocal via pow, y=-1'),
    (19, 0x4008000000000000, 0xBFE0000000000000, 0x3FE279A74590331C, 'inverse sqrt of 3'),
    (20, 0x3FF0000000000001, 0x4330000000000000, 0x4005BF0A8B145769, ' | y*ln x | = 1.0, worst conditioning still small'),
    (21, 0x7FF8000000000000, 0x0000000000000000, 0x3FF0000000000000, 'y=0 dominates EVERYTHING, even NaN base'),
    (22, 0x7FF0000000000000, 0x0000000000000000, 0x3FF0000000000000, 'y=0 with infinite base'),
    (23, 0x0000000000000000, 0x0000000000000000, 0x3FF0000000000000, '0^0 = 1 by C99'),
    (24, 0x8000000000000000, 0x8000000000000000, 0x3FF0000000000000, '(-0)^(-0) = 1'),
    (25, 0x3FF0000000000000, 0x7FF8000000000000, 0x3FF0000000000000, 'x=1 dominates NaN exponent'),
    (26, 0x3FF0000000000000, 0x7FF0000000000000, 0x3FF0000000000000, 'x=1 with infinite y'),
    (27, 0xBFF0000000000000, 0x7FF0000000000000, 0x3FF0000000000000, '(-1)^(+inf) = 1'),
    (28, 0xBFF0000000000000, 0xFFF0000000000000, 0x3FF0000000000000, '(-1)^(-inf) = 1'),
    (29, 0xC00A000000000000, 0x3FF0000000000000, 0xC00A000000000000, 'y=1 returns x bit-exactly'),
    (30, 0x0000000000000000, 0x4008000000000000, 0x0000000000000000, '+0 to positive odd int -> +0'),
    (31, 0x8000000000000000, 0x4008000000000000, 0x8000000000000000, '-0 to positive ODD int -> -0 (sign must survive)'),
    (32, 0x8000000000000000, 0x4000000000000000, 0x0000000000000000, '-0 to positive even int -> +0'),
    (33, 0x8000000000000000, 0x3FE0000000000000, 0x0000000000000000, '-0 to positive non-integer -> +0'),
    (34, 0x0000000000000000, 0xC008000000000000, 0x7FF0000000000000, '+0 to negative odd int -> +inf, divide-by-zero'),
    (35, 0x8000000000000000, 0xC008000000000000, 0xFFF0000000000000, '-0 to negative ODD int -> -inf, divide-by-zero'),
    (36, 0x8000000000000000, 0xC000000000000000, 0x7FF0000000000000, '-0 to negative even int -> +inf, divide-by-zero'),
    (37, 0x3FE0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, ' | x | <1, y=-inf -> +inf'),
    (38, 0x3FE0000000000000, 0x7FF0000000000000, 0x0000000000000000, ' | x | <1, y=+inf -> +0'),
    (39, 0x4000000000000000, 0xFFF0000000000000, 0x0000000000000000, ' | x | >1, y=-inf -> +0'),
    (40, 0x4000000000000000, 0x7FF0000000000000, 0x7FF0000000000000, ' | x | >1, y=+inf -> +inf'),
    (41, 0xFFF0000000000000, 0xC008000000000000, 0x8000000000000000, '-inf to negative ODD int -> -0'),
    (42, 0xFFF0000000000000, 0xC000000000000000, 0x0000000000000000, '-inf to negative even int -> +0'),
    (43, 0xFFF0000000000000, 0x4008000000000000, 0xFFF0000000000000, '-inf to positive ODD int -> -inf'),
    (44, 0xFFF0000000000000, 0x4000000000000000, 0x7FF0000000000000, '-inf to positive even int -> +inf'),
    (45, 0xFFF0000000000000, 0x3FE0000000000000, 0x7FF0000000000000, '-inf to positive non-integer -> +inf'),
    (46, 0x7FF0000000000000, 0xC000000000000000, 0x0000000000000000, '+inf to negative y -> +0'),
    (47, 0x7FF0000000000000, 0x4000000000000000, 0x7FF0000000000000, '+inf to positive y -> +inf'),
    (48, 0x7FF8000000000000, 0x4000000000000000, 0x7FF8000000000000, 'NaN base propagates'),
    (49, 0x4000000000000000, 0x7FF8000000000000, 0x7FF8000000000000, 'NaN exponent propagates'),
    (50, 0xC000000000000000, 0x7E37E43C8800759C, 0x7FF0000000000000, '1e300 IS an even integer -> +inf, not NaN'),
    (51, 0xBFF0000000000000, 0x3FE0000000000000, 0x7FF8000000000000, '(-1)^0.5 -> NaN, invalid'),
    (52, 0xC000000000000000, 0x4340000000000000, 0x7FF0000000000000, 'y = 2^53 is an even integer -> +inf'),
]

#         n    x bits              expected bits       note
TGAMMA_VECTORS = [
    ( 1, 0x3FF0000000000000, 0x3FF0000000000000, 'MUST be exactly 1.0'),
    ( 2, 0x4000000000000000, 0x3FF0000000000000, 'MUST be exactly 1.0'),
    ( 3, 0x4008000000000000, 0x4000000000000000, 'exactly 2.0'),
    ( 4, 0x4014000000000000, 0x4038000000000000, 'exactly 24.0 (4!)'),
    ( 5, 0x4018000000000000, 0x405E000000000000, 'exactly 120.0'),
    ( 6, 0x4026000000000000, 0x414BAF8000000000, '10! = 3628800, still exactly representable'),
    ( 7, 0x4032000000000000, 0x42F437EEECD80000, '17! = 355687428096000, exactly representable'),
    ( 8, 0x4033000000000000, 0x4336BEECCA730000, '18! = 6402373705728000, exactly representable'),
    ( 9, 0x4037000000000000, 0x444E77526159F06C, '22! = 1124000727777607680000 IS exactly representable (largest factorial that is); the algorithm is NOT required to hit it'),
    (10, 0x4038000000000000, 0x4495E5C335F8A4CE, '23! is NOT exactly representable -> nearest double'),
    (11, 0x3FE0000000000000, 0x3FFC5BF891B4EF6B, 'sqrt(pi) -- the canonical half-integer'),
    (12, 0x3FF8000000000000, 0x3FEC5BF891B4EF6B, 'sqrt(pi)/2'),
    (13, 0x4004000000000000, 0x3FF544FA6D47B390, '3*sqrt(pi)/4'),
    (14, 0x4012000000000000, 0x40274371E7866C65, 'half-integer, mid range'),
    (15, 0xBFE0000000000000, 0xC00C5BF891B4EF6B, '-2*sqrt(pi), reflection path'),
    (16, 0xBFF8000000000000, 0x4002E7FB0BCDF4F2, '4*sqrt(pi)/3, reflection past one pole'),
    (17, 0xC004000000000000, 0xBFEE3FF812E32183, '-8*sqrt(pi)/15'),
    (18, 0xC00C000000000000, 0x3FD149200ACAEE94, 'reflection, deeper'),
    (19, 0x3FB999999999999A, 0x402306EA7B280D87, 'small positive, recurrence path Gamma(x)=Gamma(x+1)/x'),
    (20, 0x01A56E1FC2F8F359, 0x7E37E43C8800759B, 'tiny x: Gamma(x) ~ 1/x, near overflow of the reciprocal'),
    (21, 0x3FF762D86356BE3F, 0x3FEC56DC82A74AEF, 'argmin of Gamma on (0,inf) (true argmin 1.46163214496836234)'),
    (22, 0x4024000000000000, 0x4116260000000000, '362880 exactly'),
    (23, 0x4034000000000000, 0x437B02B930689000, '19! exact-ish'),
    (24, 0x4059200000000000, 0x6085B98374DB8C0B, 'large half-integer'),
    (25, 0xC059200000000000, 0x9F07932FB5136292, 'large negative non-integer: reflection with | x | >>1'),
    (26, 0xC065500000000000, 0x8017D2374DFCDA7A, 'deep reflection, result still normal'),
    (27, 0xC067100000000000, 0x8000000000000000, 'Gamma is NEGATIVE here; reflection result UNDERFLOWS to -0.0'),
    (28, 0x4065600000000000, 0x7FA4AB7864418639, 'largest integer argument that does not overflow'),
    (29, 0x4065733333333333, 0x7FEC3ADADC5107B1, 'just below the overflow threshold'),
    (30, 0x406573851EB851EC, 0x7FEDB8336BA69B7E, 'closer still to the threshold'),
    (31, 0x406573FAE561F647, 0x7FEFFFFFFFFFFE51, 'LARGEST double x with Gamma(x) finite (0x406573FAE561F647)'),
    (32, 0x406573FAE561F648, 0x7FF0000000000000, 'next double up (0x...F648) -> +inf, overflow'),
    (33, 0x40657428F5C28F5C, 0x7FF0000000000000, 'just above the threshold -> +inf, overflow'),
    (34, 0x4065800000000000, 0x7FF0000000000000, 'integer argument that overflows -> +inf'),
    (35, 0x0000000000000000, 0x7FF0000000000000, 'pole at +0 -> +inf, divide-by-zero'),
    (36, 0x8000000000000000, 0xFFF0000000000000, 'pole at -0 -> -inf, divide-by-zero (sign of zero matters)'),
    (37, 0xBFF0000000000000, 0x7FF8000000000000, 'negative integer -> NaN, invalid'),
    (38, 0xC000000000000000, 0x7FF8000000000000, 'negative integer -> NaN, invalid'),
    (39, 0xC065600000000000, 0x7FF8000000000000, 'large negative integer -> NaN, invalid'),
    (40, 0x7FF8000000000000, 0x7FF8000000000000, 'NaN propagates'),
    (41, 0x7FF0000000000000, 0x7FF0000000000000, '+inf -> +inf'),
    (42, 0xFFF0000000000000, 0x7FF8000000000000, '-inf -> NaN, invalid'),
    (43, 0x4000000000000001, 0x3FF0000000000001, '1 ulp above 2: result must NOT be 1.0'),
    (44, 0x3FEFFFFFFFFFFFFF, 0x3FF0000000000000, '1 ulp below 1'),
]


# =====================================================================================
# 14.  Special-case tables, as executable rows.
#      design 1.5 (pow, C99 Annex F.9.4.4) and design 2.4 (tgamma).
#      Each row names the design row number and supplies concrete operands.  The
#      expectation is taken from libm, which is C99-conforming for exactly these --
#      EXCEPT the tgamma integer-table rows, where a 4th field carries the exactly
#      representable factorial: glibc's tgamma is 1 ulp off on 19!, 21! and 22!, so it
#      is not ground truth for design 2.4 row 8.  Verified in section [6] of the run.
# =====================================================================================
POW_SPECIAL = [
    ( 1, 'y == +0, x = NaN',            NAN,   0.0),
    ( 1, 'y == -0, x = NaN',            NAN,  -0.0),
    ( 1, 'y == +0, x = -0',            -0.0,   0.0),
    ( 1, 'y == +0, x = -inf',          -INF,   0.0),
    ( 1, 'y == -0, x = -2',            -2.0,  -0.0),
    ( 2, 'x == 1, y = NaN',             1.0,   NAN),
    ( 2, 'x == 1, y = +inf',            1.0,   INF),
    ( 2, 'x == 1, y = -inf',            1.0,  -INF),
    ( 3, 'x NaN',                       NAN,   2.0),
    ( 3, 'y NaN',                       2.0,   NAN),
    ( 4, 'y=+inf, |x|<1',               0.5,   INF),
    ( 4, 'y=+inf, |x|<1 (x<0)',        -0.5,   INF),
    ( 5, 'y=+inf, |x|>1',               2.0,   INF),
    ( 6, 'y=-inf, |x|<1',               0.5,  -INF),
    ( 7, 'y=-inf, |x|>1',               2.0,  -INF),
    ( 8, 'x=-1, y=+inf',               -1.0,   INF),
    ( 8, 'x=-1, y=-inf',               -1.0,  -INF),
    ( 9, 'y == 1',                     -3.25,  1.0),
    ( 9, 'y == 1, x = -0',             -0.0,   1.0),
    (10, 'x=+inf, y<0',                 INF,  -2.0),
    (11, 'x=+inf, y>0',                 INF,   2.0),
    (12, 'x=-inf, y<0 odd int',        -INF,  -3.0),
    (13, 'x=-inf, y<0 not odd',        -INF,  -2.0),
    (13, 'x=-inf, y<0 non-integer',    -INF,  -2.5),
    (14, 'x=-inf, y>0 odd int',        -INF,   3.0),
    (15, 'x=-inf, y>0 not odd',        -INF,   2.0),
    (15, 'x=-inf, y>0 non-integer',    -INF,   0.5),
    (16, 'x=+0, y<0 odd int',           0.0,  -3.0),
    (16, 'x=-0, y<0 odd int',          -0.0,  -3.0),
    (17, 'x=+0, y<0 even int',          0.0,  -2.0),
    (17, 'x=-0, y<0 even int',         -0.0,  -2.0),
    (17, 'x=-0, y<0 non-integer',      -0.0,  -2.5),
    (18, 'x=+0, y>0 odd int',           0.0,   3.0),
    (18, 'x=-0, y>0 odd int',          -0.0,   3.0),
    (19, 'x=-0, y>0 even int',         -0.0,   2.0),
    (19, 'x=-0, y>0 non-integer',      -0.0,   0.5),
    (20, 'x<0, y non-integer',         -2.0,   0.5),
    (21, 'x<0, y odd integer',         -2.0,   3.0),
    (21, 'x<0, y even integer',        -2.0,   4.0),
    (21, 'x<0, y = 1e300 (even int)',  -2.0,   1e300),
    (21, 'x<0, y = 2^53 (even int)',   -2.0,   9007199254740992.0),
    (22, 'general',                     3.0,   0.3333333333333333),
]

GAMMA_SPECIAL = [
    ( 1, 'NaN',                          NAN),
    ( 2, '+inf',                         INF),
    ( 3, '-inf -> NaN, invalid',        -INF),
    ( 4, '+0 -> +inf, div-by-zero',      0.0),
    ( 5, '-0 -> -inf, div-by-zero',     -0.0),
    ( 6, 'x = -1 (negative integer)',   -1.0),
    ( 6, 'x = -170 (negative integer)', -170.0),
    ( 6, 'x = -1e300 (even integer)',   -1e300),
    ( 7, 'x just above the threshold',  b2d(0x406573FAE561F648)),
    ( 7, 'x = 172',                      172.0),
    ( 8, 'x = 1 (table)',                1.0,  1.0),
    ( 8, 'x = 20 (table, 19!)',          20.0, float(math.factorial(19))),
    ( 8, 'x = 22 (table, 21!)',          22.0, float(math.factorial(21))),
    ( 8, 'x = 23 (table, 22!)',          23.0, float(math.factorial(22))),
    ( 9, 'x = -184.5 -> -0.0',          -184.5),
    ( 9, 'x = -185.5 -> +0.0',          -185.5),
    ( 9, 'x = -200.5',                  -200.5),
    (10, 'general',                      4.5),
]


# =====================================================================================
# 15.  Test harness
# =====================================================================================
_libm = ctypes.CDLL(ctypes.util.find_library('m'))
_libm.pow.restype = ctypes.c_double;    _libm.pow.argtypes = [ctypes.c_double] * 2
_libm.tgamma.restype = ctypes.c_double; _libm.tgamma.argtypes = [ctypes.c_double]

try:
    import mpmath as _mp
    _mp.mp.dps = 60
except ImportError:
    _mp = None


def _mono(x):
    u = d2b(x)
    return -(u & MASK) if (u >> 63) else u

def ulp_diff(a, b):
    """distance in representable doubles between a and b."""
    if a != a and b != b: return 0.0
    if a != a or b != b:  return INF
    if a == b:            return 0.0
    if math.isinf(a) or math.isinf(b): return INF
    return float(abs(_mono(a) - _mono(b)))

def ulp_err_exact(v, ref):
    """error of v against an exact mpmath reference, in ulp of the true value."""
    if ref == 0: return 0.0
    try:
        r = float(ref)
    except (OverflowError, ValueError):
        return 0.0 if math.isinf(v) else INF
    if math.isinf(r) or r != r:
        return 0.0 if (d2b(v) == d2b(r)) else INF
    if math.isinf(v): return INF
    u = math.ulp(r) if r != 0.0 else 5e-324
    return float(abs(_mp.mpf(v) - ref) / _mp.mpf(u))


def m_pow(x, y):
    mk = M.mark(); X, Y, R = M.alloc(), M.alloc(), M.alloc()
    M.fput(X, x); M.fput(Y, y)
    n0 = M.n; pow_(M, R, X, Y); used = M.n - n0
    v = M.fget(R); M.release(mk)
    return v, used

def m_tgamma(x):
    mk = M.mark(); X, R = M.alloc(), M.alloc()
    M.fput(X, x)
    n0 = M.n; tgamma_(M, R, X); used = M.n - n0
    v = M.fget(R); M.release(mk)
    return v, used


def _budget():
    """measured instruction counts + peak stack slots per routine"""
    rows = []
    def run(name, fn):
        M.peak = M.sp; n0 = M.n; fn(); rows.append((name, M.n - n0, M.peak - M.sp))
    mk = M.mark()
    A, B, C, D = M.alloc(), M.alloc(), M.alloc(), M.alloc()
    M.fput(A, 1.2345678901234); M.fput(B, 9.87654321e-3)
    run('TwoSum(a,b)',        lambda: TwoSum(M, C, D, A, B))
    run('QuickTwoSum(a,b)',   lambda: QuickTwoSum(M, C, D, A, B))
    run('TwoProd(a,b)',       lambda: TwoProd(M, C, D, A, B))
    DD = dd_new(M)
    run('log2_dd(x)',         lambda: log2_dd(M, DD, A))
    M.fput(A, 0.4321)
    run('exp2_core(f)',       lambda: exp2_core(M, C, A))
    M.fput(A, -37.4321)
    run('sinpi(x)',           lambda: sinpi(M, C, A))
    M.release(mk)
    for nm, xy in [('pow(x,y) general', (3.0, 0.3333333333333333)),
                   ('pow(x,y) subnormal out', (2.0, -1073.5)),
                   ('pow(x,y) special row', (0.0, -3.0)),
                   ('pow(x,y) exact int path', (10.0, 3.0))]:
        M.peak = M.sp; n0 = M.n; m_pow(*xy); rows.append((nm, M.n - n0, M.peak - M.sp))
    for nm, x in [('tgamma(x) primary', 8.75), ('tgamma(x) recurrence', 0.3),
                  ('tgamma(x) reflection', -13.4), ('tgamma(x) table row', 11.0),
                  ('tgamma(x) special row', -INF)]:
        M.peak = M.sp; n0 = M.n; m_tgamma(x); rows.append((nm, M.n - n0, M.peak - M.sp))
    return rows


POW_BANDS = [(0.0, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 500.0), (500.0, 709.0),
             (709.0, 745.0)]   # last band reaches the subnormal / overflow tails
GAM_BANDS = [(0.5, 1.5), (1.5, 10.0), (10.0, 50.0), (50.0, 120.0), (120.0, 171.6),
             (-0.5, 0.5), (-5.0, -0.5), (-20.0, -5.0), (-100.0, -20.0), (-175.0, -100.0)]

POW_CLAIM, GAM_CLAIM = 2.0, 5.0


def _sample_pow(band, rng):
    """random (x,y) with |y*ln x| inside the band."""
    lo, hi = band
    while True:
        x = math.exp(rng.uniform(-690.0, 690.0))
        if rng.random() < 0.5: x = 1.0 / x
        L = math.log(x)
        if L == 0.0: continue
        t = rng.uniform(lo, hi) * (1.0 if rng.random() < 0.5 else -1.0)
        y = t / L
        if y == 0.0 or math.isinf(y) or abs(y) >= 9.223372036854776e18: continue
        if rng.random() < 0.35: x = -x            # exercise the sign path
        return x, y


def main(argv):
    n = 50000
    if len(argv) > 1: n = int(argv[1])
    rng = random.Random(20260823)
    print('=' * 86)
    print('ffi_pow_oracle -- op-for-op model of the native pow/tgamma emitter')
    print('design: docs/audit/NATIVE_POW_TGAMMA_NUMERICAL_DESIGN_2026-08-23.md')
    print('=' * 86)

    # --- constants ---------------------------------------------------------------------
    print('\n[1] rodata self-check')
    print('    sqrt(2pi)  %s  correctly rounded: %s' %
          (hex(d2b(K['SQRT2PI'])), K['SQRT2PI'] == _rnd(S2PI_F)))
    print('    pi         %s  correctly rounded: %s' %
          (hex(d2b(K['PI'])), K['PI'] == _rnd(PI_F)))
    print('    log2(e)    %s  correctly rounded: %s' %
          (hex(d2b(K['LOG2E'])), K['LOG2E'] == _rnd(LOG2E_F)))
    print('    SPLIT      %s  == 2^27+1: %s' %
          (hex(d2b(K['SPLIT'])), K['SPLIT'] == 2.0 ** 27 + 1.0))
    print('    Lanczos g=31/8 N=13: hex column vs decimal column mismatches: %s'
          % (LANCZOS_HEX_VS_DEC or 'none'))
    print('    composed sqrt(fl(2pi)) = %s  (design says this is 1 ulp low -- %s)'
          % (hex(d2b(math.sqrt(2.0 * K['PI']))),
             'confirmed' if d2b(math.sqrt(2.0 * K['PI'])) == d2b(K['SQRT2PI']) - 1
             else 'NOT confirmed'))

    print('\n    polynomial coefficients (all rounded ONCE from an exact rational):')
    print('      log2_dd atanh tail  C2(u) = sum_{j=1..%d} u^j/(2j+3),  u = s^2' % ATANH_NC)
    for j, c in enumerate(K['ATANH']):
        print('        1/%-3d  %-24r %s' % (2 * (j + 1) + 3, c, hex(d2b(c))))
    print('      exp2 kernel        E(u) = sum_{j=0..%d} u^j/(j+2)!,  u = f*ln2' % EXP2_NE)
    for j, c in enumerate(K['EXPE']):
        print('        1/%-4s %-24r %s' % ('%d!' % (j + 2), c, hex(d2b(c))))
    print('      sinpi              Q(v) = sum_{j=1..%d} (-1)^j v^j/(2j+1)!' % SINQ_NJ)
    for j, c in enumerate(K['SINQ']):
        print('        j=%-3d  %-24r %s' % (j + 1, c, hex(d2b(c))))
    print('      cospi              C(v) = sum_{j=1..%d} (-1)^j v^(j-1)/(2j)!' % COSQ_NJ)
    for j, c in enumerate(K['COSQ']):
        print('        j=%-3d  %-24r %s' % (j + 1, c, hex(d2b(c))))
    print('      double-double constants (hi, lo):')
    for nm in ('LN2', 'LOG2E', 'THIRD', 'TL2E', 'PI'):
        print('        %-8s %-24r %r' % (nm, K[nm + '_HI'], K[nm + '_LO']))
    print('      Lanczos g = 31/8, N = 13 (design 2.2, taken as bit patterns):')
    for j, c in enumerate(K['LZ']):
        print('        c%-3d  %-24r %s' % (j, c, hex(d2b(c))))

    # --- instruction budget ---------------------------------------------------------
    print('\n[2] instruction budget (dynamic; unrolled polynomials => static == dynamic')
    print('    except the pow integer fast path, which is a loop)')
    print('    %-26s %8s  %s' % ('routine', 'instr', 'peak 8B stack slots'))
    for nm, ni, ns in _budget():
        print('    %-26s %8d  %d' % (nm, ni, ns))
    print('    NOTE: a frame over 32 slots cannot be addressed with rbp-disp8.')

    # --- reference vectors -------------------------------------------------------------
    print('\n[3] reference vectors, design 3.1 -- pow, 52 rows')
    fails = []
    for i, xb, yb, eb, note in POW_VECTORS:
        x, y, e = b2d(xb), b2d(yb), b2d(eb)
        v, _ = m_pow(x, y)
        cat = POW_CAT[i]
        u = ulp_diff(v, e)
        if cat in 'ab':
            ok = (d2b(v) == eb) or (v != v and e != e and d2b(v) == d2b(e))
            if not ok and v != v and e != e: ok = True
            status = 'exact' if ok else 'FAIL(bit)'
        else:
            ok = u <= POW_CLAIM
            status = '%.2f ulp' % u if ok else 'FAIL(%.2f ulp)' % u
        if not ok: fails.append(('pow vec %d' % i, status))
        print('    %2d [%s] %-28s -> %-22s %s' % (i, cat, note[:28], hex(d2b(v)), status))

    print('\n[4] reference vectors, design 3.2 -- tgamma, 44 rows')
    for i, xb, eb, note in TGAMMA_VECTORS:
        x, e = b2d(xb), b2d(eb)
        v, _ = m_tgamma(x)
        cat = GAM_CAT[i]
        u = ulp_diff(v, e)
        if cat in 'ab':
            ok = (d2b(v) == eb) or (v != v and e != e)
            status = 'exact' if ok else 'FAIL(bit %s want %s)' % (hex(d2b(v)), hex(eb))
        else:
            ok = u <= GAM_CLAIM
            status = '%.2f ulp' % u if ok else 'FAIL(%.2f ulp)' % u
        if not ok: fails.append(('tgamma vec %d' % i, status))
        print('    %2d [%s] %-28s -> %-22s %s' % (i, cat, note[:28], hex(d2b(v)), status))

    # --- special-case tables against libm ---------------------------------------------
    print('\n[5] special cases, design 1.5 (pow) -- expectation from libm via ctypes')
    for row, note, x, y in POW_SPECIAL:
        v, _ = m_pow(x, y)
        r = _libm.pow(x, y)
        ok = (d2b(v) == d2b(r)) or (v != v and r != r)
        if not ok: fails.append(('pow special row %d (%s)' % (row, note),
                                 '%s vs libm %s' % (hex(d2b(v)), hex(d2b(r)))))
        print('    row %2d  %-28s pow(%r,%r) = %-22s %s'
              % (row, note, x, y, hex(d2b(v)), 'ok' if ok else 'FAIL libm=%s' % hex(d2b(r))))

    print('\n[6] special cases, design 2.4 (tgamma) -- expectation from libm via ctypes')
    for spec in GAMMA_SPECIAL:
        row, note, x = spec[0], spec[1], spec[2]
        r = _libm.tgamma(x)
        want = spec[3] if len(spec) > 3 else r
        v, _ = m_tgamma(x)
        ok = (d2b(v) == d2b(want)) or (v != v and want != want)
        extra = ''
        if len(spec) > 3 and d2b(want) != d2b(r):
            extra = '   [libm disagrees: %s -- glibc is not exact here]' % hex(d2b(r))
        if not ok: fails.append(('tgamma special row %d (%s)' % (row, note),
                                 '%s want %s' % (hex(d2b(v)), hex(d2b(want)))))
        print('    row %2d  %-28s tgamma(%r) = %-22s %s%s'
              % (row, note, x, hex(d2b(v)),
                 'ok' if ok else 'FAIL want=%s' % hex(d2b(want)), extra))

    # --- random sweeps -----------------------------------------------------------------
    per = max(1, n // len(POW_BANDS))
    print('\n[7] pow -- %d random samples (%d per |y*ln x| band)' % (per * len(POW_BANDS), per))
    print('    %-14s %10s %10s %10s' % ('|y*ln x| band', 'vs libm', 'vs exact', 'worst x,y'))
    pow_worst_libm = 0.0; pow_worst_exact = 0.0; pow_worst_at = None
    for band in POW_BANDS:
        wl = we = 0.0; wat = None
        for _ in range(per):
            x, y = _sample_pow(band, rng)
            v, _ = m_pow(x, y)
            r = _libm.pow(x, y)
            if math.isinf(r) or r != r:
                continue                      # r == 0.0 is KEPT: subnormal tail
            u = ulp_diff(v, r)
            if u > wl: wl = u
            if _mp is not None:
                ref = _mp.power(_mp.mpf(x), _mp.mpf(y))
                ue = ulp_err_exact(v, ref)
                if ue > we: we, wat = ue, (x, y)
        pow_worst_libm = max(pow_worst_libm, wl)
        if we > pow_worst_exact: pow_worst_exact, pow_worst_at = we, wat
        print('    %-14s %10.3f %10.3f  %r' % ('%g - %g' % band, wl, we, wat))

    per = max(1, n // len(GAM_BANDS))
    print('\n[8] tgamma -- %d random samples (%d per band)' % (per * len(GAM_BANDS), per))
    print('    %-16s %10s %10s %10s %10s' % ('x band', 'vs libm', 'vs exact', 'libm err', 'worst x'))
    gam_worst_libm = 0.0; gam_worst_exact = 0.0; gam_worst_at = None; libm_own = 0.0
    for band in GAM_BANDS:
        wl = we = wo = 0.0; wat = None
        for _ in range(per):
            x = rng.uniform(*band)
            if x == 0.0: continue
            v, _ = m_tgamma(x)
            r = _libm.tgamma(x)
            if math.isinf(r) or r == 0.0 or r != r: continue
            u = ulp_diff(v, r)
            if u > wl: wl = u
            if _mp is not None:
                ref = _mp.gamma(_mp.mpf(x))
                ue = ulp_err_exact(v, ref)
                if ue > we: we, wat = ue, x
                wo = max(wo, ulp_err_exact(r, ref))
        gam_worst_libm = max(gam_worst_libm, wl)
        libm_own = max(libm_own, wo)
        if we > gam_worst_exact: gam_worst_exact, gam_worst_at = we, wat
        print('    %-16s %10.3f %10.3f %10.3f  %r'
              % ('%g .. %g' % band, wl, we, wo, wat))

    # --- verdict -------------------------------------------------------------------------
    print('\n' + '=' * 86)
    print('[9] verdict')
    if _mp is None:
        print('    mpmath is NOT installed -- the only reference available is libm, and')
        print('    glibc tgamma is itself several ulp off, so the tgamma number below is')
        print('    (our error + libm error) and is an UPPER BOUND, not a measurement.')
        pw, gw, refname = pow_worst_libm, gam_worst_libm, 'libm'
    else:
        print('    reference: mpmath at 60 digits (the reference the design itself used).')
        print('    libm is reported alongside; glibc tgamma is itself up to %.2f ulp off'
              % libm_own)
        print('    over these bands, so the "vs libm" column is our error PLUS libm\'s.')
        pw, gw, refname = pow_worst_exact, gam_worst_exact, 'mpmath/60d'
    print('    pow    max %.3f ulp vs %s   (claim <= %.1f)  at %r'
          % (pw, refname, POW_CLAIM, pow_worst_at))
    print('    tgamma max %.3f ulp vs %s   (claim <= %.1f)  at %r'
          % (gw, refname, GAM_CLAIM, gam_worst_at))
    print('    pow    max %.3f ulp vs libm' % pow_worst_libm)
    print('    tgamma max %.3f ulp vs libm' % gam_worst_libm)
    if fails:
        print('\n    %d vector/special-case failures:' % len(fails))
        for a, b in fails: print('      %-40s %s' % (a, b))
    ok = (not fails) and pw <= POW_CLAIM and gw <= GAM_CLAIM
    print('\n    RESULT: %s' % ('PASS' if ok else 'FAIL'))
    print('=' * 86)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
