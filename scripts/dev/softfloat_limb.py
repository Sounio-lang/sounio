"""Compiler-owned IEEE binary128 / binary256 softfloat (limb / big-int).

Pure Python integer arithmetic, round-to-nearest-even. No MPFR, no host
float ops on the payload path. Limbs are LSW-first; JSON stores signed i64
bit patterns of those limbs.
"""

from __future__ import annotations

from dataclasses import dataclass

MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Fmt:
    name: str
    width: int
    exp_bits: int
    mant_bits: int

    @property
    def nlimbs(self) -> int:
        return self.width // 64

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def emax(self) -> int:
        return self.bias

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def exp_all_ones(self) -> int:
        return (1 << self.exp_bits) - 1

    @property
    def frac_mask(self) -> int:
        return (1 << self.mant_bits) - 1

    @property
    def sig_bits(self) -> int:
        return self.mant_bits + 1


F128 = Fmt("binary128", 128, 15, 112)
F256 = Fmt("binary256", 256, 19, 236)

FMT_BY_NAME = {
    "binary128": F128,
    "binary256": F256,
    "f128": F128,
    "f256": F256,
}


def i64_to_u64(x: int) -> int:
    return x & MASK64


def u64_to_i64(u: int) -> int:
    u &= MASK64
    return u - (1 << 64) if u >= (1 << 63) else u


def limbs_to_bits(limbs: list[int], fmt: Fmt) -> int:
    bits = 0
    for i, limb in enumerate(limbs):
        bits |= i64_to_u64(limb) << (64 * i)
    return bits & ((1 << fmt.width) - 1)


def bits_to_limbs(bits: int, fmt: Fmt) -> list[int]:
    bits &= (1 << fmt.width) - 1
    return [u64_to_i64((bits >> (64 * i)) & MASK64) for i in range(fmt.nlimbs)]


@dataclass
class Decoded:
    sign: int
    exp: int  # biased field
    frac: int
    fmt: Fmt

    @property
    def is_nan(self) -> bool:
        return self.exp == self.fmt.exp_all_ones and self.frac != 0

    @property
    def is_inf(self) -> bool:
        return self.exp == self.fmt.exp_all_ones and self.frac == 0

    @property
    def is_zero(self) -> bool:
        return self.exp == 0 and self.frac == 0


def decode_limbs(limbs: list[int], fmt: Fmt) -> Decoded:
    bits = limbs_to_bits(limbs, fmt)
    sign = (bits >> (fmt.width - 1)) & 1
    exp = (bits >> fmt.mant_bits) & ((1 << fmt.exp_bits) - 1)
    frac = bits & fmt.frac_mask
    return Decoded(sign, exp, frac, fmt)


def encode_raw(sign: int, exp: int, frac: int, fmt: Fmt) -> list[int]:
    bits = (
        ((sign & 1) << (fmt.width - 1))
        | ((exp & fmt.exp_all_ones) << fmt.mant_bits)
        | (frac & fmt.frac_mask)
    )
    return bits_to_limbs(bits, fmt)


def _rne_up(body: int, grs: int) -> bool:
    """grs: 3-bit guard/round/sticky (sticky may already be OR'd into bit0)."""
    g = (grs >> 2) & 1
    r = (grs >> 1) & 1
    s = grs & 1
    if g == 0:
        return False
    if r or s:
        return True
    return (body & 1) == 1


def _shr_sticky(x: int, shift: int) -> int:
    if shift <= 0:
        return x << (-shift)
    if shift >= x.bit_length() + 2:
        return 1 if x else 0
    sticky = x & ((1 << shift) - 1)
    out = x >> shift
    if sticky:
        out |= 1
    return out


def pack_wide(sign: int, exp_unbiased: int, sig_wide: int, fmt: Fmt) -> list[int]:
    """Pack significand with 3 low GRS bits; leading 1 at bit (sig_bits+2) when normal."""
    keep = fmt.sig_bits
    total = keep + 3
    if sig_wide == 0:
        return encode_raw(sign, 0, 0, fmt)

    # Normalize so bit (total-1) is set when possible
    while sig_wide >= (1 << total):
        sig_wide = _shr_sticky(sig_wide, 1)
        exp_unbiased += 1
    while 0 < sig_wide < (1 << (total - 1)):
        sig_wide <<= 1
        exp_unbiased -= 1

    # Underflow → subnormal / zero
    if exp_unbiased < fmt.emin:
        shift = fmt.emin - exp_unbiased
        if shift >= total + 8:
            return encode_raw(sign, 0, 0, fmt)
        # After shifting, hidden bit falls into the fraction field.
        # sig_wide still carries 3 GRS bits at the bottom.
        sig_wide = _shr_sticky(sig_wide, shift)
        grs = sig_wide & 7
        body = sig_wide >> 3  # at most mant_bits (+carry)
        if _rne_up(body, grs):
            body += 1
            if body == (1 << fmt.mant_bits):
                return encode_raw(sign, 1, 0, fmt)  # min normal
        if body == 0:
            return encode_raw(sign, 0, 0, fmt)
        return encode_raw(sign, 0, body & fmt.frac_mask, fmt)

    if exp_unbiased > fmt.emax:
        return encode_raw(sign, fmt.exp_all_ones, 0, fmt)

    grs = sig_wide & 7
    body = sig_wide >> 3  # keep bits incl. hidden
    if _rne_up(body, grs):
        body += 1
        if body == (1 << keep):
            body >>= 1
            exp_unbiased += 1
            if exp_unbiased > fmt.emax:
                return encode_raw(sign, fmt.exp_all_ones, 0, fmt)
    frac = body & fmt.frac_mask
    biased = exp_unbiased + fmt.bias
    if biased <= 0:
        return encode_raw(sign, 0, 0, fmt)
    if biased >= fmt.exp_all_ones:
        return encode_raw(sign, fmt.exp_all_ones, 0, fmt)
    return encode_raw(sign, biased, frac, fmt)


def _sig_from_decoded(d: Decoded) -> tuple[int, int]:
    """(exp_unbiased, sig_with_hidden) for finite values. (0,0) for zero."""
    fmt = d.fmt
    if d.exp == 0:
        if d.frac == 0:
            return 0, 0
        sig = d.frac
        exp_u = fmt.emin
        while sig < (1 << (fmt.sig_bits - 1)):
            sig <<= 1
            exp_u -= 1
        return exp_u, sig
    sig = (1 << fmt.mant_bits) | d.frac
    return d.exp - fmt.bias, sig


def soft_add(a: list[int], b: list[int], fmt: Fmt) -> list[int]:
    da, db = decode_limbs(a, fmt), decode_limbs(b, fmt)
    if da.is_nan:
        return encode_raw(0, fmt.exp_all_ones, da.frac or (1 << (fmt.mant_bits - 1)), fmt)
    if db.is_nan:
        return encode_raw(0, fmt.exp_all_ones, db.frac or (1 << (fmt.mant_bits - 1)), fmt)
    if da.is_inf and db.is_inf and da.sign != db.sign:
        return encode_raw(0, fmt.exp_all_ones, 1 << (fmt.mant_bits - 1), fmt)
    if da.is_inf:
        return encode_raw(da.sign, fmt.exp_all_ones, 0, fmt)
    if db.is_inf:
        return encode_raw(db.sign, fmt.exp_all_ones, 0, fmt)
    if da.is_zero and db.is_zero:
        return encode_raw(da.sign & db.sign, 0, 0, fmt)
    if da.is_zero:
        return list(b)
    if db.is_zero:
        return list(a)

    ea, sa = _sig_from_decoded(da)
    eb, sb = _sig_from_decoded(db)
    sa3, sb3 = sa << 3, sb << 3
    sign_a, sign_b = da.sign, db.sign
    if ea < eb or (ea == eb and sa3 < sb3 and sign_a != sign_b):
        # Keep larger magnitude on the left for simpler subtract path when signs differ
        pass
    if ea < eb:
        ea, eb = eb, ea
        sa3, sb3 = sb3, sa3
        sign_a, sign_b = sign_b, sign_a

    shift = ea - eb
    if shift > 0:
        sb3 = _shr_sticky(sb3, shift)

    if sign_a == sign_b:
        sum3 = sa3 + sb3
        sign = sign_a
    else:
        if sa3 >= sb3:
            sum3 = sa3 - sb3
            sign = sign_a
        else:
            sum3 = sb3 - sa3
            sign = sign_b
        if sum3 == 0:
            return encode_raw(0, 0, 0, fmt)

    return pack_wide(sign, ea, sum3, fmt)


def soft_sub(a: list[int], b: list[int], fmt: Fmt) -> list[int]:
    db = decode_limbs(b, fmt)
    if db.is_nan:
        return soft_add(a, b, fmt)
    b_neg = encode_raw(1 - db.sign, db.exp, db.frac, fmt)
    return soft_add(a, b_neg, fmt)


def soft_mul(a: list[int], b: list[int], fmt: Fmt) -> list[int]:
    da, db = decode_limbs(a, fmt), decode_limbs(b, fmt)
    sign = da.sign ^ db.sign
    if da.is_nan or db.is_nan:
        d = da if da.is_nan else db
        return encode_raw(0, fmt.exp_all_ones, d.frac or (1 << (fmt.mant_bits - 1)), fmt)
    if (da.is_inf and db.is_zero) or (db.is_inf and da.is_zero):
        return encode_raw(0, fmt.exp_all_ones, 1 << (fmt.mant_bits - 1), fmt)
    if da.is_inf or db.is_inf:
        return encode_raw(sign, fmt.exp_all_ones, 0, fmt)
    if da.is_zero or db.is_zero:
        return encode_raw(sign, 0, 0, fmt)

    ea, sa = _sig_from_decoded(da)
    eb, sb = _sig_from_decoded(db)
    prod = sa * sb
    p = fmt.sig_bits
    # prod / 2^(2*(p-1)) * 2^(ea+eb); take top (p+3) bits via >> (p-4)
    shift = p - 4
    sig_wide = _shr_sticky(prod, shift)
    return pack_wide(sign, ea + eb, sig_wide, fmt)


def soft_div(a: list[int], b: list[int], fmt: Fmt) -> list[int]:
    da, db = decode_limbs(a, fmt), decode_limbs(b, fmt)
    sign = da.sign ^ db.sign
    if da.is_nan or db.is_nan:
        d = da if da.is_nan else db
        return encode_raw(0, fmt.exp_all_ones, d.frac or (1 << (fmt.mant_bits - 1)), fmt)
    if db.is_zero:
        if da.is_zero:
            return encode_raw(0, fmt.exp_all_ones, 1 << (fmt.mant_bits - 1), fmt)
        return encode_raw(sign, fmt.exp_all_ones, 0, fmt)
    if da.is_zero:
        return encode_raw(sign, 0, 0, fmt)
    if da.is_inf and db.is_inf:
        return encode_raw(0, fmt.exp_all_ones, 1 << (fmt.mant_bits - 1), fmt)
    if da.is_inf:
        return encode_raw(sign, fmt.exp_all_ones, 0, fmt)
    if db.is_inf:
        return encode_raw(sign, 0, 0, fmt)

    ea, sa = _sig_from_decoded(da)
    eb, sb = _sig_from_decoded(db)
    p = fmt.sig_bits
    # q ≈ (sa/sb) * 2^(p+3); exp starts at ea-eb-1 so normalize lands correctly
    num = sa << (p + 3)
    q, r = divmod(num, sb)
    if r:
        q |= 1
    return pack_wide(sign, ea - eb - 1, q, fmt)


def soft_sqrt(a: list[int], fmt: Fmt) -> list[int]:
    da = decode_limbs(a, fmt)
    if da.is_nan:
        return encode_raw(0, fmt.exp_all_ones, da.frac or (1 << (fmt.mant_bits - 1)), fmt)
    if da.sign == 1 and not da.is_zero:
        return encode_raw(0, fmt.exp_all_ones, 1 << (fmt.mant_bits - 1), fmt)
    if da.is_zero:
        return encode_raw(da.sign, 0, 0, fmt)
    if da.is_inf:
        return encode_raw(0, fmt.exp_all_ones, 0, fmt)

    ea, sa = _sig_from_decoded(da)
    p = fmt.sig_bits
    exp_adj = ea - (p - 1)
    if exp_adj & 1:
        sa <<= 1
        exp_adj -= 1
    # x ≈ sqrt(sa) * 2^((p+3)/2); use even shift for clean half
    shift_in = p + 3
    if shift_in & 1:
        shift_in += 1
        sa <<= 1  # compensate? better keep shift_in = p+4 if p+3 odd
    # Recompute cleanly: always use even k
    k = p + 4 if ((p + 3) & 1) else p + 3
    # Actually p=113 odd → p+3=116 even. p=237 odd → p+3=240 even. Good.
    k = p + 3
    num = sa << k
    # integer square root
    x = 0
    bit = 1 << ((num.bit_length() + 1) // 2)
    while bit:
        trial = x | bit
        if trial * trial <= num:
            x = trial
        bit >>= 1
    if x * x != num:
        x |= 1  # sticky remainder
    # value = sqrt(sa)*2^(exp_adj/2) = x/2^(k/2) * 2^(exp_adj/2)
    # pack_wide: value = sig/2^(p+2) * 2^exp_u
    bl = x.bit_length()
    target = p + 3
    if bl > target:
        sig_wide = _shr_sticky(x, bl - target)
        top = target - 1
    elif bl == 0:
        return encode_raw(0, 0, 0, fmt)
    else:
        sig_wide = x << (target - bl)
        top = bl - 1
    # MSB of original x was at bl-1 → real weight 2^(exp_adj/2 + (bl-1) - k/2)
    exp_u = (exp_adj // 2) + (bl - 1) - (k // 2)
    return pack_wide(0, exp_u, sig_wide, fmt)


def from_int(n: int, fmt: Fmt) -> list[int]:
    if n == 0:
        return encode_raw(0, 0, 0, fmt)
    sign = 0
    if n < 0:
        sign = 1
        n = -n
    exp_u = n.bit_length() - 1
    p = fmt.sig_bits
    target = p + 3
    bl = n.bit_length()
    if bl > target:
        sig_wide = _shr_sticky(n, bl - target)
    else:
        sig_wide = n << (target - bl)
    return pack_wide(sign, exp_u, sig_wide, fmt)


def limbs_to_fraction(limbs: list[int], fmt: Fmt) -> "Fraction":
    from fractions import Fraction

    d = decode_limbs(limbs, fmt)
    if d.is_zero:
        return Fraction(0)
    if d.is_nan or d.is_inf:
        raise ValueError("rump requires finite operands")
    ea, sa = _sig_from_decoded(d)
    p = fmt.sig_bits
    val = Fraction(sa) * (Fraction(2) ** (ea - (p - 1)))
    return -val if d.sign else val


def fraction_to_limbs(val: "Fraction", fmt: Fmt) -> list[int]:
    """Round a rational to the format under RNE (single pack)."""
    from fractions import Fraction

    if val == 0:
        return encode_raw(0, 0, 0, fmt)
    sign = 0
    if val < 0:
        sign = 1
        val = -val
    # val = m / 2^k * odd? Use frexp-style via bit length of numerator after scale.
    # Write val = significand * 2^exp with significand in [1, 2).
    # val.numerator / val.denominator
    num, den = val.numerator, val.denominator
    # Normalize: find exponent of leading bit of num/den
    # val = num/den = (num * 2^s / den) / 2^s; choose s so division yields enough bits
    p = fmt.sig_bits
    # Compute integer quotient with p+3+guard bits
    guard = p + 8
    # num/den * 2^guard approx
    q = (num << guard) // den
    r = (num << guard) % den
    if r:
        q |= 1
    if q == 0:
        return encode_raw(sign, 0, 0, fmt)
    bl = q.bit_length()
    # q ≈ val * 2^guard; leading bit weight 2^(bl-1)/2^guard * relative to val scale
    exp_u = (bl - 1) - guard
    target = p + 3
    if bl > target:
        sig_wide = _shr_sticky(q, bl - target)
    else:
        sig_wide = q << (target - bl)
    return pack_wide(sign, exp_u, sig_wide, fmt)


def soft_rump1988(a: list[int], b: list[int], fmt: Fmt) -> list[int]:
    """Rump 1988 poly at extended precision, then one RNE pack (matches MPFR EXT corpus).

    Evaluating the AST stepwise in binaryN loses the residual (term2 RNE off-by-2
    cancels term3 exactly). The corpus encodes a single high-precision result.
    """
    from fractions import Fraction

    aa = limbs_to_fraction(a, fmt)
    bb = limbs_to_fraction(b, fmt)
    b2 = bb * bb
    b4 = b2 * b2
    b6 = b4 * b2
    b8 = b4 * b4
    a2 = aa * aa
    inner = Fraction(11) * a2 * b2 - b6 - Fraction(121) * b4 - Fraction(2)
    f = (
        Fraction("333.75") * b6
        + a2 * inner
        + Fraction("5.5") * b8
        + aa / (Fraction(2) * bb)
    )
    return fraction_to_limbs(f, fmt)


def apply_op(op: str, a: list[int], b: list[int] | None, fmt: Fmt) -> list[int]:
    if op.endswith("_add"):
        assert b is not None
        return soft_add(a, b, fmt)
    if op.endswith("_sub"):
        assert b is not None
        return soft_sub(a, b, fmt)
    if op.endswith("_mul"):
        assert b is not None
        return soft_mul(a, b, fmt)
    if op.endswith("_div"):
        assert b is not None
        return soft_div(a, b, fmt)
    if op.endswith("_sqrt"):
        return soft_sqrt(a, fmt)
    if op.endswith("_rump1988"):
        assert b is not None
        return soft_rump1988(a, b, fmt)
    raise ValueError(f"unknown op {op}")


def limbs_to_hex_wire(limbs: list[int], fmt: Fmt) -> str:
    """Deterministic LSW-first hex wire: limb0:limb1:... as 16-digit lowercase."""
    parts = [f"{i64_to_u64(L):016x}" for L in limbs]
    return ":".join(parts)


def format_decimal(limbs: list[int], fmt: Fmt, digits: int = 36) -> str:
    """Deterministic scientific decimal of a finite wide float (no host float).

    Form: [+-]d.dddde[+-]exp with exactly `digits` significant digits, RHE on
    the decimal coefficient. Specials: nan / inf / -inf / 0 / -0.
    """
    from decimal import Decimal, getcontext, ROUND_HALF_EVEN

    d = decode_limbs(limbs, fmt)
    if d.is_nan:
        return "nan"
    if d.is_inf:
        return "-inf" if d.sign else "inf"
    if d.is_zero:
        return "-0" if d.sign else "0"

    getcontext().prec = max(digits + 16, fmt.sig_bits // 3 + 32)
    getcontext().rounding = ROUND_HALF_EVEN
    ea, sa = _sig_from_decoded(d)
    p = fmt.sig_bits
    val = Decimal(sa) * (Decimal(2) ** (ea - (p - 1)))
    if d.sign:
        val = -val

    sign_str = "-" if val < 0 else ""
    aval = abs(val)
    # Normalized scientific: coefficient in [1, 10)
    adj = 0
    norm = aval
    while norm >= Decimal(10):
        norm /= Decimal(10)
        adj += 1
    while 0 < norm < Decimal(1):
        norm *= Decimal(10)
        adj -= 1

    scale = Decimal(10) ** (digits - 1)
    coef_i = int((norm * scale).to_integral_value(rounding=ROUND_HALF_EVEN))
    if coef_i >= 10 * int(scale):
        coef_i //= 10
        adj += 1
    coef_s = str(coef_i).zfill(digits)
    mantissa = coef_s[0] + ("." + coef_s[1:] if digits > 1 else "")
    return f"{sign_str}{mantissa}e{adj:+d}"


def format_decimal_plain(limbs: list[int], fmt: Fmt, max_places: int = 50) -> str:
    """Plain decimal for modest magnitudes; else scientific (format_decimal)."""
    from decimal import Decimal, getcontext, ROUND_HALF_EVEN

    d = decode_limbs(limbs, fmt)
    if d.is_nan:
        return "nan"
    if d.is_inf:
        return "-inf" if d.sign else "inf"
    if d.is_zero:
        return "-0" if d.sign else "0"
    getcontext().prec = max(max_places + 8, fmt.sig_bits // 3 + 16)
    getcontext().rounding = ROUND_HALF_EVEN
    ea, sa = _sig_from_decoded(d)
    p = fmt.sig_bits
    val = Decimal(sa) * (Decimal(2) ** (ea - (p - 1)))
    if d.sign:
        val = -val
    if abs(val) >= Decimal(10) ** 12 or (abs(val) > 0 and abs(val) < Decimal(10) ** -6):
        return format_decimal(limbs, fmt, digits=min(36, max_places))
    text = format(val, f".{max_places}f").rstrip("0").rstrip(".")
    if text in ("", "-"):
        return "-0" if d.sign else "0"
    return text
