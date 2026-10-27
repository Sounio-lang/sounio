#!/usr/bin/env python3
"""Independent exact-integer oracle for the Cayley-Dickson tower API.

No repository code is imported. The XOR/sign convolution and the recursive
Dickson-doubling formula are implemented independently and compared through
level 6 (64 dimensions). Basis associators are also enumerated exhaustively.
"""

from __future__ import annotations


def sigma(a: int, b: int, bits: int) -> int:
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi, b_hi = a >= half, b >= half
    a_lo, b_lo = a & (half - 1), b & (half - 1)
    if not a_hi and not b_hi:
        return sigma(a_lo, b_lo, bits - 1)
    if not a_hi and b_hi:
        return sigma(b_lo, a_lo, bits - 1)
    if a_hi and not b_hi:
        if b_lo == 0:
            return sigma(a_lo, 0, bits - 1)
        return -sigma(a_lo, b_lo, bits - 1)
    if b_lo == 0:
        return -sigma(0, a_lo, bits - 1)
    return sigma(b_lo, a_lo, bits - 1)


def add(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b)]


def sub(a: list[int], b: list[int]) -> list[int]:
    return [x - y for x, y in zip(a, b)]


def scale(a: list[int], scalar: int) -> list[int]:
    return [scalar * x for x in a]


def conj(a: list[int]) -> list[int]:
    return [a[0], *[-x for x in a[1:]]]


def mul(a: list[int], b: list[int], bits: int) -> list[int]:
    assert len(a) == len(b) == 1 << bits
    result = [0] * (1 << bits)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if y != 0:
                result[i ^ j] += sigma(i, j, bits) * x * y
    return result


def split(a: list[int]) -> tuple[list[int], list[int]]:
    assert len(a) >= 2 and len(a) & (len(a) - 1) == 0
    half = len(a) // 2
    return a[:half], a[half:]


def mul_doubling(a: list[int], b: list[int], bits: int) -> list[int]:
    """Recursive Dickson doubling down to a real-scalar base case.

    This evaluator shares no code with `mul` or `sigma`: every level recurses
    through `mul_doubling` itself, so a defect in the XOR/sign kernel cannot be
    inherited by the route it is compared against (and vice versa).
    """
    assert len(a) == len(b) == 1 << bits
    if bits == 0:
        return [a[0] * b[0]]
    a_lo, a_hi = split(a)
    b_lo, b_hi = split(b)
    lower = bits - 1
    # Independent Dickson presentation: (a,b)(c,d)=(ac-conj(d)b, da+b conj(c)).
    low = sub(mul_doubling(a_lo, b_lo, lower), mul_doubling(conj(b_hi), a_hi, lower))
    high = add(mul_doubling(b_hi, a_lo, lower), mul_doubling(a_hi, conj(b_lo), lower))
    return low + high


def embed(a: list[int], bits: int) -> list[int]:
    assert bits >= len(a).bit_length() - 1
    return a + [0] * ((1 << bits) - len(a))


def vector(bits: int, seed: int) -> list[int]:
    return [(i * 17 + seed * 11 + bits * 7) % 7 - 3 for i in range(1 << bits)]


def basis_wave(i: int, j: int, k: int, bits: int) -> int:
    alpha = sigma(i, j, bits) * sigma(i ^ j, k, bits)
    beta = sigma(j, k, bits) * sigma(i, j ^ k, bits)
    return alpha - beta


def main() -> None:
    product_cases = 0
    basis_triples = 0
    nonassociative = {}
    for bits in range(1, 7):
        x, y = vector(bits, 2), vector(bits, 5)
        assert mul_doubling(x, y, bits) == mul(x, y, bits)
        assert split(x)[0] + split(x)[1] == x
        low, high = split(x)
        assert conj(x) == conj(low) + scale(high, -1)
        xx = mul(x, x, bits)
        assert mul(xx, x, bits) == mul(x, xx, bits)
        product_cases += 4
        if bits < 6:
            ex, ey = embed(x, bits + 1), embed(y, bits + 1)
            assert mul(ex, ey, bits + 1) == embed(mul(x, y, bits), bits + 1)
            assert mul_doubling(ex, ey, bits + 1) == mul(ex, ey, bits + 1)
            product_cases += 2

        if bits in (3, 4, 6):
            # Emit the exact product coefficients for the shared gate cases so
            # scripts/ci/cayley_dickson_tower_gate.sh can compare Sounio's
            # output against this oracle instead of trusting success markers.
            # Both independent evaluators must agree before anything is
            # printed; the vector is zero-padded to the 64-wide CDElement
            # layout the Sounio fixture uses.
            direct = mul(x, y, bits)
            doubled = mul_doubling(x, y, bits)
            assert doubled == direct
            padded = direct + [0] * (64 - len(direct))
            print(
                f"CD_TOWER_ORACLE_EXPECT k={bits} "
                f"coeffs={','.join(str(c) for c in padded)}"
            )

        count = 0
        for i in range(1, 1 << bits):
            for j in range(1, 1 << bits):
                for k in range(1, 1 << bits):
                    wave = basis_wave(i, j, k, bits)
                    assert wave in (-2, 0, 2), (bits, i, j, k, wave)
                    count += wave != 0
                    basis_triples += 1
        nonassociative[bits] = count

    assert nonassociative[1] == 0
    assert nonassociative[2] == 0
    assert nonassociative[3] == 168
    assert nonassociative[4] == 1848
    counts = ",".join(f"k{k}={v}" for k, v in nonassociative.items())
    print(
        f"CD_TOWER_ORACLE_OK levels=6 product_cases={product_cases} "
        f"basis_triples={basis_triples} nonassociative={counts}"
    )


if __name__ == "__main__":
    main()
