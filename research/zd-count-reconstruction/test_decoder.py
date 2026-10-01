#!/usr/bin/env python3
"""Small exact controls, including direct native products, without Lean caches."""
from functools import lru_cache
from itertools import combinations
import json
from decoder import counts, decode_counts, decode_native, label_codes, origin_bit


@lru_cache(None)
def basis_product(n, a, b):
    """(sign,index) for e_a e_b using (ac-dbar*b, da+b*cbar)."""
    if n == 0:
        return 1, 0
    h = 1 << (n - 1)
    ah, al = divmod(a, h)
    bh, bl = divmod(b, h)
    if ah == 0 and bh == 0:
        sign, index = basis_product(n - 1, al, bl)
    elif ah == 0:
        sign, index = basis_product(n - 1, bl, al)
    elif bh == 0:
        sign, index = basis_product(n - 1, al, bl)
        sign *= 1 if bl == 0 else -1
    else:
        sign, index = basis_product(n - 1, bl, al)
        sign *= -1 if bl == 0 else 1
    return sign, index + (h if ah != bh else 0)


def product(n, x, y):
    out = {}
    for a, s in x:
        for b, t in y:
            sign, index = basis_product(n, a, b)
            out[index] = out.get(index, 0) + s * t * sign
    return {i: v for i, v in out.items() if v}


def native_counts(depth, label):
    n = depth + 3
    h = 1 << n
    vertices = [((a, 1), (h + (a ^ label), s))
                for a in range(1, h) if a != label for s in (-1, 1)]
    neighbours = [0] * len(vertices)
    edges = []
    for i, j in combinations(range(len(vertices)), 2):
        if not product(n + 1, vertices[i], vertices[j]) and not product(n + 1, vertices[j], vertices[i]):
            neighbours[i] |= 1 << j
            neighbours[j] |= 1 << i
            edges.append((i, j))
    triangle_mass = sum((neighbours[i] & neighbours[j]).bit_count() for i, j in edges)
    assert triangle_mass % 3 == 0
    assert all(neighbours), "principal vertices must have zero-divisor witnesses"
    return len(edges), triangle_mass // 3


def catalogue(depth):
    if depth == 0:
        return ["K", "Z"]
    previous = catalogue(depth - 1)
    return ["R"] + ["T" + w for w in previous] + ["C" + w for w in previous]


def rejects(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError(("accepted invalid input", args))


def main():
    # Complete tiny codomain boxes verify rejection, not only round trips.
    checked = 0
    for d in range(7):
        words = catalogue(d)
        image = {counts(d, word): word for word in words}
        assert len(image) == len(words)
        for pair, word in image.items():
            assert decode_counts(d, *pair) == word
            checked += 1
        if d <= 1:
            for m in range(23):
                for p in range(20):
                    if (m, p) in image:
                        assert decode_counts(d, m, p) == image[m, p]
                    else:
                        rejects(decode_counts, d, m, p)
        # Native image is precisely the X-channel, not all auxiliary codes.
        native_words = {label_codes(d, w)[0] for w in range(1, 1 << (d + 3))}
        assert native_words == {w for w in words if w[0] in "KRT"}
        for word in native_words:
            m, p = counts(d, word)
            assert decode_native(d, 8 * m, 16 * p) == word
    for d, word in [(64, "T" * 64 + "K"), (64, "TC" * 32 + "Z"), (64, "C" * 61 + "R")]:
        assert decode_counts(d, *counts(d, word)) == word
    for args in [(-1, 0, 0), (1, -1, 0), (1, 0, -1), (True, 0, 0), (1, 2.0, 0)]:
        rejects(decode_counts, *args)
    for args in [(1, 1, 0), (1, 0, 1), (0, 0, 0), (1, 96, 0)]:
        rejects(decode_native, *args)
    # Anchors independent of the recursive graph catalogue.
    assert basis_product(2, 1, 2) == (1, 3)
    assert basis_product(2, 2, 1) == (-1, 3)
    assert basis_product(3, 1, 7) == (1, 6)
    cases = [(0, w) for w in range(1, 8)]
    cases += [(1, w) for w in range(1, 16)]
    cases += [(2, w) for w in range(1, 32)]
    cases += [(3, w) for w in (1, 24, 31, 32, 33, 48, 63)]
    cases += [(5, 208), (5, 201)]
    observed = {}
    for d, w in cases:
        e, t = native_counts(d, w)
        word = label_codes(d, w)[0]
        m, p = counts(d, word)
        assert (e, t) == (8 * m, 16 * p), (d, w, e, t, m, p)
        assert decode_native(d, e, t) == word
        observed[d, w] = (e, t)
    assert observed[1, 1] == (168, 288)
    assert observed[1, 8] == (168, 0)
    assert observed[1, 9] == (72, 0)
    e1, t1 = observed[5, 208]
    e2, t2 = observed[5, 201]
    assert e1 != e2 and 3 * e1 + t1 == 3 * e2 + t2 == 1006776
    assert origin_bit(label_codes(5, 208)[0])
    assert not origin_bit(label_codes(5, 201)[0])
    print(json.dumps({"status": "PASS", "abstract_roundtrips": checked,
                      "native_direct_product_cases": len(cases),
                      "direct_cases": [{"depth": d, "W": w, "E": observed[d, w][0],
                                        "triangles": observed[d, w][1]} for d, w in cases],
                      "sharp_scalar": 1006776}, indent=2))


if __name__ == "__main__":
    main()
