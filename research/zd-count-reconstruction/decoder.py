#!/usr/bin/env python3
"""Exact reconstruction within the principal Cayley-Dickson component family.

A word is read outside-in: TCK = T(C(K)); R resets at the remaining depth.
decode_native returns a canonical class, not the original XOR label, and
does not recognize membership of an arbitrary input graph from its counts.
"""
import argparse
import json


def natural(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def order(depth):
    return (1 << (natural(depth, "depth") + 2)) - 1


def counts(depth, word):
    """Forward (m,p) from the signed normal form, independently of decoding."""
    natural(depth, "depth")
    if not isinstance(word, str) or not word:
        raise ValueError("empty or non-string code")
    if depth == 0:
        if word == "K":
            return 3, 0
        if word == "Z":
            return 0, 0
        raise ValueError("depth zero requires K or Z")
    q = order(depth - 1)
    if word == "R":
        return q * (2 * q + 1), 0
    if word[0] not in "TC":
        raise ValueError("positive depth requires R, T or C")
    m, p = counts(depth - 1, word[1:])
    return (3 * q + 4 * m, 6 * m + 8 * p) if word[0] == "T" else (4 * m, 8 * p)


def decode_counts(depth, m, p):
    """Inverse on the exact image; reject negative or nonintegral pullbacks."""
    natural(depth, "depth")
    natural(m, "m")
    natural(p, "p")
    prefix = []
    while depth:
        q = order(depth - 1)
        if m == q * (2 * q + 1) and p == 0:
            return "".join(prefix) + "R"
        if m % 2 == 0:
            if m % 4 or p % 8:
                raise ValueError("invalid C pullback")
            m, p = m // 4, p // 8
            prefix.append("C")
        else:
            remainder = m - 3 * q
            if remainder < 0 or remainder % 4:
                raise ValueError("invalid T edge pullback")
            child_m = remainder // 4
            remainder_p = p - 6 * child_m
            if remainder_p < 0 or remainder_p % 8:
                raise ValueError("invalid T triangle pullback")
            m, p = child_m, remainder_p // 8
            prefix.append("T")
        depth -= 1
    if (m, p) == (3, 0):
        return "".join(prefix) + "K"
    if (m, p) == (0, 0):
        return "".join(prefix) + "Z"
    raise ValueError("invalid base counts")


def decode_native(depth, edges, triangles):
    natural(edges, "edges")
    natural(triangles, "triangles")
    if edges % 8 or triangles % 16:
        raise ValueError("native counts require E divisible by 8 and triangles by 16")
    word = decode_counts(depth, edges // 8, triangles // 16)
    if word[0] not in "KRT":
        raise ValueError("counts belong only to the auxiliary channel")
    return word


def label_codes(depth, label):
    """Existing two-channel label recurrence; independent of count decoding."""
    natural(depth, "depth")
    if type(label) is not int or not 0 < label < (1 << (depth + 3)):
        raise ValueError("label must satisfy 0 < W < 2**(depth+3)")
    if depth == 0:
        return "K", "Z"
    h = 1 << (depth + 2)
    if label == h:
        return "R", "C" * depth + "Z"
    x, y = label_codes(depth - 1, label if label < h else label - h)
    return ("T" + x, "C" + y) if label < h else ("T" + y, "C" + x)


def origin_bit(word):
    return word[-1] in "KR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth", type=int)
    parser.add_argument("edges", type=int)
    parser.add_argument("triangles", type=int)
    args = parser.parse_args()
    try:
        word = decode_native(args.depth, args.edges, args.triangles)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"depth": args.depth, "canonical_code": word,
                      "origin_bit": origin_bit(word),
                      "family_membership_assumed": True}, sort_keys=True))
