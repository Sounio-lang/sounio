"""Fano plane combinatorics and S_7 / Aut(Fano) coset representatives.

Canonical convention (matches stdlib/algebra/octonion.sio lines 62-74):
the seven imaginary octonion basis elements are indexed {1..7}, and the
seven Fano lines are exactly the 3-element subsets {i,j,k} with
    i XOR j XOR k == 0.
There are seven such lines; they form the Fano plane PG(2, F_2).

Aut(Fano) = Aut(PG(2, F_2)) = PGL(3, F_2) = PSL(2,7), of order 168.
|S_7 / Aut(Fano)| = 5040 / 168 = 30.

This module enumerates both and exposes a deterministic list of 30
lex-smallest coset representatives, with fixed canonical ordering.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterator

__all__ = [
    "FANO_LINES",
    "is_fano_triple",
    "non_fano_ordered_triples",
    "fano_automorphism_group",
    "coset_representatives_S7_mod_Aut",
]


def _fano_lines_xor() -> tuple[tuple[int, int, int], ...]:
    """Return the seven Fano lines as sorted 3-tuples on {1..7}."""
    lines: list[tuple[int, int, int]] = []
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            if k <= j or k > 7:
                continue
            lines.append((i, j, k))
    # Exactly 7 lines.
    assert len(lines) == 7, f"expected 7 Fano lines, got {len(lines)}"
    return tuple(sorted(lines))


FANO_LINES: tuple[tuple[int, int, int], ...] = _fano_lines_xor()
"""The seven Fano lines on {1..7} under the XOR characterization."""


def is_fano_triple(i: int, j: int, k: int) -> bool:
    """Return True iff {i,j,k} is a Fano line (requires i,j,k distinct, all in 1..7)."""
    if not (1 <= i <= 7 and 1 <= j <= 7 and 1 <= k <= 7):
        return False
    if i == j or j == k or i == k:
        return False
    return (i ^ j ^ k) == 0


def non_fano_ordered_triples() -> Iterator[tuple[int, int, int]]:
    """Yield each of the 168 ordered triples (i,j,k) with i,j,k distinct in {1..7}
    and {i,j,k} NOT a Fano line.

    Count check: 7*6*5 = 210 ordered triples with distinct entries;
    of these, 7 Fano lines * 3! = 42 are associative;
    the remaining 210 - 42 = 168 are non-associative.
    """
    for i, j, k in permutations(range(1, 8), 3):
        if (i ^ j ^ k) != 0:
            yield (i, j, k)


def _permute_line(line: tuple[int, int, int], sigma: tuple[int, ...]) -> tuple[int, int, int]:
    """Apply sigma (tuple of length 7, 1-indexed permutation) to a Fano line.

    sigma[i-1] is the image of i under sigma, for i in {1..7}.
    Returns the sorted image line.
    """
    a, b, c = line
    return tuple(sorted((sigma[a - 1], sigma[b - 1], sigma[c - 1])))  # type: ignore[return-value]


def _is_fano_automorphism(sigma: tuple[int, ...]) -> bool:
    """True iff sigma maps the set of Fano lines to itself."""
    lines_set = set(FANO_LINES)
    for line in FANO_LINES:
        if _permute_line(line, sigma) not in lines_set:
            return False
    return True


def fano_automorphism_group() -> tuple[tuple[int, ...], ...]:
    """Enumerate Aut(Fano) as a subgroup of S_7.

    Returns a tuple of |Aut(Fano)| = 168 permutations, each a length-7
    tuple in 1-indexed S_7 convention. Ordering is lexicographic.
    """
    auts = [sigma for sigma in permutations(range(1, 8)) if _is_fano_automorphism(sigma)]
    assert len(auts) == 168, f"expected |Aut(Fano)| = 168, got {len(auts)}"
    return tuple(auts)


def coset_representatives_S7_mod_Aut() -> tuple[tuple[int, ...], ...]:
    """Return one representative per left coset of Aut(Fano) in S_7.

    Left cosets: sigma * H = {sigma * h : h in H}, where H = Aut(Fano).
    Two permutations sigma, tau are in the same left coset iff
    sigma^{-1} * tau in H, i.e. iff they induce the same labelling
    of the Fano plane up to a Fano-preserving relabelling.

    The representative chosen from each coset is the lex-smallest element.
    The returned list is sorted lex among representatives, giving a
    deterministic canonical ordering independent of Python version or
    hash seeding.

    The count is exactly |S_7| / |Aut(Fano)| = 5040 / 168 = 30.
    """
    aut = set(fano_automorphism_group())
    seen: set[tuple[int, ...]] = set()
    reps: list[tuple[int, ...]] = []
    for sigma in permutations(range(1, 8)):
        if sigma in seen:
            continue
        # The left coset sigma * H contains {sigma o h : h in H},
        # where (sigma o h)(i) = sigma(h(i)).
        coset = {
            tuple(sigma[h[i] - 1] for i in range(7))  # type: ignore[misc]
            for h in aut
        }
        seen |= coset
        reps.append(min(coset))
    reps.sort()
    assert len(reps) == 30, f"expected 30 coset representatives, got {len(reps)}"
    return tuple(reps)
