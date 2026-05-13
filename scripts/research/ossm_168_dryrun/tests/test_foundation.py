"""Foundation tests for the ossm_168 dry-run harness.

Asserts the algebraic invariants that anchor section 7.1 of the frozen
pre-registration (tag prereg/ossm-168-depression-v1):

- 7 Fano lines exist on {1..7} under the XOR characterization
- 42 ordered Fano triples, 168 ordered non-Fano triples (T1 / 168 Theorem)
- |Aut(Fano)| = 168
- |S_7 / Aut(Fano)| = 30, matching the count claimed in the protocol

These tests do NOT exercise any clinical hypothesis; they guard the
algebraic plumbing.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from scripts.research.ossm_168_dryrun import fano_orbits, octonion


def test_fano_lines_are_seven():
    assert len(fano_orbits.FANO_LINES) == 7


def test_fano_lines_xor_zero():
    for i, j, k in fano_orbits.FANO_LINES:
        assert (i ^ j ^ k) == 0
        assert 1 <= i < j < k <= 7


def test_fano_lines_unique():
    # Each unordered {i,j,k} appears at most once.
    as_set = {frozenset(line) for line in fano_orbits.FANO_LINES}
    assert len(as_set) == 7


def test_fano_ordered_counts_168_theorem():
    total_distinct = sum(
        1 for _ in permutations(range(1, 8), 3)
    )
    assert total_distinct == 210  # 7 * 6 * 5

    non_fano = sum(1 for _ in fano_orbits.non_fano_ordered_triples())
    assert non_fano == 168  # T1 of the 168 Theorem

    fano_ordered = total_distinct - non_fano
    assert fano_ordered == 42  # 7 lines * 3!


def test_aut_fano_order_168():
    aut = fano_orbits.fano_automorphism_group()
    assert len(aut) == 168
    # Identity must be present.
    assert (1, 2, 3, 4, 5, 6, 7) in set(aut)


def test_coset_reps_count_30():
    reps = fano_orbits.coset_representatives_S7_mod_Aut()
    assert len(reps) == 30
    # All reps are length-7 permutations of {1..7}.
    for r in reps:
        assert sorted(r) == [1, 2, 3, 4, 5, 6, 7]
    # All distinct.
    assert len(set(reps)) == 30


def test_coset_reps_are_in_different_cosets():
    aut = set(fano_orbits.fano_automorphism_group())
    reps = fano_orbits.coset_representatives_S7_mod_Aut()
    # For any two distinct reps r1, r2, their "difference" r2 o r1^{-1}
    # must not be in Aut(Fano), i.e. they are in different left cosets.
    def compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(a[b[i] - 1] for i in range(7))

    def inverse(a: tuple[int, ...]) -> tuple[int, ...]:
        inv = [0] * 7
        for i in range(7):
            inv[a[i] - 1] = i + 1
        return tuple(inv)

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            diff = compose(inverse(reps[i]), reps[j])
            assert diff not in aut, (
                f"representatives {reps[i]} and {reps[j]} are in the "
                f"same left coset"
            )


def test_oct_mul_identity():
    e0 = octonion.oct_basis(0)
    for i in range(8):
        ei = octonion.oct_basis(i)
        np.testing.assert_allclose(octonion.oct_mul(e0, ei), ei, atol=1e-12)
        np.testing.assert_allclose(octonion.oct_mul(ei, e0), ei, atol=1e-12)


def test_oct_mul_imaginary_squares():
    for i in range(1, 8):
        ei = octonion.oct_basis(i)
        expected = -octonion.oct_basis(0)
        np.testing.assert_allclose(octonion.oct_mul(ei, ei), expected, atol=1e-12)


def test_oct_associator_vanishes_on_fano_lines():
    # By T2 (Lemma 2): basis associator norms are exactly {0, 2}.
    # On Fano lines (the 7 lines), the three imaginary basis elements
    # associate (associator vanishes); on non-Fano distinct triples,
    # norm is 2 (up to sign conventions of Cayley-Dickson).
    for line in fano_orbits.FANO_LINES:
        a, b, c = (octonion.oct_basis(i) for i in line)
        assoc = octonion.oct_associator(a, b, c)
        np.testing.assert_allclose(
            assoc, np.zeros(8), atol=1e-12,
            err_msg=f"associator [e{line[0]},e{line[1]},e{line[2]}] != 0",
        )


def test_oct_associator_norm_on_non_fano_triples_is_two():
    # For any non-Fano ordered (i,j,k), ||[e_i, e_j, e_k]|| = 2.
    non_fano_sample = list(fano_orbits.non_fano_ordered_triples())
    assert len(non_fano_sample) == 168
    for i, j, k in non_fano_sample:
        a, b, c = octonion.oct_basis(i), octonion.oct_basis(j), octonion.oct_basis(k)
        assoc = octonion.oct_associator(a, b, c)
        np.testing.assert_allclose(
            octonion.oct_norm(assoc), 2.0, atol=1e-12,
            err_msg=f"||[e{i},e{j},e{k}]|| != 2",
        )


def test_permute_input_round_trip():
    sigma = (3, 1, 4, 1, 5, 9, 2)  # not a permutation; sanity check below
    # Actually construct a valid S_7 permutation:
    sigma = (7, 6, 5, 4, 3, 2, 1)
    x = np.arange(7, dtype=np.float64)
    y = octonion.permute_input(x, sigma)
    # sigma[i] gives which original index feeds into output position i.
    expected = np.array([x[sigma[i] - 1] for i in range(7)])
    np.testing.assert_allclose(y, expected)
