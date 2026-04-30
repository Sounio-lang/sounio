"""Sedenion structure tests.

Verifies:
- Cayley-Dickson doubling identity: (a, b) represents a + b * e_8
- Canonical primitive zero-divisor pair (e_3 + e_10)(e_6 - e_15) = 0
- CD flexibility: [a, b, a] = 0, hence ||[c,b,a]|| = ||[a,b,c]|| always
- Non-alternativity: exists a, b with [a, a, b] != 0 in S_16
- O_168_sedenion_triples() returns exactly 168 triples with the expected form
- v3 F_3 (L1 associator mass) is non-zero on generic states
"""

from __future__ import annotations

import numpy as np

from scripts.research.ossm_168_dryrun import octonion, sedenion


def test_sedenion_basis_indexing():
    for i in range(16):
        ei = sedenion.sed_basis(i)
        assert ei.shape == (16,)
        assert ei[i] == 1.0
        assert np.sum(np.abs(ei)) == 1.0


def test_cayley_dickson_doubling_reproduces_octonion_on_lower_half():
    # When both operands have zero "imaginary doubled" half, the sedenion
    # product reduces to the octonion product on the lower 8 components.
    a = np.array([0.5, 1.0, -0.3, 0.7, 0.1, -0.2, 0.4, 0.9], dtype=np.float64)
    b = np.array([1.0, 0.2, 0.1, -0.4, 0.3, 0.8, -0.5, 0.6], dtype=np.float64)
    sa = sedenion.sed_from_pair(a, np.zeros(8))
    sb = sedenion.sed_from_pair(b, np.zeros(8))
    sab = sedenion.sed_mul(sa, sb)
    oct_ab = octonion.oct_mul(a, b)
    np.testing.assert_allclose(sab[0:8], oct_ab, atol=1e-12)
    np.testing.assert_allclose(sab[8:16], np.zeros(8), atol=1e-12)


def test_canonical_zero_divisor_pair_multiplies_to_zero():
    z, w = sedenion.CANONICAL_ZERO_DIVISOR_PAIR
    # Primitive pair (e_3 + e_10)(e_6 - e_15) should equal zero in S_16
    # per de Marrais 2000.
    product = sedenion.sed_mul(z, w)
    np.testing.assert_allclose(product, np.zeros(16), atol=1e-12)


def test_sedenion_is_flexible_as_expected():
    # All Cayley-Dickson algebras are flexible: [a, b, a] = 0 for all a, b.
    # This is the property that, per DEVIATIONS entry 0004, forced H3 to
    # be reformulated in v3 (flexibility implies ||[c,b,a]|| = ||[a,b,c]||
    # throughout the CD tower).
    rng = np.random.default_rng(20260421)
    for _ in range(5):
        a = rng.standard_normal(16)
        b = rng.standard_normal(16)
        flex = sedenion.sed_associator(a, b, a)
        np.testing.assert_allclose(flex, np.zeros(16), atol=1e-10)


def test_sedenion_is_not_alternative_but_associator_is_not_zero():
    # Sedenions are not alternative: exists a, b with [a, a, b] != 0.
    # This is the known Cayley-Dickson tower fact; a simple witness
    # is a = e_1 + e_10, b = e_15 (or any generic mix of halves).
    a = sedenion.sed_basis(1) + sedenion.sed_basis(10)
    b = sedenion.sed_basis(15)
    assoc = sedenion.sed_associator(a, a, b)
    assert sedenion.sed_norm(assoc) > 1e-8, (
        "[a, a, b] = 0 for (e_1 + e_10, e_15) -- unexpected alternativity"
    )


def test_O_168_sedenion_triples_count_and_structure():
    triples = sedenion.O_168_sedenion_triples()
    assert len(triples) == 168
    # All distinct.
    assert len(set(triples)) == 168
    # Each (I, J, K) has I, K in {1..7}, J in {9..15}.
    for I, J, K in triples:
        assert 1 <= I <= 7, f"I out of range: {I}"
        assert 9 <= J <= 15, f"J out of range: {J}"
        assert 1 <= K <= 7, f"K out of range: {K}"
        # I != K (non-Fano triples have i != k always, since distinct).
        assert I != K, f"I == K in triple ({I},{J},{K})"


def test_O_168_sedenion_triples_have_nontrivial_associator():
    # For each of the 168 triples, ||[e_I, e_J, e_K]|| > 0 in S_16.
    # (They are constructed to be non-associative.)
    triples = sedenion.O_168_sedenion_triples()
    zero_count = 0
    for I, J, K in triples:
        a = sedenion.sed_basis(I)
        b = sedenion.sed_basis(J)
        c = sedenion.sed_basis(K)
        assoc = sedenion.sed_associator(a, b, c)
        if sedenion.sed_norm(assoc) < 1e-10:
            zero_count += 1
    # Allow a few degenerate triples (where the structure happens to
    # associate), but require the bulk to be non-associative.
    assert zero_count < len(triples) // 4, (
        f"too many associating triples: {zero_count} of {len(triples)}"
    )


def test_O_168_sedenion_f3_l1_mass_is_nonzero_on_generic_state():
    # v3 F_3 = (1/T*168) * sum_{(I,J,K)} ||[h*e_I, h*e_J, h*e_K]||_{L^1}.
    # On a generic sedenion state h, this must be strictly positive.
    # If it were zero, v3 would have the same fate as v1/v2.
    rng = np.random.default_rng(20260421)
    h = rng.standard_normal(16)
    triples = sedenion.O_168_sedenion_triples()
    total_l1 = 0.0
    for I, J, K in triples:
        a = sedenion.sed_mul(h, sedenion.sed_basis(I))
        b = sedenion.sed_mul(h, sedenion.sed_basis(J))
        c = sedenion.sed_mul(h, sedenion.sed_basis(K))
        assoc = sedenion.sed_associator(a, b, c)
        total_l1 += float(np.sum(np.abs(assoc)))
    assert total_l1 > 1e-6, (
        f"v3 F_3 L1 mass collapsed to zero on generic state: total={total_l1}"
    )
