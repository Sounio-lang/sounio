"""Tests for the RDKit integration.

All tests are skipped automatically when rdkit is not installed.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit not installed")

from sounio.integrations.rdkit_ext import (
    mol_from_smiles,
    calculate_properties,
    batch_screen,
    fingerprint_similarity,
    lipinski_rule_of_five,
)
from sounio.knowledge import Knowledge
from sounio.types import Molecule


# ---------------------------------------------------------------------------
# mol_from_smiles
# ---------------------------------------------------------------------------


class TestMolFromSmiles:
    def test_valid_smiles_returns_mol(self):
        mol = mol_from_smiles("c1ccccc1")  # benzene
        assert mol is not None

    def test_aspirin_smiles(self):
        mol = mol_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None

    def test_invalid_smiles_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid or unparseable SMILES"):
            mol_from_smiles("not_a_smiles!!!")

    def test_empty_smiles_raises_value_error(self):
        with pytest.raises(ValueError):
            mol_from_smiles("")


# ---------------------------------------------------------------------------
# calculate_properties
# ---------------------------------------------------------------------------


class TestCalculateProperties:
    def test_aspirin_properties(self):
        mol = calculate_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(mol, Molecule)
        assert isinstance(mol.molecular_weight, Knowledge)
        assert isinstance(mol.logp, Knowledge)
        # Aspirin MW ~180.16
        assert 178.0 < mol.molecular_weight.value < 182.0

    def test_uncertainty_is_one_percent(self):
        mol = calculate_properties("CC(=O)Oc1ccccc1C(=O)O")
        ratio = mol.molecular_weight.epsilon / mol.molecular_weight.value
        assert abs(ratio - 0.01) < 1e-9

    def test_provenance_label(self):
        mol = calculate_properties("c1ccccc1")  # benzene
        assert mol.molecular_weight.provenance == "rdkit_computed"
        assert mol.logp.provenance == "rdkit_computed"

    def test_hbd_hba_are_integers(self):
        mol = calculate_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(mol.hbd, int)
        assert isinstance(mol.hba, int)

    def test_lipinski_evaluated(self):
        mol = calculate_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert mol.passes_lipinski is not None

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            calculate_properties("INVALID")


# ---------------------------------------------------------------------------
# batch_screen
# ---------------------------------------------------------------------------


class TestBatchScreen:
    # Aspirin (passes Ro5), atorvastatin-like (MW ~558 — fails), caffeine (passes)
    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
    CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
    HEAVY = "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c(C(=O)Nc2ccccc2)c1-c1ccc(F)cc1"

    def test_filters_ro5_failures(self):
        results = batch_screen([self.ASPIRIN, self.CAFFEINE, self.HEAVY])
        smiles_out = [m.smiles for m in results]
        assert self.ASPIRIN in smiles_out
        assert self.CAFFEINE in smiles_out
        # HEAVY (atorvastatin MW ~558) should be filtered
        assert self.HEAVY not in smiles_out

    def test_invalid_smiles_skipped(self):
        results = batch_screen([self.ASPIRIN, "INVALID!!!", self.CAFFEINE])
        assert len(results) == 2

    def test_empty_list(self):
        assert batch_screen([]) == []

    def test_all_fail_returns_empty(self):
        results = batch_screen([self.HEAVY])
        assert results == []


# ---------------------------------------------------------------------------
# fingerprint_similarity
# ---------------------------------------------------------------------------


class TestFingerprintSimilarity:
    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
    IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

    def test_self_similarity_is_one(self):
        sim = fingerprint_similarity(self.ASPIRIN, self.ASPIRIN)
        assert abs(sim - 1.0) < 1e-9

    def test_different_molecules(self):
        sim = fingerprint_similarity(self.ASPIRIN, self.IBUPROFEN)
        assert 0.0 <= sim < 1.0

    def test_symmetry(self):
        s1 = fingerprint_similarity(self.ASPIRIN, self.IBUPROFEN)
        s2 = fingerprint_similarity(self.IBUPROFEN, self.ASPIRIN)
        assert abs(s1 - s2) < 1e-9

    def test_returns_float(self):
        assert isinstance(fingerprint_similarity(self.ASPIRIN, self.IBUPROFEN), float)


# ---------------------------------------------------------------------------
# lipinski_rule_of_five (standalone on Molecule)
# ---------------------------------------------------------------------------


class TestLipinskiRuleOfFive:
    def _make_mol(self, mw, logp, hbd, hba):
        return Molecule(
            name="test",
            smiles="C",
            molecular_weight=Knowledge(mw, 0.0, "test"),
            logp=Knowledge(logp, 0.0, "test"),
            hbd=hbd,
            hba=hba,
        )

    def test_all_pass(self):
        mol = self._make_mol(300.0, 2.0, 2, 5)
        passes, violations = lipinski_rule_of_five(mol)
        assert passes is True
        assert violations == []

    def test_mw_violation(self):
        mol = self._make_mol(600.0, 2.0, 2, 5)
        passes, violations = lipinski_rule_of_five(mol)
        assert passes is False
        assert any("MW" in v for v in violations)

    def test_logp_violation(self):
        mol = self._make_mol(300.0, 6.0, 2, 5)
        passes, violations = lipinski_rule_of_five(mol)
        assert passes is False
        assert any("LogP" in v for v in violations)

    def test_hbd_violation(self):
        mol = self._make_mol(300.0, 2.0, 6, 5)
        passes, violations = lipinski_rule_of_five(mol)
        assert passes is False
        assert any("HBD" in v for v in violations)

    def test_hba_violation(self):
        mol = self._make_mol(300.0, 2.0, 2, 11)
        passes, violations = lipinski_rule_of_five(mol)
        assert passes is False
        assert any("HBA" in v for v in violations)

    def test_multiple_violations(self):
        mol = self._make_mol(600.0, 6.0, 6, 11)
        passes, violations = lipinski_rule_of_five(mol)
        assert passes is False
        assert len(violations) == 4

    def test_boundary_mw_exactly_500(self):
        mol = self._make_mol(500.0, 2.0, 2, 5)
        passes, _ = lipinski_rule_of_five(mol)
        assert passes is True

    def test_boundary_mw_500_plus_epsilon(self):
        mol = self._make_mol(500.001, 2.0, 2, 5)
        passes, _ = lipinski_rule_of_five(mol)
        assert passes is False
