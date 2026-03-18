"""Tests for the PubChem integration.

Offline cache tests require no network.  Online tests mock urllib.request.urlopen
so they also run in air-gapped environments.
"""

from __future__ import annotations

import json
import unittest.mock as mock
from io import BytesIO
from typing import Any

import pytest

from sounio.integrations.pubchem import (
    fetch_by_name,
    fetch_by_cid,
    search_molecules,
    _OFFLINE_CACHE,
    _CID_TO_NAME,
    _MW_EPSILON,
    _LOGP_EPSILON,
)
from sounio.knowledge import Knowledge
from sounio.types import Molecule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pubchem_response(
    mw: float = 180.159,
    logp: float = 1.2,
    hbd: int = 1,
    hba: int = 4,
    smiles: str = "CC(=O)Oc1ccccc1C(=O)O",
    iupac: str = "2-(acetyloxy)benzoic acid",
) -> bytes:
    """Build a minimal PubChem PUG REST JSON response."""
    payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 2244,
                    "MolecularWeight": str(mw),
                    "XLogP": logp,
                    "HBondDonorCount": hbd,
                    "HBondAcceptorCount": hba,
                    "IsomericSMILES": smiles,
                    "IUPACName": iupac,
                }
            ]
        }
    }
    return json.dumps(payload).encode()


def _make_cid_response(cids: list) -> bytes:
    return json.dumps({"IdentifierList": {"CID": cids}}).encode()


class _MockHTTPResponse:
    """Minimal urllib response-like object."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Offline cache tests
# ---------------------------------------------------------------------------


class TestOfflineCache:
    def test_all_expected_drugs_present(self):
        expected = {
            "aspirin", "ibuprofen", "metformin", "paracetamol", "caffeine",
            "penicillin", "warfarin", "omeprazole", "atorvastatin", "insulin",
        }
        cache_names = {entry["name"] for entry in _OFFLINE_CACHE.values()}
        assert expected.issubset(cache_names), (
            f"Missing drugs: {expected - cache_names}"
        )

    def test_acetaminophen_alias(self):
        assert "acetaminophen" in _OFFLINE_CACHE
        assert _OFFLINE_CACHE["acetaminophen"]["name"] == "paracetamol"

    def test_all_entries_have_required_fields(self):
        required = {"name", "smiles", "mw", "logp", "hbd", "hba"}
        for key, entry in _OFFLINE_CACHE.items():
            missing = required - set(entry.keys())
            assert not missing, f"Entry {key!r} missing fields: {missing}"

    def test_cid_map_populated(self):
        # At least aspirin CID should be present
        assert 2244 in _CID_TO_NAME


class TestFetchByNameOffline:
    def test_aspirin_offline(self):
        mol = fetch_by_name("aspirin", offline=True)
        assert isinstance(mol, Molecule)
        assert mol.name == "aspirin"

    def test_aspirin_mw_value(self):
        mol = fetch_by_name("aspirin", offline=True)
        assert abs(mol.molecular_weight.value - 180.159) < 0.01

    def test_mw_epsilon(self):
        mol = fetch_by_name("aspirin", offline=True)
        assert mol.molecular_weight.epsilon == _MW_EPSILON

    def test_logp_epsilon(self):
        mol = fetch_by_name("aspirin", offline=True)
        assert mol.logp.epsilon == _LOGP_EPSILON

    def test_knowledge_provenance(self):
        mol = fetch_by_name("caffeine", offline=True)
        assert mol.molecular_weight.provenance == "pubchem"
        assert mol.logp.provenance == "pubchem_xlogp3"

    def test_ibuprofen_passes_lipinski(self):
        mol = fetch_by_name("ibuprofen", offline=True)
        assert mol.passes_lipinski is True

    def test_atorvastatin_fails_lipinski(self):
        mol = fetch_by_name("atorvastatin", offline=True)
        # MW ~558 > 500 → should fail
        assert mol.passes_lipinski is False

    def test_metformin_offline(self):
        mol = fetch_by_name("metformin", offline=True)
        assert mol.name == "metformin"

    def test_acetaminophen_alias_resolves(self):
        mol = fetch_by_name("acetaminophen", offline=True)
        assert mol.name == "paracetamol"

    def test_case_insensitive_lookup(self):
        mol = fetch_by_name("Aspirin", offline=True)
        assert mol.name == "aspirin"

    def test_cache_miss_offline_raises_key_error(self):
        with pytest.raises(KeyError):
            fetch_by_name("xyzabc_not_a_drug", offline=True)

    def test_returns_molecule_type(self):
        mol = fetch_by_name("warfarin", offline=True)
        assert isinstance(mol, Molecule)

    def test_hbd_hba_are_integers(self):
        mol = fetch_by_name("aspirin", offline=True)
        assert isinstance(mol.hbd, int)
        assert isinstance(mol.hba, int)

    @pytest.mark.parametrize("drug", [
        "aspirin", "ibuprofen", "metformin", "paracetamol",
        "caffeine", "penicillin", "warfarin", "omeprazole", "atorvastatin",
    ])
    def test_all_drugs_fetchable(self, drug):
        mol = fetch_by_name(drug, offline=True)
        assert isinstance(mol, Molecule)
        assert mol.molecular_weight.value > 0


class TestFetchByCIDOffline:
    def test_aspirin_by_cid(self):
        mol = fetch_by_cid(2244, offline=True)
        assert mol.name == "aspirin"

    def test_ibuprofen_by_cid(self):
        mol = fetch_by_cid(3672, offline=True)
        assert mol.name == "ibuprofen"

    def test_unknown_cid_offline_raises_key_error(self):
        with pytest.raises(KeyError):
            fetch_by_cid(999999999, offline=True)

    def test_caffeine_by_cid(self):
        mol = fetch_by_cid(2519, offline=True)
        assert mol.name == "caffeine"


class TestSearchMoleculesOffline:
    def test_search_aspirin(self):
        results = search_molecules("aspirin")
        # Should hit cache before network
        names = [m.name for m in results]
        assert "aspirin" in names

    def test_max_results_respected(self):
        results = search_molecules("a", max_results=2)
        assert len(results) <= 2

    def test_empty_query_returns_list(self):
        results = search_molecules("zzzzz_no_match_expected", max_results=3)
        # May be empty or have network results; type must be list
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Mocked network tests
# ---------------------------------------------------------------------------


class TestFetchByNameOnline:
    def test_network_fetch_aspirin(self):
        mock_resp = _MockHTTPResponse(_make_pubchem_response())
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            mol = fetch_by_name("unknown_compound_xyz")
        assert isinstance(mol, Molecule)
        assert abs(mol.molecular_weight.value - 180.159) < 0.01

    def test_network_fetch_logp(self):
        mock_resp = _MockHTTPResponse(_make_pubchem_response(logp=3.5))
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            mol = fetch_by_name("test_compound")
        assert abs(mol.logp.value - 3.5) < 1e-9

    def test_network_fetch_hbd_hba(self):
        mock_resp = _MockHTTPResponse(_make_pubchem_response(hbd=2, hba=6))
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            mol = fetch_by_name("test_compound")
        assert mol.hbd == 2
        assert mol.hba == 6

    def test_network_fetch_smiles(self):
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        mock_resp = _MockHTTPResponse(_make_pubchem_response(smiles=smiles))
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            mol = fetch_by_name("test_compound")
        assert mol.smiles == smiles

    def test_cache_takes_priority_over_network(self):
        # aspirin is in cache — urlopen should NOT be called
        with mock.patch("urllib.request.urlopen") as mock_open:
            mol = fetch_by_name("aspirin")
        mock_open.assert_not_called()
        assert mol.name == "aspirin"

    def test_http_error_raises_value_error(self):
        import urllib.error
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None),
        ):
            with pytest.raises(ValueError, match="HTTP error"):
                fetch_by_name("nonexistent_compound_xyz")

    def test_connection_error_raises_connection_error(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=OSError("Network unreachable"),
        ):
            with pytest.raises(ConnectionError):
                fetch_by_name("some_compound_xyz")


class TestFetchByCIDOnline:
    def test_network_fetch_by_cid(self):
        mock_resp = _MockHTTPResponse(_make_pubchem_response())
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            mol = fetch_by_cid(99999)
        assert isinstance(mol, Molecule)

    def test_cache_takes_priority_for_known_cid(self):
        with mock.patch("urllib.request.urlopen") as mock_open:
            mol = fetch_by_cid(2244)  # aspirin
        mock_open.assert_not_called()
        assert mol.name == "aspirin"


class TestSearchMoleculesOnline:
    def test_search_falls_back_to_network(self):
        cid_resp = _MockHTTPResponse(_make_cid_response([2244, 3672]))
        prop_resp = _MockHTTPResponse(_make_pubchem_response())

        call_count = [0]

        def _urlopen_side_effect(url, timeout=10):
            call_count[0] += 1
            if "cids" in url:
                return cid_resp
            return prop_resp

        with mock.patch("urllib.request.urlopen", side_effect=_urlopen_side_effect):
            # Use a query that won't match cache (aspirin IS in cache, so use full name)
            results = search_molecules("2-(acetyloxy)benzoic acid", max_results=3)
        assert isinstance(results, list)

    def test_search_returns_cache_hits_first(self):
        with mock.patch("urllib.request.urlopen") as mock_open:
            results = search_molecules("aspirin", max_results=1)
        # aspirin in cache → no network call
        mock_open.assert_not_called()
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Knowledge field contracts
# ---------------------------------------------------------------------------


class TestKnowledgeContracts:
    """Verify that all numeric fields obey the Knowledge epistemic contract."""

    def test_mw_is_knowledge(self):
        mol = fetch_by_name("caffeine", offline=True)
        assert isinstance(mol.molecular_weight, Knowledge)

    def test_logp_is_knowledge(self):
        mol = fetch_by_name("caffeine", offline=True)
        assert isinstance(mol.logp, Knowledge)

    def test_mw_epsilon_constant(self):
        for drug in ("aspirin", "caffeine", "warfarin"):
            mol = fetch_by_name(drug, offline=True)
            assert mol.molecular_weight.epsilon == pytest.approx(_MW_EPSILON)

    def test_logp_epsilon_constant(self):
        for drug in ("aspirin", "caffeine", "warfarin"):
            mol = fetch_by_name(drug, offline=True)
            assert mol.logp.epsilon == pytest.approx(_LOGP_EPSILON)

    def test_confidence_accessible(self):
        mol = fetch_by_name("ibuprofen", offline=True)
        c = mol.molecular_weight.confidence
        assert 0.0 <= c <= 1.0
