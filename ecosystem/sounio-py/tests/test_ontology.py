"""Tests for hybrid local + federated ontology resolution."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path


from sounio.ontology import OntologyResolver

# We build fake ontology bundles locally instead of relying on external data files.
def create_mock_bundles(home: Path):
    home.mkdir(parents=True, exist_ok=True)
    
    # helper
    def write_dontology(name: str, ontology: str, terms: list):
        payload = json.dumps({"ontology": ontology, "terms": terms}).encode("utf-8")
        header = b"DONT" + struct.pack("<II", 1, len(payload))
        (home / f"{name}.dontology").write_bytes(header + payload)

    write_dontology("snomed", "SNOMED", [
        {
            "curie": "SNOMED:44054006",
            "label": "Type 2 diabetes mellitus",
            "definition": "",
            "parents": ["SNOMED:73211009", "SNOMED:138875005"],
            "synonyms": [],
            "iri": ""
        },
        {
            "curie": "SNOMED:73211009",
            "label": "Diabetes mellitus",
            "definition": "",
            "parents": [],
            "synonyms": [],
            "iri": ""
        },
        {
            "curie": "SNOMED:138875005",
            "label": "SNOMED CT Concept",
            "definition": "",
            "parents": [],
            "synonyms": [],
            "iri": ""
        }
    ])

    write_dontology("loinc", "LOINC", [
        {
            "curie": "LOINC:4548-4",
            "label": "Hemoglobin A1c/Hemoglobin.total in Blood",
            "definition": "",
            "parents": [],
            "synonyms": [],
            "iri": ""
        },
        {
            "curie": "LOINC:1558-6",
            "label": "Fasting glucose",
            "definition": "",
            "parents": [],
            "synonyms": ["glucose"],
            "iri": ""
        }
    ])

    write_dontology("hpo", "HPO", [
        {
            "curie": "HPO:0005978",
            "label": "Type II diabetes mellitus",
            "definition": "",
            "parents": [],
            "synonyms": ["glucose intolerance"],
            "iri": ""
        },
        {
            "curie": "HPO:0003074",
            "label": "Hyperglycemia",
            "definition": "",
            "parents": [],
            "synonyms": [],
            "iri": ""
        }
    ])

    write_dontology("chebi", "CHEBI", [
        {
            "curie": "CHEBI:17234",
            "label": "glucose",
            "definition": "",
            "parents": [],
            "synonyms": [],
            "iri": ""
        }
    ])

    mappings_dir = home / "mappings"
    mappings_dir.mkdir(exist_ok=True)
    mappings_file = mappings_dir / "test.sssom.json"
    mappings_file.write_text(json.dumps({
        "mappings": [
            {
                "subject_id": "HPO:0003074",
                "predicate": "skos:exactMatch",
                "object_id": "GO:0042593",
                "confidence": 0.91,
                "justification": "semapv:LexicalMatching"
            },
            {
                "subject_id": "HPO:0003074",
                "predicate": "skos:broadMatch",
                "object_id": "CHEBI:6801",
                "confidence": 0.85,
                "justification": ""
            }
        ]
    }))


def make_resolver(tmp_path: Path, mode: str = "local") -> OntologyResolver:
    bundle_home = tmp_path / "bundles"
    create_mock_bundles(bundle_home)
    cache_dir = tmp_path / "cache"
    return OntologyResolver(ontology_home=bundle_home, cache_dir=cache_dir, mode=mode)


def test_resolve_local_term(tmp_path):
    resolver = make_resolver(tmp_path)
    term = resolver.resolve("SNOMED:44054006")

    assert term is not None
    assert term.curie == "SNOMED:44054006"
    assert term.label == "Type 2 diabetes mellitus"
    assert term.source_layer == "local"
    assert term.provenance == "local:snomed"


def test_resolve_writes_positive_cache(tmp_path):
    resolver = make_resolver(tmp_path)
    term = resolver.resolve("LOINC:4548-4")

    assert term is not None
    cache_path = tmp_path / "cache" / "LOINC_4548-4.json"
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text())
    assert payload["missing"] is False
    assert payload["term"]["curie"] == "LOINC:4548-4"


def test_resolve_writes_negative_cache(tmp_path):
    resolver = make_resolver(tmp_path)
    assert resolver.resolve("SNOMED:99999999") is None

    cache_path = tmp_path / "cache" / "SNOMED_99999999.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload["missing"] is True


def test_rejects_invalid_curie(tmp_path):
    resolver = make_resolver(tmp_path)
    assert resolver.resolve("definitely-not-a-curie") is None


def test_search_local_terms(tmp_path):
    resolver = make_resolver(tmp_path)
    hits = resolver.search("glucose")

    curies = {hit.curie for hit in hits}
    assert "HPO:0005978" in curies
    assert "LOINC:1558-6" in curies
    assert "CHEBI:17234" in curies


def test_ancestors_and_subclass(tmp_path):
    resolver = make_resolver(tmp_path)

    ancestors = resolver.ancestors("SNOMED:44054006")
    assert "SNOMED:73211009" in ancestors
    assert "SNOMED:138875005" in ancestors
    assert resolver.is_subclass("SNOMED:44054006", "SNOMED:73211009") is True
    assert resolver.is_subclass("SNOMED:44054006", "HPO:0000118") is False


def test_direct_mapping_returns_confidence_sorted(tmp_path):
    resolver = make_resolver(tmp_path)
    mappings = resolver.map_term("HPO:0003074", "GO")

    assert len(mappings) == 1
    assert mappings[0].object_id == "GO:0042593"
    assert mappings[0].confidence == 0.91


def test_clinical_normalize_mixed_payload(tmp_path):
    resolver = make_resolver(tmp_path)
    normalized = resolver.clinical_normalize(
        {
            "patient_id": "pt-001",
            "diagnoses": ["SNOMED:44054006"],
            "labs": ["LOINC:4548-4"],
            "phenotypes": ["HPO:0003074"],
        }
    )

    assert normalized["patient_id"] == "pt-001"
    assert normalized["diagnoses"][0]["resolved"]["label"] == "Type 2 diabetes mellitus"
    assert normalized["labs"][0]["resolved"]["curie"] == "LOINC:4548-4"
    assert normalized["phenotypes"][0]["mappings"][0]["object_id"] == "GO:0042593"

    discovery_targets = {item["object_id"] for item in normalized["discovery_links"]}
    assert "CHEBI:6801" in discovery_targets
    assert "GO:0042593" in discovery_targets


def test_hybrid_resolve_uses_federated_fallback_on_local_miss(tmp_path, monkeypatch):
    resolver = make_resolver(tmp_path, mode="hybrid")

    def fake_remote(curie: str):
        assert curie == "GO:1234567"
        from sounio.ontology import ResolvedOntologyTerm

        return ResolvedOntologyTerm(
            curie=curie,
            label="Federated term",
            definition="Loaded from remote",
            parents=[],
            synonyms=["Remote synonym"],
            source_layer="federated",
            iri="http://example.org/GO_1234567",
            mapping_confidence=0.6,
            provenance="federated:test",
        )

    monkeypatch.setattr(resolver, "_resolve_federated", fake_remote)
    term = resolver.resolve("GO:1234567")

    assert term is not None
    assert term.source_layer == "federated"
    assert term.provenance == "federated:test"


def test_hybrid_search_uses_remote_only_when_local_empty(tmp_path, monkeypatch):
    resolver = make_resolver(tmp_path, mode="hybrid")

    called = {"remote": False}

    def fake_search(query: str, limit: int = 10):
        called["remote"] = True
        from sounio.ontology import ResolvedOntologyTerm

        return [
            ResolvedOntologyTerm(
                curie="GO:9999999",
                label=f"Remote {query}",
                definition="",
                parents=[],
                synonyms=[],
                source_layer="federated",
                iri="http://example.org/GO_9999999",
                mapping_confidence=0.6,
                provenance="federated:test-search",
            )
        ]

    monkeypatch.setattr(resolver, "_search_federated", fake_search)

    local_hits = resolver.search("glucose")
    assert local_hits
    assert called["remote"] is False

    remote_hits = resolver.search("no-local-hit-expected")
    assert called["remote"] is True
    assert remote_hits[0].curie == "GO:9999999"
