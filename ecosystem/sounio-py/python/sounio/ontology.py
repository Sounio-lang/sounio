"""Hybrid local + federated ontology resolver for Sounio.

Phase 1 goals:
- load local .dontology biomedical shards
- resolve/search/ancestors/is_subclass against local bundles
- map terms via SSSOM-like mapping shards
- persist positive and negative cache entries
- optionally query federated sources on misses
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error, parse, request


MAGIC = b"DONT"
VERSION = 1
DEFAULT_MODE = "hybrid"
SUPPORTED_ONTOLOGIES = {"SNOMED", "LOINC", "HPO", "GO", "CHEBI"}
CURIE_RE = re.compile(r"^(SNOMED|LOINC|HPO|GO|CHEBI):[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ResolvedOntologyTerm:
    curie: str
    label: str
    definition: str
    parents: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    source_layer: str = "local"
    iri: str = ""
    mapping_confidence: float = 1.0
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curie": self.curie,
            "label": self.label,
            "definition": self.definition,
            "parents": list(self.parents),
            "synonyms": list(self.synonyms),
            "source_layer": self.source_layer,
            "iri": self.iri,
            "mapping_confidence": self.mapping_confidence,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class OntologyMapping:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    justification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "justification": self.justification,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_ontology_home() -> Path:
    env = os.environ.get("SOUNIO_ONTOLOGY_HOME")
    if env:
        return Path(env)
    return _repo_root() / "stdlib" / "data" / "data" / "ontology" / "bundles"


def _default_cache_dir() -> Path:
    env = os.environ.get("SOUNIO_ONTOLOGY_CACHE_DIR")
    if env:
        return Path(env)
    return _repo_root() / "data" / "ontology" / "cache"


def _default_mode() -> str:
    return os.environ.get("SOUNIO_ONTOLOGY_MODE", DEFAULT_MODE).lower()


def _safe_cache_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)


def _read_bundle(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"bundle too short: {path}")
    magic = raw[:4]
    if magic != MAGIC:
        raise ValueError(f"invalid bundle magic in {path}: {magic!r}")
    version, size = struct.unpack("<II", raw[4:12])
    if version != VERSION:
        raise ValueError(f"unsupported bundle version in {path}: {version}")
    payload = raw[12 : 12 + size]
    return json.loads(payload.decode("utf-8"))


class OntologyResolver:
    """Resolve biomedical ontology terms from local bundles and federated APIs."""

    def __init__(
        self,
        ontology_home: Optional[str | Path] = None,
        mode: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.ontology_home = Path(ontology_home) if ontology_home else _default_ontology_home()
        self.mode = (mode or _default_mode()).lower()
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bundles: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._search_index: List[Dict[str, Any]] = []
        self._mappings: List[OntologyMapping] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        self._bundles.clear()
        self._search_index.clear()
        for bundle_path in sorted(self.ontology_home.glob("*.dontology")):
            payload = _read_bundle(bundle_path)
            ontology = str(payload["ontology"]).upper()
            term_table: Dict[str, Dict[str, Any]] = {}
            for term in payload.get("terms", []):
                curie = str(term["curie"])
                enriched = {
                    **term,
                    "ontology": ontology,
                    "source_layer": "local",
                    "provenance": f"local:{ontology.lower()}",
                    "mapping_confidence": 1.0,
                }
                term_table[curie] = enriched
                self._search_index.append(enriched)
            self._bundles[ontology] = term_table

        self._mappings.clear()
        mapping_dir = self.ontology_home / "mappings"
        if mapping_dir.exists():
            for mapping_path in sorted(mapping_dir.glob("*.sssom.json")):
                doc = json.loads(mapping_path.read_text())
                for mapping in doc.get("mappings", []):
                    self._mappings.append(
                        OntologyMapping(
                            subject_id=mapping["subject_id"],
                            predicate=mapping["predicate"],
                            object_id=mapping["object_id"],
                            confidence=float(mapping["confidence"]),
                            justification=mapping.get("justification", ""),
                        )
                    )

        self._loaded = True

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, curie: str) -> Path:
        return self.cache_dir / f"{_safe_cache_name(curie)}.json"

    def _write_cache(self, curie: str, payload: Dict[str, Any]) -> None:
        self._cache_path(curie).write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _read_cache(self, curie: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(curie)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, curie: str) -> Optional[ResolvedOntologyTerm]:
        self.load()
        normalized = curie.strip().upper()
        if not normalized or not CURIE_RE.match(normalized):
            return None

        cached = self._read_cache(normalized)
        if cached is not None:
            if cached.get("missing"):
                return None
            return ResolvedOntologyTerm(**cached["term"])

        ontology = normalized.split(":", 1)[0] if ":" in normalized else ""
        if ontology in self._bundles and normalized in self._bundles[ontology]:
            raw = self._bundles[ontology][normalized]
            term = ResolvedOntologyTerm(
                curie=raw["curie"],
                label=raw["label"],
                definition=raw.get("definition", ""),
                parents=list(raw.get("parents", [])),
                synonyms=list(raw.get("synonyms", [])),
                source_layer=raw.get("source_layer", "local"),
                iri=raw.get("iri", ""),
                mapping_confidence=float(raw.get("mapping_confidence", 1.0)),
                provenance=raw.get("provenance", f"local:{ontology.lower()}"),
            )
            self._write_cache(normalized, {"missing": False, "term": term.to_dict()})
            return term

        if self.mode in {"hybrid", "federated"}:
            remote = self._resolve_federated(normalized)
            if remote is not None:
                self._write_cache(normalized, {"missing": False, "term": remote.to_dict()})
                return remote

        self._write_cache(normalized, {"missing": True, "curie": normalized})
        return None

    def search(self, query: str, limit: int = 10) -> List[ResolvedOntologyTerm]:
        self.load()
        q = query.strip().lower()
        hits: List[ResolvedOntologyTerm] = []
        for term in self._search_index:
            haystack = " ".join(
                [term["curie"], term["label"], *term.get("synonyms", [])]
            ).lower()
            if q in haystack:
                hits.append(
                    ResolvedOntologyTerm(
                        curie=term["curie"],
                        label=term["label"],
                        definition=term.get("definition", ""),
                        parents=list(term.get("parents", [])),
                        synonyms=list(term.get("synonyms", [])),
                        source_layer=term.get("source_layer", "local"),
                        iri=term.get("iri", ""),
                        mapping_confidence=float(term.get("mapping_confidence", 1.0)),
                        provenance=term.get("provenance", "local"),
                    )
                )
                if len(hits) >= limit:
                    break
        if hits:
            return hits
        if self.mode in {"hybrid", "federated"}:
            return self._search_federated(query, limit=limit)
        return hits

    def ancestors(self, curie: str) -> List[str]:
        visited: set[str] = set()
        queue: List[str] = [curie]
        result: List[str] = []
        while queue:
            current = queue.pop(0)
            term = self.resolve(current)
            if term is None:
                continue
            for parent in term.parents:
                if parent not in visited:
                    visited.add(parent)
                    result.append(parent)
                    queue.append(parent)
        return result

    def is_subclass(self, child: str, parent: str) -> bool:
        if child.upper() == parent.upper():
            return True
        return parent.upper() in {ancestor.upper() for ancestor in self.ancestors(child)}

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def map_term(self, source_curie: str, to_ontology: str) -> List[OntologyMapping]:
        self.load()
        target = to_ontology.strip().upper()
        direct = [
            mapping
            for mapping in self._mappings
            if mapping.subject_id.upper() == source_curie.upper()
            and mapping.object_id.upper().startswith(f"{target}:")
        ]
        if direct:
            return sorted(direct, key=lambda item: item.confidence, reverse=True)

        source_term = self.resolve(source_curie)
        if source_term is None:
            return []

        # Fallback heuristic: label overlap search when no explicit mapping exists.
        target_hits = self.search(source_term.label, limit=5)
        inferred: List[OntologyMapping] = []
        for hit in target_hits:
            if not hit.curie.startswith(f"{target}:"):
                continue
            inferred.append(
                OntologyMapping(
                    subject_id=source_curie,
                    predicate="relatedMatch",
                    object_id=hit.curie,
                    confidence=0.35,
                    justification="inferred label overlap fallback",
                )
            )
        return inferred

    # ------------------------------------------------------------------
    # Clinical normalization
    # ------------------------------------------------------------------

    def clinical_normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        diagnoses = self._normalize_codes(payload.get("diagnoses", []), "HPO")
        labs = self._normalize_codes(payload.get("labs", []), "HPO")
        phenotypes = self._normalize_codes(payload.get("phenotypes", []), "GO")

        discovery_links: List[Dict[str, Any]] = []
        for bucket in ("diagnoses", "phenotypes"):
            for code in payload.get(bucket, []):
                for target in ("GO", "CHEBI"):
                    for mapping in self.map_term(code, target):
                        discovery_links.append(mapping.to_dict())

        return {
            "patient_id": payload.get("patient_id", ""),
            "diagnoses": diagnoses,
            "labs": labs,
            "phenotypes": phenotypes,
            "discovery_links": discovery_links,
            "provenance": "clinical_normalize[local_bundle,federated_cache,mappings]",
        }

    def _normalize_codes(self, codes: Iterable[str], bridge_target: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for code in codes:
            resolved = self.resolve(code)
            mappings = [m.to_dict() for m in self.map_term(code, bridge_target)]
            out.append(
                {
                    "code": code,
                    "resolved": resolved.to_dict() if resolved else None,
                    "mappings": mappings,
                    "status": "resolved" if resolved else "unresolved",
                }
            )
        return out

    # ------------------------------------------------------------------
    # Federation
    # ------------------------------------------------------------------

    def _search_federated(self, query: str, limit: int = 10) -> List[ResolvedOntologyTerm]:
        for resolver in (self._search_via_ols4, self._search_via_bioportal):
            try:
                results = resolver(query, limit=limit)
            except Exception:
                results = []
            if results:
                return results
        return []

    def _resolve_federated(self, curie: str) -> Optional[ResolvedOntologyTerm]:
        for resolver in (self._resolve_via_ols4, self._resolve_via_bioportal):
            try:
                term = resolver(curie)
            except Exception:
                term = None
            if term is not None:
                return term
        return None

    def _resolve_via_ols4(self, curie: str) -> Optional[ResolvedOntologyTerm]:
        base = os.environ.get("SOUNIO_ONTOLOGY_OLS4_BASE_URL")
        if not base:
            return None
        url = f"{base.rstrip('/')}/api/search?q={parse.quote(curie)}&exact=true"
        payload = self._fetch_json(url)
        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            return None
        doc = docs[0]
        return ResolvedOntologyTerm(
            curie=doc.get("obo_id", curie),
            label=doc.get("label", curie),
            definition="",
            parents=[],
            synonyms=list(doc.get("synonym", []) or []),
            source_layer="federated",
            iri=doc.get("iri", ""),
            mapping_confidence=0.6,
            provenance="federated:ols4",
        )

    def _search_via_ols4(self, query: str, limit: int = 10) -> List[ResolvedOntologyTerm]:
        base = os.environ.get("SOUNIO_ONTOLOGY_OLS4_BASE_URL")
        if not base:
            return []
        url = f"{base.rstrip('/')}/api/search?q={parse.quote(query)}&rows={limit}"
        payload = self._fetch_json(url)
        docs = payload.get("response", {}).get("docs", [])
        out: List[ResolvedOntologyTerm] = []
        for doc in docs[:limit]:
            curie = doc.get("obo_id", "")
            if not curie:
                continue
            out.append(
                ResolvedOntologyTerm(
                    curie=curie,
                    label=doc.get("label", curie),
                    definition="",
                    parents=[],
                    synonyms=list(doc.get("synonym", []) or []),
                    source_layer="federated",
                    iri=doc.get("iri", ""),
                    mapping_confidence=0.6,
                    provenance="federated:ols4",
                )
            )
        return out

    def _resolve_via_bioportal(self, curie: str) -> Optional[ResolvedOntologyTerm]:
        api_key = os.environ.get("SOUNIO_ONTOLOGY_BIOPORTAL_API_KEY")
        if not api_key:
            return None
        url = f"https://data.bioontology.org/search?q={parse.quote(curie)}"
        payload = self._fetch_json(url, headers={"Authorization": f"apikey token={api_key}"})
        collection = payload.get("collection", [])
        if not collection:
            return None
        doc = collection[0]
        return ResolvedOntologyTerm(
            curie=doc.get("@id", curie).split("/")[-1].replace("_", ":"),
            label=doc.get("prefLabel", curie),
            definition=doc.get("definition", [""])[0] if isinstance(doc.get("definition"), list) else "",
            parents=[],
            synonyms=list(doc.get("synonym", []) or []),
            source_layer="federated",
            iri=doc.get("@id", ""),
            mapping_confidence=0.55,
            provenance="federated:bioportal",
        )

    def _search_via_bioportal(self, query: str, limit: int = 10) -> List[ResolvedOntologyTerm]:
        api_key = os.environ.get("SOUNIO_ONTOLOGY_BIOPORTAL_API_KEY")
        if not api_key:
            return []
        url = f"https://data.bioontology.org/search?q={parse.quote(query)}"
        payload = self._fetch_json(url, headers={"Authorization": f"apikey token={api_key}"})
        collection = payload.get("collection", [])
        out: List[ResolvedOntologyTerm] = []
        for doc in collection[:limit]:
            iri = doc.get("@id", "")
            curie = iri.split("/")[-1].replace("_", ":") if iri else ""
            if not curie:
                continue
            out.append(
                ResolvedOntologyTerm(
                    curie=curie,
                    label=doc.get("prefLabel", curie),
                    definition=doc.get("definition", [""])[0] if isinstance(doc.get("definition"), list) else "",
                    parents=[],
                    synonyms=list(doc.get("synonym", []) or []),
                    source_layer="federated",
                    iri=iri,
                    mapping_confidence=0.55,
                    provenance="federated:bioportal",
                )
            )
        return out

    def _fetch_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        req = request.Request(url, headers=headers or {})
        with request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))


_DEFAULT_RESOLVER: Optional[OntologyResolver] = None


def get_resolver() -> OntologyResolver:
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = OntologyResolver()
    return _DEFAULT_RESOLVER


def reset_resolver() -> None:
    global _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = None


def resolve(curie: str) -> Optional[ResolvedOntologyTerm]:
    return get_resolver().resolve(curie)


def search(query: str, limit: int = 10) -> List[ResolvedOntologyTerm]:
    return get_resolver().search(query, limit=limit)


def ancestors(curie: str) -> List[str]:
    return get_resolver().ancestors(curie)


def is_subclass(child: str, parent: str) -> bool:
    return get_resolver().is_subclass(child, parent)


def map_term(source_curie: str, to_ontology: str) -> List[OntologyMapping]:
    return get_resolver().map_term(source_curie, to_ontology)


def clinical_normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    return get_resolver().clinical_normalize(payload)


__all__ = [
    "ResolvedOntologyTerm",
    "OntologyMapping",
    "OntologyResolver",
    "get_resolver",
    "reset_resolver",
    "resolve",
    "search",
    "ancestors",
    "is_subclass",
    "map_term",
    "clinical_normalize",
]
