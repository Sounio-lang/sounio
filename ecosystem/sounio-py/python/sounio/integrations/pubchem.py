"""PubChem PUG REST integration for Sounio.

Uses only stdlib HTTP (``urllib.request``) so no third-party dependencies are
required.  An offline cache of 10 common drugs is bundled to allow unit tests
and CI to run without network access.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional

from ..knowledge import Knowledge
from ..types import Molecule


# ---------------------------------------------------------------------------
# Offline cache
# ---------------------------------------------------------------------------

# Each entry: (name, smiles, mw, logp, hbd, hba)
# MW values from PubChem / authoritative references; XLogP3 values from
# PubChem where available.  MW epsilon=0.001 (exact formula weight);
# LogP epsilon=0.5 (XLogP3 precision).

_OFFLINE_CACHE: Dict[str, Dict] = {
    "aspirin": {
        "name": "aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "mw": 180.159,
        "logp": 1.2,
        "hbd": 1,
        "hba": 4,
        "cid": 2244,
    },
    "acetylsalicylic acid": {  # alias
        "name": "aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "mw": 180.159,
        "logp": 1.2,
        "hbd": 1,
        "hba": 4,
        "cid": 2244,
    },
    "ibuprofen": {
        "name": "ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "mw": 206.285,
        "logp": 3.97,
        "hbd": 1,
        "hba": 2,
        "cid": 3672,
    },
    "metformin": {
        "name": "metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "mw": 129.164,
        "logp": -1.43,
        "hbd": 4,
        "hba": 5,
        "cid": 4091,
    },
    "paracetamol": {
        "name": "paracetamol",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "mw": 151.163,
        "logp": 0.46,
        "hbd": 2,
        "hba": 3,
        "cid": 1983,
    },
    "acetaminophen": {  # alias
        "name": "paracetamol",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "mw": 151.163,
        "logp": 0.46,
        "hbd": 2,
        "hba": 3,
        "cid": 1983,
    },
    "caffeine": {
        "name": "caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "mw": 194.194,
        "logp": -0.07,
        "hbd": 0,
        "hba": 6,
        "cid": 2519,
    },
    "penicillin": {
        "name": "penicillin",
        "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
        "mw": 334.390,
        "logp": 1.83,
        "hbd": 2,
        "hba": 6,
        "cid": 5904,
    },
    "insulin": {
        # Small-molecule approximation of insulin A-chain fragment (MW
        # approximation; true insulin is a 5808 Da protein and cannot be
        # represented as a simple SMILES).  This entry is intentionally a
        # rough approximation for offline-cache testing purposes.
        "name": "insulin",
        "smiles": "N",  # placeholder — insulin is a macromolecule
        "mw": 5807.6,
        "logp": -3.0,
        "hbd": 40,
        "hba": 40,
        "cid": 16132359,
    },
    "warfarin": {
        "name": "warfarin",
        "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "mw": 308.328,
        "logp": 2.7,
        "hbd": 1,
        "hba": 4,
        "cid": 54678486,
    },
    "omeprazole": {
        "name": "omeprazole",
        "smiles": "COc1ccc2c(c1)nc(n2)[S@@](=O)Cc1ncc(C)c(OC)c1C",
        "mw": 345.417,
        "logp": 2.23,
        "hbd": 1,
        "hba": 6,
        "cid": 4594,
    },
    "atorvastatin": {
        "name": "atorvastatin",
        "smiles": "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c(C(=O)Nc2ccccc2)c1-c1ccc(F)cc1",
        "mw": 558.644,
        "logp": 4.06,
        "hbd": 4,
        "hba": 8,
        "cid": 60823,
    },
}

# CID → name mapping for CID-based lookups
_CID_TO_NAME: Dict[int, str] = {
    v["cid"]: k
    for k, v in _OFFLINE_CACHE.items()
    if "cid" in v
}

_PUGREST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

_MW_EPSILON = 0.001   # exact formula weight — sub-mDa uncertainty
_LOGP_EPSILON = 0.5   # XLogP3 algorithm precision


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entry_to_molecule(entry: Dict) -> Molecule:
    """Convert an offline-cache or parsed PubChem entry dict to a Molecule."""
    mw = float(entry["mw"])
    logp = float(entry["logp"])
    mol = Molecule(
        name=entry["name"],
        smiles=entry["smiles"],
        molecular_weight=Knowledge(mw, _MW_EPSILON, "pubchem"),
        logp=Knowledge(logp, _LOGP_EPSILON, "pubchem_xlogp3"),
        hbd=int(entry["hbd"]),
        hba=int(entry["hba"]),
    )
    mol.lipinski_rule_of_five()
    return mol


def _get_json(url: str) -> dict:
    """Fetch a URL and return parsed JSON, raising on HTTP errors."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise ValueError(f"PubChem HTTP error {exc.code} for URL: {url}") from exc
    except Exception as exc:
        raise ConnectionError(f"Failed to reach PubChem: {exc}") from exc


def _parse_pubchem_response(data: dict, cid: Optional[int] = None) -> Molecule:
    """Parse a PubChem PUG REST property response into a Molecule."""
    try:
        props = data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected PubChem response structure: {exc}") from exc

    mw = float(props.get("MolecularWeight", 0.0))
    logp = float(props.get("XLogP", 0.0))
    hbd = int(props.get("HBondDonorCount", 0))
    hba = int(props.get("HBondAcceptorCount", 0))
    smiles = str(props.get("IsomericSMILES", ""))
    name = str(props.get("IUPACName", smiles or str(cid or "")))

    mol = Molecule(
        name=name,
        smiles=smiles,
        molecular_weight=Knowledge(mw, _MW_EPSILON, "pubchem"),
        logp=Knowledge(logp, _LOGP_EPSILON, "pubchem_xlogp3"),
        hbd=hbd,
        hba=hba,
    )
    mol.lipinski_rule_of_five()
    return mol


_PROPERTY_LIST = "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,IsomericSMILES,IUPACName"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_by_name(name: str, offline: bool = False) -> Molecule:
    """Fetch molecule properties from PubChem by compound name.

    Parameters
    ----------
    name : str
        Common or IUPAC name (e.g. ``"aspirin"``).
    offline : bool
        When True, only consult the built-in cache; raise ``KeyError`` on
        a cache miss.  Default: False (network request on cache miss).

    Returns
    -------
    Molecule
        Populated Molecule with epistemic uncertainty on MW and LogP.

    Raises
    ------
    KeyError
        When *offline* is True and *name* is not in the cache.
    ValueError
        On an invalid PubChem response.
    ConnectionError
        On network failure.
    """
    key = name.lower().strip()
    if key in _OFFLINE_CACHE:
        return _entry_to_molecule(_OFFLINE_CACHE[key])

    if offline:
        raise KeyError(f"Compound {name!r} not in offline cache")

    url = (
        f"{_PUGREST_BASE}/compound/name/"
        f"{urllib.parse.quote(name)}/property/"
        f"{_PROPERTY_LIST}/JSON"
    )
    data = _get_json(url)
    return _parse_pubchem_response(data)


def fetch_by_cid(cid: int, offline: bool = False) -> Molecule:
    """Fetch molecule properties from PubChem by compound CID.

    Parameters
    ----------
    cid : int
        PubChem Compound Identifier (CID).
    offline : bool
        When True, only consult the built-in cache.

    Returns
    -------
    Molecule
        Populated Molecule with epistemic uncertainty on MW and LogP.

    Raises
    ------
    KeyError
        When *offline* is True and *cid* is not in the cache.
    ValueError
        On an invalid PubChem response.
    ConnectionError
        On network failure.
    """
    if cid in _CID_TO_NAME:
        name = _CID_TO_NAME[cid]
        return _entry_to_molecule(_OFFLINE_CACHE[name])

    if offline:
        raise KeyError(f"CID {cid} not in offline cache")

    url = (
        f"{_PUGREST_BASE}/compound/cid/{cid}/property/"
        f"{_PROPERTY_LIST}/JSON"
    )
    data = _get_json(url)
    return _parse_pubchem_response(data, cid=cid)


def search_molecules(query: str, max_results: int = 5) -> List[Molecule]:
    """Search PubChem for molecules matching *query* and return top results.

    The function first performs a full-text/name search to obtain CIDs, then
    fetches properties for each.  The offline cache is consulted before any
    network call.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int
        Maximum number of results to return (default 5).

    Returns
    -------
    list[Molecule]
        Up to *max_results* molecules with properties populated.
    """
    # Check offline cache first (case-insensitive substring match)
    q = query.lower().strip()
    results: List[Molecule] = []
    seen_names: set = set()

    for key, entry in _OFFLINE_CACHE.items():
        if q in key or q in entry["name"].lower():
            canonical = entry["name"]
            if canonical not in seen_names:
                seen_names.add(canonical)
                results.append(_entry_to_molecule(entry))
            if len(results) >= max_results:
                return results

    # Fall back to PubChem name search for remaining slots
    remaining = max_results - len(results)
    if remaining <= 0:
        return results

    try:
        search_url = (
            f"{_PUGREST_BASE}/compound/name/"
            f"{urllib.parse.quote(query)}/cids/JSON"
            f"?MaxRecords={remaining + 10}"  # fetch extra in case some fail
        )
        cid_data = _get_json(search_url)
        cids: List[int] = cid_data.get("IdentifierList", {}).get("CID", [])
    except (ValueError, ConnectionError):
        return results

    for cid in cids:
        if len(results) >= max_results:
            break
        try:
            mol = fetch_by_cid(cid, offline=False)
            results.append(mol)
        except (ValueError, ConnectionError):
            continue

    return results[:max_results]
