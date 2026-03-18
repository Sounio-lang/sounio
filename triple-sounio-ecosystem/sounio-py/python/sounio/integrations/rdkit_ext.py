"""RDKit integration for Sounio: molecular property calculation with epistemic uncertainty.

RDKit is an optional dependency. All public functions raise ``ImportError`` with
a helpful message when rdkit is not installed rather than failing at import time.
"""

from __future__ import annotations

import math
from typing import List, Tuple, TYPE_CHECKING

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem import DataStructs
    from rdkit.Chem import AllChem
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    DataStructs = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from rdkit import Chem as _Chem  # noqa: F811

from ..knowledge import Knowledge
from ..types import Molecule


def _require_rdkit() -> None:
    if not _HAS_RDKIT:
        raise ImportError(
            "rdkit is required for this function. "
            "Install it with: pip install rdkit"
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> "Chem.Mol":
    """Parse a SMILES string into an RDKit Mol object.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.

    Returns
    -------
    Chem.Mol
        The parsed molecule.

    Raises
    ------
    ImportError
        When rdkit is not installed.
    ValueError
        When the SMILES string is invalid or cannot be parsed.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid or unparseable SMILES: {smiles!r}")
    return mol


# ---------------------------------------------------------------------------
# Property calculation
# ---------------------------------------------------------------------------


def calculate_properties(smiles: str) -> Molecule:
    """Compute physicochemical properties for a SMILES string via RDKit.

    Calculated properties: molecular weight, LogP (Crippen), hydrogen-bond
    donor count, hydrogen-bond acceptor count.  Each numeric result is wrapped
    in a :class:`~sounio.knowledge.Knowledge` with epsilon = 1% of the central
    value (reflecting typical inter-software computed-property spread) and
    provenance ``"rdkit_computed"``.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.

    Returns
    -------
    Molecule
        Populated :class:`~sounio.types.Molecule` dataclass.

    Raises
    ------
    ValueError
        When the SMILES is invalid.
    """
    _require_rdkit()
    mol = mol_from_smiles(smiles)

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    prov = "rdkit_computed"
    # epsilon = 1% of the computed value; use a floor of 1e-6 for zero values
    mw_eps = max(mw * 0.01, 1e-6)
    logp_eps = max(abs(logp) * 0.01, 1e-6)

    molecule = Molecule(
        name=smiles,
        smiles=smiles,
        molecular_weight=Knowledge(mw, mw_eps, prov),
        logp=Knowledge(logp, logp_eps, prov),
        hbd=int(hbd),
        hba=int(hba),
    )
    # Evaluate and cache Lipinski result
    molecule.lipinski_rule_of_five()
    return molecule


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------


def batch_screen(smiles_list: List[str]) -> List[Molecule]:
    """Calculate properties for a list of SMILES and filter by Lipinski Ro5.

    Invalid SMILES strings are silently skipped (a warning is not emitted to
    keep the function pure; callers that need error details should iterate
    :func:`calculate_properties` directly).

    Parameters
    ----------
    smiles_list : list[str]
        Input SMILES strings.

    Returns
    -------
    list[Molecule]
        Molecules that pass Lipinski's Rule of Five.
    """
    _require_rdkit()
    passed: List[Molecule] = []
    for smi in smiles_list:
        try:
            mol = calculate_properties(smi)
        except ValueError:
            continue
        if mol.passes_lipinski:
            passed.append(mol)
    return passed


# ---------------------------------------------------------------------------
# Fingerprint similarity
# ---------------------------------------------------------------------------


def fingerprint_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute the Tanimoto similarity between two molecules.

    Uses Morgan circular fingerprints with radius=2 and nBits=2048 (ECFP4
    equivalent).

    Parameters
    ----------
    smiles_a : str
        First SMILES string.
    smiles_b : str
        Second SMILES string.

    Returns
    -------
    float
        Tanimoto coefficient in [0.0, 1.0].

    Raises
    ------
    ValueError
        When either SMILES is invalid.
    """
    _require_rdkit()
    mol_a = mol_from_smiles(smiles_a)
    mol_b = mol_from_smiles(smiles_b)

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)

    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


# ---------------------------------------------------------------------------
# Lipinski Rule of Five (standalone, works on Molecule dataclass)
# ---------------------------------------------------------------------------


def lipinski_rule_of_five(mol: Molecule) -> Tuple[bool, List[str]]:
    """Evaluate Lipinski's Rule of Five for a :class:`~sounio.types.Molecule`.

    This function operates on the already-populated ``Molecule`` dataclass and
    does **not** require RDKit to be installed; it uses the values stored in
    the dataclass fields.

    Parameters
    ----------
    mol : Molecule
        A molecule with populated ``molecular_weight``, ``logp``, ``hbd``,
        and ``hba`` fields.

    Returns
    -------
    tuple[bool, list[str]]
        ``(passes, violations)`` where ``passes`` is True when no rules are
        violated and ``violations`` is a (possibly empty) list of human-readable
        violation strings.
    """
    violations: List[str] = []

    if mol.molecular_weight.value > 500.0:
        violations.append(
            f"MW {mol.molecular_weight.value:.2f} > 500 Da"
        )
    if mol.logp.value > 5.0:
        violations.append(f"LogP {mol.logp.value:.2f} > 5.0")
    if mol.hbd > 5:
        violations.append(f"HBD {mol.hbd} > 5")
    if mol.hba > 10:
        violations.append(f"HBA {mol.hba} > 10")

    passes = len(violations) == 0
    return passes, violations
