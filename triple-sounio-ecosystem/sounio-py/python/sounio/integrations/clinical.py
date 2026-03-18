"""Clinical trial data import from CDISC/SDTM domains.

This module provides functions to parse CDISC-standard clinical trial data
(Demographics, Exposure, Pharmacoconcentration) and return epistemic-aware
data structures with uncertainty tracking per FDA BMV 2018 guidelines.

Requires pandas.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False
    pd = None  # type: ignore[assignment]

from ..knowledge import Knowledge
from ..types import PatientData
from .pandas_ext import EpistemicDataFrame


def read_sdtm_domain(path: str, domain: Optional[str] = None) -> "pd.DataFrame":
    """Read CDISC/SDTM domain CSV (or SAS7BDAT if available).

    Parameters
    ----------
    path : str
        Path to CSV or SAS7BDAT file.
    domain : str, optional
        Two-letter SDTM domain code (DM, EX, PC, etc.).
        If not provided, auto-detected from filename stem (e.g., 'dm.csv' → 'DM').

    Returns
    -------
    pd.DataFrame
        The domain table with standard CDISC column names.

    Raises
    ------
    ImportError
        If pandas is not available.
    ValueError
        If file format is not recognized.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for read_sdtm_domain")

    # Auto-detect domain from filename if not provided
    if domain is None:
        import pathlib
        stem = pathlib.Path(path).stem.upper()
        domain = stem if len(stem) == 2 else None

    # Read based on file extension
    if path.endswith(".sas7bdat"):
        try:
            df = pd.read_sas(path, format="sas7bdat")
        except Exception as e:
            raise ValueError(f"Failed to read SAS7BDAT file: {e}")
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format (expected .csv or .sas7bdat): {path}"
        )

    return df


def parse_dm(
    df: "pd.DataFrame",
    weight_epsilon: float = 0.5,
) -> List[PatientData]:
    """Parse Demographics (DM) domain to PatientData list.

    Per FDA BMV 2018, body weight uncertainty defaults to ±0.5 kg.

    Parameters
    ----------
    df : pd.DataFrame
        Demographics domain table with columns:
        - USUBJID: Unique subject identifier
        - AGE: Age in years
        - SEX: M/F
        - RACE: Patient race (WHITE, BLACK, ASIAN, etc.)
        - ACTARMCD: Actual arm code (ACTIVE, PLACEBO, etc.)
        - WEIGHT: (optional) Body weight in kg; if missing, set to NaN

    weight_epsilon : float
        Standard uncertainty for weight in kg (default 0.5, FDA BMV 2018).

    Returns
    -------
    list[PatientData]
        Parsed patient records with weight as Knowledge.

    Raises
    ------
    ImportError
        If pandas is not available.
    KeyError
        If required columns are missing.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for parse_dm")

    required = {"USUBJID", "AGE", "SEX", "RACE", "ACTARMCD"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in DM domain: {missing}")

    patients = []
    for _, row in df.iterrows():
        # Extract weight with uncertainty
        if "WEIGHT" in df.columns and pd.notna(row.get("WEIGHT")):
            weight_val = float(row["WEIGHT"])
        else:
            weight_val = float("nan")

        weight = Knowledge(
            value=weight_val,
            epsilon=weight_epsilon,
            provenance=f"DM({row['USUBJID']})",
        )

        # Dose is 0 by default (will be populated by EX domain)
        dose = Knowledge(
            value=0.0,
            epsilon=0.0,
            provenance=f"DM({row['USUBJID']})",
        )

        patient = PatientData(
            patient_id=str(row["USUBJID"]),
            weight=weight,
            age=int(row["AGE"]),
            dose=dose,
        )
        patients.append(patient)

    return patients


def parse_ex(
    df: "pd.DataFrame",
    dose_epsilon_pct: float = 0.02,
) -> EpistemicDataFrame:
    """Parse Exposure (EX) domain to EpistemicDataFrame.

    Per FDA BMV 2018, dose uncertainty defaults to ±2% (relative).

    Parameters
    ----------
    df : pd.DataFrame
        Exposure domain table with columns:
        - USUBJID: Unique subject identifier
        - EXDOSE: Dose value
        - EXDOSU: Dose unit (e.g., 'mg')
        - EXSTDTC: Start datetime (ISO 8601)

    dose_epsilon_pct : float
        Relative dose uncertainty as a fraction (default 0.02 = 2%, FDA BMV 2018).

    Returns
    -------
    EpistemicDataFrame
        Contains dose column with knowledge-based uncertainty.
        Columns: dose, dose_epsilon, dose_prov, plus any additional source columns.

    Raises
    ------
    ImportError
        If pandas is not available.
    KeyError
        If required columns are missing.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for parse_ex")

    required = {"USUBJID", "EXDOSE", "EXDOSU", "EXSTDTC"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in EX domain: {missing}")

    rows: Dict[str, list] = {
        "subject_id": [],
        "dose": [],
        "dose_epsilon": [],
        "dose_prov": [],
        "dose_unit": [],
        "dose_date": [],
    }

    for _, row in df.iterrows():
        subject_id = str(row["USUBJID"])
        dose_val = float(row["EXDOSE"])
        dose_unit = str(row.get("EXDOSU", ""))
        dose_date = str(row.get("EXSTDTC", ""))

        # Calculate epsilon as dose_val * dose_epsilon_pct
        dose_eps = abs(dose_val) * dose_epsilon_pct

        rows["subject_id"].append(subject_id)
        rows["dose"].append(dose_val)
        rows["dose_epsilon"].append(dose_eps)
        rows["dose_prov"].append(f"EX({subject_id})")
        rows["dose_unit"].append(dose_unit)
        rows["dose_date"].append(dose_date)

    ex_df = pd.DataFrame(rows)
    return EpistemicDataFrame(ex_df)


def parse_pc(
    df: "pd.DataFrame",
    assay_cv: float = 0.15,
) -> EpistemicDataFrame:
    """Parse Pharmacoconcentration (PC) domain to EpistemicDataFrame.

    Per FDA BMV 2018, assay CV (coefficient of variation) defaults to 15%.

    Parameters
    ----------
    df : pd.DataFrame
        Pharmacoconcentration domain table with columns:
        - USUBJID: Unique subject identifier
        - PCSTRESN: Result (numeric concentration value)
        - PCSTRESU: Result unit (e.g., 'ng/mL')
        - PCTPTNUM: Planned time point number (0, 1, 4, etc.)

    assay_cv : float
        Assay coefficient of variation as a fraction (default 0.15 = 15%,
        FDA BMV 2018).

    Returns
    -------
    EpistemicDataFrame
        Contains concentration column with knowledge-based uncertainty.
        Columns: concentration, concentration_epsilon, concentration_prov, plus
        any additional source columns (subject_id, unit, timepoint).

    Raises
    ------
    ImportError
        If pandas is not available.
    KeyError
        If required columns are missing.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for parse_pc")

    required = {"USUBJID", "PCSTRESN", "PCSTRESU", "PCTPTNUM"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in PC domain: {missing}")

    rows: Dict[str, list] = {
        "subject_id": [],
        "concentration": [],
        "concentration_epsilon": [],
        "concentration_prov": [],
        "concentration_unit": [],
        "timepoint": [],
    }

    for _, row in df.iterrows():
        subject_id = str(row["USUBJID"])
        conc_val = float(row["PCSTRESN"])
        conc_unit = str(row.get("PCSTRESU", ""))
        timepoint = int(row.get("PCTPTNUM", -1))

        # Calculate epsilon as conc_val * assay_cv
        conc_eps = abs(conc_val) * assay_cv

        rows["subject_id"].append(subject_id)
        rows["concentration"].append(conc_val)
        rows["concentration_epsilon"].append(conc_eps)
        rows["concentration_prov"].append(f"PC({subject_id}:T{timepoint})")
        rows["concentration_unit"].append(conc_unit)
        rows["timepoint"].append(timepoint)

    pc_df = pd.DataFrame(rows)
    return EpistemicDataFrame(pc_df)


def load_study(directory: str) -> Dict[str, any]:
    """Load a complete clinical study from a directory.

    Loads DM + EX + PC domains from standard filenames (dm.csv, ex.csv, pc.csv)
    in the given directory.

    Parameters
    ----------
    directory : str
        Path to directory containing SDTM domain files.

    Returns
    -------
    dict
        Keys: 'dm', 'ex', 'pc'
        - 'dm': list[PatientData]
        - 'ex': EpistemicDataFrame
        - 'pc': EpistemicDataFrame

    Raises
    ------
    ImportError
        If pandas is not available.
    FileNotFoundError
        If required domain files are missing.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for load_study")

    import pathlib

    base_path = pathlib.Path(directory)
    dm_file = base_path / "dm.csv"
    ex_file = base_path / "ex.csv"
    pc_file = base_path / "pc.csv"

    # Load domains
    if not dm_file.exists():
        raise FileNotFoundError(f"DM domain file not found: {dm_file}")
    dm_df = read_sdtm_domain(str(dm_file), domain="DM")
    dm_patients = parse_dm(dm_df)

    if not ex_file.exists():
        raise FileNotFoundError(f"EX domain file not found: {ex_file}")
    ex_df = read_sdtm_domain(str(ex_file), domain="EX")
    ex_epistemic = parse_ex(ex_df)

    if not pc_file.exists():
        raise FileNotFoundError(f"PC domain file not found: {pc_file}")
    pc_df = read_sdtm_domain(str(pc_file), domain="PC")
    pc_epistemic = parse_pc(pc_df)

    # Merge EX doses into PatientData
    ex_doses = ex_epistemic.get_knowledge_column("dose")
    ex_subjects = ex_epistemic.df["subject_id"].tolist()

    for patient in dm_patients:
        # Find corresponding dose in EX
        for subj, dose_k in zip(ex_subjects, ex_doses):
            if subj == patient.patient_id:
                patient.dose = dose_k
                break

    return {
        "dm": dm_patients,
        "ex": ex_epistemic,
        "pc": pc_epistemic,
    }
