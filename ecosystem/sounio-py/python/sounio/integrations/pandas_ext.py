"""pandas integration: EpistemicDataFrame — DataFrame with uncertainty columns.

Requires pandas >= 1.3.  Convention: for every measured column ``col`` the
DataFrame contains:
  - ``col``           — central values
  - ``col_epsilon``   — standard uncertainties
  - ``col_prov``      — provenance labels (optional, defaults to col name)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    import numpy as np
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False
    pd = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

from ..knowledge import Knowledge


class EpistemicDataFrame:
    """A pandas DataFrame with per-column epistemic uncertainty.

    The underlying DataFrame stores triples ``(col, col_epsilon, col_prov)``
    for each measured column.

    Parameters
    ----------
    df : pd.DataFrame
        Existing DataFrame following the naming convention above.

    Examples
    --------
    >>> edf = EpistemicDataFrame.from_measurements({
    ...     "dose": [(100.0, 2.5, "syringe"), (200.0, 3.0, "syringe")],
    ...     "weight": [(70.0, 0.5), (85.0, 0.5)],
    ... })
    >>> edf.describe_epistemic()
    """

    def __init__(self, df: "pd.DataFrame") -> None:
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for EpistemicDataFrame")
        self._df = df.copy()

    # ---- Factory methods --------------------------------------------------

    @classmethod
    def from_measurements(
        cls,
        data: Dict[str, List[Union[Tuple[float, float], Tuple[float, float, str]]]],
    ) -> "EpistemicDataFrame":
        """Create from a mapping of column name → list of (value, epsilon[, prov]) tuples.

        Parameters
        ----------
        data : dict
            Keys are column names.  Each value is a list of tuples:
            - ``(value, epsilon)`` — provenance defaults to column name.
            - ``(value, epsilon, provenance)`` — explicit provenance.
        """
        rows: Dict[str, list] = {}
        for col, measurements in data.items():
            rows[col] = []
            rows[f"{col}_epsilon"] = []
            rows[f"{col}_prov"] = []
            for m in measurements:
                rows[col].append(float(m[0]))
                rows[f"{col}_epsilon"].append(float(m[1]))
                rows[f"{col}_prov"].append(str(m[2]) if len(m) > 2 else col)
        return cls(pd.DataFrame(rows))

    @classmethod
    def from_knowledge_columns(
        cls, data: Dict[str, List[Knowledge]]
    ) -> "EpistemicDataFrame":
        """Create from a mapping of column name → list of Knowledge values."""
        rows: Dict[str, list] = {}
        for col, knowledges in data.items():
            rows[col] = [k.value for k in knowledges]
            rows[f"{col}_epsilon"] = [k.epsilon for k in knowledges]
            rows[f"{col}_prov"] = [k.provenance for k in knowledges]
        return cls(pd.DataFrame(rows))

    # ---- Column access ----------------------------------------------------

    def get_knowledge_column(self, col: str) -> List[Knowledge]:
        """Return a list of Knowledge objects for the given column.

        Parameters
        ----------
        col : str
            Base column name (without ``_epsilon`` / ``_prov`` suffix).
        """
        if col not in self._df.columns:
            raise KeyError(f"Column {col!r} not found in EpistemicDataFrame")
        eps_col = f"{col}_epsilon"
        prov_col = f"{col}_prov"
        eps = self._df[eps_col] if eps_col in self._df.columns else [0.0] * len(self._df)
        prov = self._df[prov_col] if prov_col in self._df.columns else [col] * len(self._df)
        return [
            Knowledge(float(v), float(e), str(p))
            for v, e, p in zip(self._df[col], eps, prov)
        ]

    def set_knowledge_column(self, col: str, knowledges: List[Knowledge]) -> None:
        """Set (or overwrite) a Knowledge column."""
        self._df[col] = [k.value for k in knowledges]
        self._df[f"{col}_epsilon"] = [k.epsilon for k in knowledges]
        self._df[f"{col}_prov"] = [k.provenance for k in knowledges]

    @property
    def measured_columns(self) -> List[str]:
        """List of base column names that have associated _epsilon columns."""
        return [
            c for c in self._df.columns
            if not c.endswith(("_epsilon", "_prov")) and f"{c}_epsilon" in self._df.columns
        ]

    # ---- Statistics -------------------------------------------------------

    def describe_epistemic(self) -> "pd.DataFrame":
        """Return a DataFrame with mean, std, mean_epsilon, max_epsilon, and rel_uncertainty
        for each measured column."""
        stats: Dict[str, Dict[str, float]] = {}
        for col in self.measured_columns:
            eps_col = f"{col}_epsilon"
            vals = self._df[col]
            eps = self._df[eps_col]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (eps / vals.abs()).replace([np.inf, -np.inf], np.nan)
            stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "mean_epsilon": float(eps.mean()),
                "max_epsilon": float(eps.max()),
                "rel_uncertainty": float(rel.mean()),
            }
        if not stats:
            return pd.DataFrame()
        return pd.DataFrame(stats).T

    def epistemic_summary(self, col: str) -> Knowledge:
        """Return a single Knowledge summarising a column (mean ± pooled uncertainty)."""
        ks = self.get_knowledge_column(col)
        if not ks:
            return Knowledge(float("nan"), float("nan"), col)
        import math
        n = len(ks)
        mean_val = sum(k.value for k in ks) / n
        pooled_eps = math.sqrt(sum(k.epsilon**2 for k in ks)) / n
        return Knowledge(mean_val, pooled_eps, f"mean({col})")

    # ---- Convenience accessors --------------------------------------------

    @property
    def df(self) -> "pd.DataFrame":
        """The underlying pandas DataFrame."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        cols = self.measured_columns
        return (
            f"EpistemicDataFrame(rows={len(self)}, measured_cols={cols!r})"
        )

    def head(self, n: int = 5) -> "pd.DataFrame":
        return self._df.head(n)
