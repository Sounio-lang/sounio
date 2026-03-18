"""NumPy integration: UncertainArray — vectorised GUM uncertainty propagation.

Requires numpy >= 1.20.  Import guard is deliberately permissive so the rest
of the sounio package works without numpy installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as np  # noqa: F811

from ..knowledge import Knowledge


class UncertainArray:
    """A NumPy array paired with per-element GUM standard uncertainties.

    Parameters
    ----------
    values : array-like
        Central estimates.  Converted to float64 ndarray.
    epsilons : array-like, optional
        Per-element uncertainties.  Defaults to zeros.
    provenance : str
        Label for the dataset.

    Arithmetic
    ----------
    Addition / subtraction : ε = sqrt(εa² + εb²)  element-wise
    Multiplication         : ε = |a·b| · sqrt((εa/a)² + (εb/b)²)
    Division               : same relative-uncertainty rule
    Scalar operations      : uncertainty scales linearly with |factor|
    """

    def __init__(
        self,
        values,
        epsilons=None,
        provenance: str = "numpy",
    ) -> None:
        if not _HAS_NUMPY:
            raise ImportError("numpy is required for UncertainArray")
        self.values = np.asarray(values, dtype=np.float64)
        if epsilons is None:
            self.epsilons = np.zeros_like(self.values)
        else:
            self.epsilons = np.asarray(epsilons, dtype=np.float64)
        if self.values.shape != self.epsilons.shape:
            raise ValueError(
                f"values shape {self.values.shape} != epsilons shape {self.epsilons.shape}"
            )
        self.provenance = str(provenance)

    # ---- Arithmetic -------------------------------------------------------

    def __add__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            return UncertainArray(
                self.values + other.values,
                np.sqrt(self.epsilons**2 + other.epsilons**2),
                f"({self.provenance})+({other.provenance})",
            )
        scalar = float(other)
        return UncertainArray(self.values + scalar, self.epsilons.copy(), self.provenance)

    def __radd__(self, other) -> "UncertainArray":
        return self.__add__(other)

    def __sub__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            return UncertainArray(
                self.values - other.values,
                np.sqrt(self.epsilons**2 + other.epsilons**2),
                f"({self.provenance})-({other.provenance})",
            )
        scalar = float(other)
        return UncertainArray(self.values - scalar, self.epsilons.copy(), self.provenance)

    def __rsub__(self, other) -> "UncertainArray":
        scalar = float(other)
        return UncertainArray(scalar - self.values, self.epsilons.copy(), self.provenance)

    def __mul__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            vals = self.values * other.values
            # Relative uncertainties — suppress divide-by-zero warnings
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_a = np.where(self.values != 0, self.epsilons / np.abs(self.values), 0.0)
                rel_b = np.where(other.values != 0, other.epsilons / np.abs(other.values), 0.0)
            return UncertainArray(
                vals,
                np.abs(vals) * np.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})*({other.provenance})",
            )
        factor = float(other)
        return UncertainArray(
            self.values * factor,
            self.epsilons * abs(factor),
            self.provenance,
        )

    def __rmul__(self, other) -> "UncertainArray":
        return self.__mul__(other)

    def __truediv__(self, other: Union["UncertainArray", float, int]) -> "UncertainArray":
        if isinstance(other, UncertainArray):
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = self.values / other.values
                rel_a = np.where(self.values != 0, self.epsilons / np.abs(self.values), 0.0)
                rel_b = np.where(other.values != 0, other.epsilons / np.abs(other.values), 0.0)
            return UncertainArray(
                vals,
                np.abs(vals) * np.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})/({other.provenance})",
            )
        divisor = float(other)
        if divisor == 0.0:
            raise ZeroDivisionError("UncertainArray division by zero scalar")
        return UncertainArray(
            self.values / divisor,
            self.epsilons / abs(divisor),
            self.provenance,
        )

    def __neg__(self) -> "UncertainArray":
        return UncertainArray(-self.values, self.epsilons.copy(), f"-({self.provenance})")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx) -> Knowledge:
        return Knowledge(float(self.values[idx]), float(self.epsilons[idx]), self.provenance)

    # ---- Statistical summaries -------------------------------------------

    def mean(self) -> Knowledge:
        """Arithmetic mean with combined GUM uncertainty (ε / √n)."""
        n = len(self.values)
        if n == 0:
            return Knowledge(float("nan"), float("nan"), self.provenance)
        return Knowledge(
            float(np.mean(self.values)),
            float(np.sqrt(np.sum(self.epsilons**2))) / n,
            f"mean({self.provenance})",
        )

    def std(self) -> Knowledge:
        """Standard deviation with combined uncertainty estimate."""
        return Knowledge(
            float(np.std(self.values)),
            float(np.mean(self.epsilons)),
            f"std({self.provenance})",
        )

    def sum(self) -> Knowledge:
        """Element-wise sum with GUM-combined uncertainty."""
        return Knowledge(
            float(np.sum(self.values)),
            float(np.sqrt(np.sum(self.epsilons**2))),
            f"sum({self.provenance})",
        )

    def min(self) -> Knowledge:
        idx = int(np.argmin(self.values))
        return Knowledge(float(self.values[idx]), float(self.epsilons[idx]), self.provenance)

    def max(self) -> Knowledge:
        idx = int(np.argmax(self.values))
        return Knowledge(float(self.values[idx]), float(self.epsilons[idx]), self.provenance)

    # ---- Conversion -------------------------------------------------------

    def to_knowledge_list(self) -> List[Knowledge]:
        """Convert to a list of Knowledge objects."""
        return [
            Knowledge(float(v), float(e), self.provenance)
            for v, e in zip(self.values, self.epsilons)
        ]

    @classmethod
    def from_knowledge_list(cls, knowledges: List[Knowledge]) -> "UncertainArray":
        """Build an UncertainArray from a list of Knowledge objects."""
        if not knowledges:
            return cls([], [], "")
        values = [k.value for k in knowledges]
        epsilons = [k.epsilon for k in knowledges]
        prov = knowledges[0].provenance
        return cls(values, epsilons, prov)

    # ---- Representation ---------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UncertainArray(n={len(self)}, "
            f"mean={np.mean(self.values):.4g}, "
            f"mean_eps={np.mean(self.epsilons):.4g}, "
            f"prov={self.provenance!r})"
        )
