"""Pure Python Knowledge class — GUM-based epistemic value with uncertainty propagation.

This module provides a Python-native fallback implementation. When the Rust
extension (sounio._sounio_native) is available it is preferred for performance.
"""

from __future__ import annotations

import math
from typing import Union


_Numeric = Union[float, int, "Knowledge"]


class Knowledge:
    """An epistemic quantity: a measured value with GUM standard uncertainty.

    Attributes
    ----------
    value : float
        Central estimate of the quantity.
    epsilon : float
        Standard uncertainty (k=1, one sigma) in the same units as value.
    provenance : str
        Free-form label describing the measurement source.

    GUM propagation rules implemented
    ----------------------------------
    Addition / subtraction : ε = sqrt(εa² + εb²)
    Multiplication         : ε = |a·b| · sqrt((εa/a)² + (εb/b)²)
    Division               : ε = |a/b| · sqrt((εa/a)² + (εb/b)²)
    Scalar multiplication  : ε = |factor| · εa
    """

    __slots__ = ("value", "epsilon", "provenance")

    def __init__(
        self,
        value: float,
        epsilon: float = 0.0,
        provenance: str = "",
    ) -> None:
        self.value = float(value)
        self.epsilon = float(epsilon)
        self.provenance = str(provenance)

    # ---- Arithmetic -------------------------------------------------------

    def __add__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            return Knowledge(
                self.value + other.value,
                math.sqrt(self.epsilon**2 + other.epsilon**2),
                f"({self.provenance})+({other.provenance})",
            )
        other = float(other)
        return Knowledge(self.value + other, self.epsilon, self.provenance)

    def __radd__(self, other: _Numeric) -> "Knowledge":
        return self.__add__(other)

    def __sub__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            return Knowledge(
                self.value - other.value,
                math.sqrt(self.epsilon**2 + other.epsilon**2),
                f"({self.provenance})-({other.provenance})",
            )
        other = float(other)
        return Knowledge(self.value - other, self.epsilon, self.provenance)

    def __rsub__(self, other: _Numeric) -> "Knowledge":
        other = float(other)
        return Knowledge(other - self.value, self.epsilon, self.provenance)

    def __mul__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            val = self.value * other.value
            rel_a = self.epsilon / self.value if self.value != 0.0 else 0.0
            rel_b = other.epsilon / other.value if other.value != 0.0 else 0.0
            # Use absolute values for relative uncertainty
            rel_a = self.epsilon / abs(self.value) if self.value != 0.0 else 0.0
            rel_b = other.epsilon / abs(other.value) if other.value != 0.0 else 0.0
            return Knowledge(
                val,
                abs(val) * math.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})*({other.provenance})",
            )
        factor = float(other)
        return Knowledge(
            self.value * factor,
            self.epsilon * abs(factor),
            self.provenance,
        )

    def __rmul__(self, other: _Numeric) -> "Knowledge":
        return self.__mul__(other)

    def __truediv__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            if other.value == 0.0:
                raise ZeroDivisionError("Knowledge division by zero")
            val = self.value / other.value
            rel_a = self.epsilon / abs(self.value) if self.value != 0.0 else 0.0
            rel_b = other.epsilon / abs(other.value) if other.value != 0.0 else 0.0
            return Knowledge(
                val,
                abs(val) * math.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})/({other.provenance})",
            )
        divisor = float(other)
        if divisor == 0.0:
            raise ZeroDivisionError("Knowledge division by zero scalar")
        return Knowledge(
            self.value / divisor,
            self.epsilon / abs(divisor),
            self.provenance,
        )

    def __rtruediv__(self, other: _Numeric) -> "Knowledge":
        if self.value == 0.0:
            raise ZeroDivisionError("Knowledge division by zero")
        numerator = float(other)
        val = numerator / self.value
        rel = self.epsilon / abs(self.value)
        return Knowledge(val, abs(val) * rel, self.provenance)

    def __neg__(self) -> "Knowledge":
        return Knowledge(-self.value, self.epsilon, f"-({self.provenance})")

    def __abs__(self) -> "Knowledge":
        return Knowledge(abs(self.value), self.epsilon, f"|{self.provenance}|")

    # ---- Comparison (by central value) ------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Knowledge):
            return (
                abs(self.value - other.value) < 1e-15
                and abs(self.epsilon - other.epsilon) < 1e-15
                and self.provenance == other.provenance
            )
        return NotImplemented

    def __lt__(self, other: _Numeric) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value < v

    def __le__(self, other: _Numeric) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value <= v

    def __gt__(self, other: _Numeric) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value > v

    def __ge__(self, other: _Numeric) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value >= v

    def __hash__(self) -> int:
        return hash((self.value, self.epsilon, self.provenance))

    # ---- Derived properties -----------------------------------------------

    @property
    def relative_uncertainty(self) -> float:
        """ε / |value|, or 0.0 when value is zero."""
        if self.value == 0.0:
            return 0.0
        return self.epsilon / abs(self.value)

    @property
    def confidence(self) -> float:
        """1 − relative_uncertainty, clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.relative_uncertainty))

    def is_reliable(self, threshold: float = 0.05) -> bool:
        """Return True if relative uncertainty is below *threshold* (default 5%)."""
        return self.relative_uncertainty < threshold

    def scale(self, factor: float) -> "Knowledge":
        """Multiply by a scalar without adding new uncertainty."""
        return Knowledge(self.value * factor, self.epsilon * abs(factor), self.provenance)

    # ---- Serialization helpers --------------------------------------------

    def to_dict(self) -> dict:
        return {"value": self.value, "epsilon": self.epsilon, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d: dict) -> "Knowledge":
        return cls(d["value"], d.get("epsilon", 0.0), d.get("provenance", ""))

    # ---- Representation ---------------------------------------------------

    def __repr__(self) -> str:
        return (
            f'Knowledge {{ value: {self.value:.3f} epsilon: {self.epsilon:.3f}'
            f' prov: "{self.provenance}" }}'
        )

    def __str__(self) -> str:
        return f"Knowledge({self.value:.3f} ± {self.epsilon:.3f}, prov='{self.provenance}')"

    def __format__(self, spec: str) -> str:
        if spec == "":
            return str(self)
        # Allow e.g. f"{k:.6f}" to format the central value
        return format(self.value, spec)
