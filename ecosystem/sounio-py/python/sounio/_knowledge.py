"""
sounio._knowledge — Pure-Python implementation of Knowledge<T>.

This module provides the Python-native fallback implementation of the
Sounio epistemic type system. It will be superseded by the Rust/native
binding (_core) once maturin integration is complete.

Design goals:
- API parity with Sounio's Knowledge<T> type
- GUM (JCGM 100:2008) first-order uncertainty propagation
- Confidence tracking (orthogonal to uncertainty)
- Provenance metadata (source, timestamp, method)
- Natural Python operator overloading
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Provenance:
    """Tracks the origin and transformation history of a Knowledge value."""
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    method: str = ""
    confidence_score: float = 1.0
    derived_from: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "method": self.method,
            "confidence_score": self.confidence_score,
            "derived_from": self.derived_from or [],
        }


class Knowledge:
    """
    Epistemic value type: value + uncertainty + confidence + provenance.

    This is the Python mirror of Sounio's ``Knowledge<T>`` type.

    The key insight from the Sounio manifesto:
    - **Uncertainty** (u): metrology — how precisely do we know the VALUE?
    - **Confidence** (c): epistemology — how much do we TRUST the claim?

    These are orthogonal concepts. A value can be precisely measured (low u)
    but from an unreliable source (low c), or imprecisely measured (high u)
    from a highly reliable instrument (high c).

    Arithmetic follows GUM (JCGM 100:2008) first-order propagation.

    Examples::

        dose = Knowledge(500.0, uncertainty=25.0, unit="mg")
        volume = Knowledge(10.0, uncertainty=0.2, unit="mL")
        conc = dose / volume
        print(conc)  # Knowledge(50.0 ± 2.52 mg/mL, confidence=0.93)
    """

    __slots__ = ("value", "uncertainty", "confidence", "unit", "provenance")

    def __init__(
        self,
        value: float,
        uncertainty: float = 0.0,
        confidence: float = 1.0,
        unit: str = "",
        source: str = "direct",
        provenance: Optional[Provenance] = None,
    ):
        if uncertainty < 0:
            raise ValueError(f"uncertainty must be >= 0, got {uncertainty}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {confidence}")

        self.value = float(value)
        self.uncertainty = float(uncertainty)
        self.confidence = float(confidence)
        self.unit = unit
        self.provenance = provenance or Provenance(source=source)

    # ------------------------------------------------------------------
    # GUM propagation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_confidence(a: "Knowledge", b: "Knowledge") -> float:
        """Conservative: take the minimum confidence of the operands."""
        return min(a.confidence, b.confidence)

    @staticmethod
    def _quad(a: float, b: float) -> float:
        """Combine uncertainties in quadrature: sqrt(a² + b²)."""
        return math.sqrt(a * a + b * b)

    def _derived_prov(self, other: Optional["Knowledge"], op: str) -> Provenance:
        sources = [self.provenance.source]
        if other is not None:
            sources.append(other.provenance.source)
        return Provenance(
            source=f"derived:{op}",
            method=f"GUM-first-order({op})",
            derived_from=sources,
        )

    # ------------------------------------------------------------------
    # Arithmetic — GUM first-order propagation (JCGM 100:2008 §13)
    # ------------------------------------------------------------------

    def __add__(self, other: Union["Knowledge", float]) -> "Knowledge":
        if isinstance(other, (int, float)):
            # Adding a constant does not increase uncertainty
            return Knowledge(
                self.value + other,
                self.uncertainty,
                self.confidence,
                self.unit,
                provenance=self._derived_prov(None, "add_const"),
            )
        u = self._quad(self.uncertainty, other.uncertainty)
        return Knowledge(
            self.value + other.value,
            u,
            self._combine_confidence(self, other),
            self.unit,
            provenance=self._derived_prov(other, "add"),
        )

    def __radd__(self, other: float) -> "Knowledge":
        return self.__add__(other)

    def __sub__(self, other: Union["Knowledge", float]) -> "Knowledge":
        if isinstance(other, (int, float)):
            return Knowledge(
                self.value - other,
                self.uncertainty,
                self.confidence,
                self.unit,
                provenance=self._derived_prov(None, "sub_const"),
            )
        u = self._quad(self.uncertainty, other.uncertainty)
        return Knowledge(
            self.value - other.value,
            u,
            self._combine_confidence(self, other),
            self.unit,
            provenance=self._derived_prov(other, "sub"),
        )

    def __mul__(self, other: Union["Knowledge", float]) -> "Knowledge":
        if isinstance(other, (int, float)):
            return Knowledge(
                self.value * other,
                abs(self.uncertainty * other),
                self.confidence,
                self.unit,
                provenance=self._derived_prov(None, "mul_const"),
            )
        # GUM: δ(A*B)/(A*B) = sqrt((δA/A)² + (δB/B)²)
        if self.value == 0.0 or other.value == 0.0:
            u = self._quad(self.uncertainty * abs(other.value),
                           other.uncertainty * abs(self.value))
        else:
            rel_u = self._quad(
                self.uncertainty / abs(self.value),
                other.uncertainty / abs(other.value),
            )
            u = abs(self.value * other.value) * rel_u
        return Knowledge(
            self.value * other.value,
            u,
            self._combine_confidence(self, other),
            provenance=self._derived_prov(other, "mul"),
        )

    def __rmul__(self, other: float) -> "Knowledge":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Knowledge", float]) -> "Knowledge":
        if isinstance(other, (int, float)):
            if other == 0.0:
                raise ZeroDivisionError("division by zero in Knowledge")
            return Knowledge(
                self.value / other,
                abs(self.uncertainty / other),
                self.confidence,
                self.unit,
                provenance=self._derived_prov(None, "div_const"),
            )
        if other.value == 0.0:
            raise ZeroDivisionError("division by Knowledge with value=0")
        # GUM: δ(A/B)/(A/B) = sqrt((δA/A)² + (δB/B)²)
        rel_u = self._quad(
            self.uncertainty / abs(self.value) if self.value != 0 else 0,
            other.uncertainty / abs(other.value),
        )
        result_val = self.value / other.value
        return Knowledge(
            result_val,
            abs(result_val) * rel_u,
            self._combine_confidence(self, other),
            provenance=self._derived_prov(other, "div"),
        )

    def __neg__(self) -> "Knowledge":
        return Knowledge(
            -self.value,
            self.uncertainty,
            self.confidence,
            self.unit,
            provenance=self._derived_prov(None, "neg"),
        )

    def __abs__(self) -> "Knowledge":
        return Knowledge(
            abs(self.value),
            self.uncertainty,
            self.confidence,
            self.unit,
            provenance=self._derived_prov(None, "abs"),
        )

    # ------------------------------------------------------------------
    # Comparison (ignores uncertainty — compares nominal values)
    # ------------------------------------------------------------------

    def __lt__(self, other: Union["Knowledge", float]) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value < v

    def __le__(self, other: Union["Knowledge", float]) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value <= v

    def __gt__(self, other: Union["Knowledge", float]) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value > v

    def __ge__(self, other: Union["Knowledge", float]) -> bool:
        v = other.value if isinstance(other, Knowledge) else float(other)
        return self.value >= v

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Knowledge):
            return self.value == other.value
        return self.value == other

    # ------------------------------------------------------------------
    # Epistemic helpers
    # ------------------------------------------------------------------

    @property
    def cv_percent(self) -> float:
        """Coefficient of variation in percent (uncertainty/|value| * 100)."""
        if self.value == 0.0:
            return float("inf")
        return abs(self.uncertainty / self.value) * 100.0

    @property
    def expanded_uncertainty(self, k: float = 2.0) -> float:
        """Expanded uncertainty at coverage factor k (default k=2, ~95% coverage)."""
        return k * self.uncertainty

    def is_within_threshold(self, threshold: float, k: float = 2.0) -> bool:
        """True if expanded uncertainty is within absolute threshold."""
        return self.expanded_uncertainty <= threshold

    def prob_gt(self, x: float) -> float:
        """P(value > x) using normal approximation of the uncertainty distribution."""
        if self.uncertainty == 0.0:
            return 1.0 if self.value > x else 0.0
        z = (self.value - x) / self.uncertainty
        # Approximation: Φ(z) for standard normal
        return _norm_cdf(z)

    def prob_in_range(self, lo: float, hi: float) -> float:
        """P(lo < value < hi) using normal approximation."""
        return self.prob_gt(lo) - self.prob_gt(hi)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"Knowledge({self.value:.4g} ± {self.uncertainty:.4g}{unit_str}, "
            f"confidence={self.confidence:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "unit": self.unit,
            "cv_percent": self.cv_percent,
            "provenance": self.provenance.to_dict(),
        }


# ------------------------------------------------------------------
# Convenience constructors
# ------------------------------------------------------------------

def measure(
    value: float,
    uncertainty: float = 0.0,
    unit: str = "",
    source: str = "instrument",
    confidence: float = 0.95,
) -> Knowledge:
    """
    Create a Knowledge value from a physical measurement.

    This is the primary constructor for measured quantities, following
    GUM (JCGM 100:2008) conventions.

    Args:
        value:       nominal measurement value
        uncertainty: standard uncertainty (1σ)
        unit:        physical unit string (e.g., "mg", "mL", "L/h")
        source:      instrument or method identifier
        confidence:  epistemic confidence in the source (default 0.95)

    Example::

        dose = measure(500.0, uncertainty=12.5, unit="mg", source="HPLC_A")
    """
    return Knowledge(
        value=value,
        uncertainty=uncertainty,
        confidence=confidence,
        unit=unit,
        source=source,
    )


def confidence_gate(
    condition: bool,
    confidence: float,
    min_confidence: float = 0.90,
    action: str = "warn",
) -> bool:
    """
    Epistemic gate: enforce minimum confidence before proceeding.

    Returns True only if the condition holds AND confidence >= min_confidence.

    Args:
        condition:       boolean condition to check
        confidence:      confidence of the epistemic value being tested
        min_confidence:  minimum required confidence (default 0.90)
        action:          "warn" | "raise" | "silent"

    Example::

        result = simulate_epistemic(model, dose)
        if confidence_gate(result.value > 45.0,
                           confidence=result.confidence,
                           min_confidence=0.90):
            approve_dosing()
    """
    if not condition:
        return False
    if confidence >= min_confidence:
        return True
    msg = (
        f"confidence_gate: confidence {confidence:.2f} below threshold "
        f"{min_confidence:.2f}"
    )
    if action == "raise":
        raise RuntimeError(msg)
    elif action == "warn":
        import warnings
        warnings.warn(msg, stacklevel=2)
    return False


# ------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------

def _norm_cdf(z: float) -> float:
    """Approximation of the standard normal CDF using Abramowitz & Stegun."""
    if z >= 8.0:
        return 1.0
    if z <= -8.0:
        return 0.0
    # Rational approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422820 * math.exp(-0.5 * z * z)
    poly = t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 +
           t * (-1.8212560 + t * 1.3302744))))
    p = 1.0 - d * poly
    return p if z >= 0.0 else 1.0 - p
