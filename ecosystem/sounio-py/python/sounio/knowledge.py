"""Pure Python Knowledge class — GUM-based epistemic value with uncertainty propagation.

This module provides a Python-native fallback implementation. When the Rust
extension (sounio._sounio_native) is available it is preferred for performance.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union


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

    Provenance chain tracking
    -------------------------
    Use Knowledge.tracked() to create a Knowledge value with full DAG tracking.
    All arithmetic on tracked values automatically extends the chain.

    >>> a = Knowledge.tracked(500.0, 2.5, "calibration")
    >>> b = Knowledge.tracked(1.2, 0.05, "correction")
    >>> c = a * b
    >>> print(c.chain.summary())
    ProvenanceChain: 3 nodes (2 sources, 1 operations)
    """

    __slots__ = ("value", "epsilon", "provenance", "_chain", "_node_id")

    def __init__(
        self,
        value: float,
        epsilon: float = 0.0,
        provenance: str = "",
    ) -> None:
        self.value = float(value)
        self.epsilon = float(epsilon)
        self.provenance = str(provenance)
        self._chain = None   # Optional[ProvenanceChain]
        self._node_id = None  # Optional[str]

    # ------------------------------------------------------------------
    # Factory: tracked Knowledge with provenance chain
    # ------------------------------------------------------------------

    @classmethod
    def tracked(
        cls,
        value: float,
        epsilon: float = 0.0,
        provenance: str = "",
        **metadata,
    ) -> "Knowledge":
        """Create a Knowledge value with full provenance chain tracking.

        All subsequent arithmetic on this value (and any value derived from it)
        will record an entry in the provenance DAG.

        Parameters
        ----------
        value, epsilon, provenance : same as __init__
        **metadata : extra fields stored in the root ProvenanceNode.
        """
        from .provenance import ProvenanceChain

        k = cls(value, epsilon, provenance)
        chain = ProvenanceChain()
        node = chain.add_source(provenance, value, epsilon, **metadata)
        k._chain = chain
        k._node_id = node.id
        return k

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            result = Knowledge(
                self.value + other.value,
                math.sqrt(self.epsilon**2 + other.epsilon**2),
                f"({self.provenance})+({other.provenance})",
            )
            if self._chain is not None or other._chain is not None:
                from .provenance import _merge_chains
                result._chain, result._node_id = _merge_chains(
                    self._chain, self._node_id,
                    other._chain, other._node_id,
                    label=result.provenance,
                    operation="add",
                    value=result.value,
                    epsilon=result.epsilon,
                )
            return result
        other = float(other)
        return Knowledge(self.value + other, self.epsilon, self.provenance)

    def __radd__(self, other: _Numeric) -> "Knowledge":
        return self.__add__(other)

    def __sub__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            result = Knowledge(
                self.value - other.value,
                math.sqrt(self.epsilon**2 + other.epsilon**2),
                f"({self.provenance})-({other.provenance})",
            )
            if self._chain is not None or other._chain is not None:
                from .provenance import _merge_chains
                result._chain, result._node_id = _merge_chains(
                    self._chain, self._node_id,
                    other._chain, other._node_id,
                    label=result.provenance,
                    operation="sub",
                    value=result.value,
                    epsilon=result.epsilon,
                )
            return result
        other = float(other)
        return Knowledge(self.value - other, self.epsilon, self.provenance)

    def __rsub__(self, other: _Numeric) -> "Knowledge":
        other = float(other)
        return Knowledge(other - self.value, self.epsilon, self.provenance)

    def __mul__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            val = self.value * other.value
            rel_a = self.epsilon / abs(self.value) if self.value != 0.0 else 0.0
            rel_b = other.epsilon / abs(other.value) if other.value != 0.0 else 0.0
            result = Knowledge(
                val,
                abs(val) * math.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})*({other.provenance})",
            )
            if self._chain is not None or other._chain is not None:
                from .provenance import _merge_chains
                result._chain, result._node_id = _merge_chains(
                    self._chain, self._node_id,
                    other._chain, other._node_id,
                    label=result.provenance,
                    operation="mul",
                    value=result.value,
                    epsilon=result.epsilon,
                )
            return result
        factor = float(other)
        result = Knowledge(
            self.value * factor,
            self.epsilon * abs(factor),
            self.provenance,
        )
        if self._chain is not None:
            from .provenance import _unary_chain
            result._chain, result._node_id = _unary_chain(
                self._chain, self._node_id,
                label=f"{self.provenance}*{factor}",
                operation="scale",
                value=result.value,
                epsilon=result.epsilon,
            )
        return result

    def __rmul__(self, other: _Numeric) -> "Knowledge":
        return self.__mul__(other)

    def __truediv__(self, other: _Numeric) -> "Knowledge":
        if isinstance(other, Knowledge):
            if other.value == 0.0:
                raise ZeroDivisionError("Knowledge division by zero")
            val = self.value / other.value
            rel_a = self.epsilon / abs(self.value) if self.value != 0.0 else 0.0
            rel_b = other.epsilon / abs(other.value) if other.value != 0.0 else 0.0
            result = Knowledge(
                val,
                abs(val) * math.sqrt(rel_a**2 + rel_b**2),
                f"({self.provenance})/({other.provenance})",
            )
            if self._chain is not None or other._chain is not None:
                from .provenance import _merge_chains
                result._chain, result._node_id = _merge_chains(
                    self._chain, self._node_id,
                    other._chain, other._node_id,
                    label=result.provenance,
                    operation="div",
                    value=result.value,
                    epsilon=result.epsilon,
                )
            return result
        divisor = float(other)
        if divisor == 0.0:
            raise ZeroDivisionError("Knowledge division by zero scalar")
        result = Knowledge(
            self.value / divisor,
            self.epsilon / abs(divisor),
            self.provenance,
        )
        if self._chain is not None:
            from .provenance import _unary_chain
            result._chain, result._node_id = _unary_chain(
                self._chain, self._node_id,
                label=f"{self.provenance}/{divisor}",
                operation="scale",
                value=result.value,
                epsilon=result.epsilon,
            )
        return result

    def __rtruediv__(self, other: _Numeric) -> "Knowledge":
        if self.value == 0.0:
            raise ZeroDivisionError("Knowledge division by zero")
        numerator = float(other)
        val = numerator / self.value
        rel = self.epsilon / abs(self.value)
        return Knowledge(val, abs(val) * rel, self.provenance)

    def __neg__(self) -> "Knowledge":
        result = Knowledge(-self.value, self.epsilon, f"-({self.provenance})")
        if self._chain is not None:
            from .provenance import _unary_chain
            result._chain, result._node_id = _unary_chain(
                self._chain, self._node_id,
                label=result.provenance, operation="neg",
                value=result.value, epsilon=result.epsilon,
            )
        return result

    def __abs__(self) -> "Knowledge":
        result = Knowledge(abs(self.value), self.epsilon, f"|{self.provenance}|")
        if self._chain is not None:
            from .provenance import _unary_chain
            result._chain, result._node_id = _unary_chain(
                self._chain, self._node_id,
                label=result.provenance, operation="abs",
                value=result.value, epsilon=result.epsilon,
            )
        return result

    # ------------------------------------------------------------------
    # Comparison (by central value)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Provenance chain access
    # ------------------------------------------------------------------

    @property
    def chain(self):
        """The ProvenanceChain for this value, or None if not tracked."""
        return self._chain

    def ancestry(self) -> List:
        """Return all ancestor ProvenanceNodes (empty list if not tracked)."""
        if self._chain is None or self._node_id is None:
            return []
        return self._chain.ancestors_of(self._node_id)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {"value": self.value, "epsilon": self.epsilon, "provenance": self.provenance}
        if self._chain is not None:
            d["chain"] = self._chain.to_dict()
            d["node_id"] = self._node_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Knowledge":
        k = cls(d["value"], d.get("epsilon", 0.0), d.get("provenance", ""))
        if "chain" in d:
            from .provenance import ProvenanceChain
            k._chain = ProvenanceChain.from_dict(d["chain"])
            k._node_id = d.get("node_id")
        return k

    def to_sounio_format(self) -> str:
        """Serialize to canonical Sounio output format.

        Returns
        -------
        str
            Canonical format: Knowledge { value: X epsilon: Y prov: "Z" }
            where X and Y are formatted with 6 significant digits.

        Examples
        --------
        >>> k = Knowledge(500.0, 2.5, "calibration_batch_2026")
        >>> k.to_sounio_format()
        'Knowledge { value: 500 epsilon: 2.5 prov: "calibration_batch_2026" }'
        """
        return (
            f'Knowledge {{ value: {self.value:.6g} epsilon: {self.epsilon:.6g} '
            f'prov: "{self.provenance}" }}'
        )

    @classmethod
    def from_sounio_output(cls, text: str) -> "Knowledge":
        """Parse a single Knowledge value from Sounio output format.

        Parameters
        ----------
        text : str
            Text containing a Knowledge value in the canonical format.

        Returns
        -------
        Knowledge
            Parsed Knowledge instance.

        Raises
        ------
        ValueError
            If the text does not match the expected pattern.

        Examples
        --------
        >>> text = 'Knowledge { value: 500 epsilon: 2.5 prov: "calibration_batch_2026" }'
        >>> k = Knowledge.from_sounio_output(text)
        >>> k.value
        500.0
        >>> k.epsilon
        2.5
        >>> k.provenance
        'calibration_batch_2026'
        """
        import re

        pattern = (
            r'Knowledge \{ value: ([\d.e+-]+) epsilon: ([\d.e+-]+) '
            r'prov: "([^"]*)" \}'
        )
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Cannot parse Knowledge from: {text}")
        return cls(float(match.group(1)), float(match.group(2)), match.group(3))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

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
