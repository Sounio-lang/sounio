"""
sounio._epistemic — GUM propagation utilities and epistemic result types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from sounio._knowledge import Knowledge, measure


@dataclass
class EpistemicResult:
    """
    A result from an epistemic computation — wraps multiple Knowledge values
    with correlation information and a summary score.
    """
    values: dict[str, Knowledge]
    correlation_matrix: Optional[list[list[float]]] = None
    epistemic_score: float = 0.0
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def get(self, key: str) -> Knowledge:
        if key not in self.values:
            raise KeyError(f"EpistemicResult has no field '{key}'")
        return self.values[key]

    def confidence_summary(self) -> float:
        """Minimum confidence across all result values."""
        if not self.values:
            return 0.0
        return min(v.confidence for v in self.values.values())

    def provenance_summary(self) -> str:
        sources = []
        for k, v in self.values.items():
            sources.append(f"{k}: {v.provenance.source}")
        return " | ".join(sources)

    def to_dict(self) -> dict:
        return {
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "epistemic_score": self.epistemic_score,
            "confidence_summary": self.confidence_summary(),
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{k}={v.value:.4g}±{v.uncertainty:.4g}"
            for k, v in self.values.items()
        )
        return f"EpistemicResult({fields}, score={self.epistemic_score:.2f})"


class GUMPropagation:
    """
    GUM (JCGM 100:2008) first-order uncertainty propagation.

    Provides numerical differentiation-based propagation for arbitrary
    functions of Knowledge inputs.

    Example::

        gum = GUMPropagation()
        result = gum.propagate(
            fn=lambda x, y: x * y / (x + y),  # parallel resistance
            inputs={"R1": measure(100, 2), "R2": measure(200, 5)},
        )
    """

    def __init__(self, step_factor: float = 1e-5):
        """
        Args:
            step_factor: relative step size for numerical differentiation
                         (h = step_factor * |x| + 1e-10)
        """
        self.step_factor = step_factor

    def _numerical_gradient(
        self,
        fn: Callable,
        inputs: Sequence[float],
        idx: int,
        h: float,
    ) -> float:
        """Central-difference approximation of ∂fn/∂x_i."""
        inputs_fwd = list(inputs)
        inputs_bwd = list(inputs)
        inputs_fwd[idx] += h
        inputs_bwd[idx] -= h
        return (fn(*inputs_fwd) - fn(*inputs_bwd)) / (2.0 * h)

    def propagate(
        self,
        fn: Callable,
        inputs: dict[str, Knowledge],
        output_unit: str = "",
        output_source: str = "GUM-propagated",
    ) -> Knowledge:
        """
        Propagate uncertainty through fn via GUM first-order method.

        Args:
            fn:           function accepting the nominal values as positional args
            inputs:       dict of named Knowledge inputs
            output_unit:  unit for the result
            output_source: provenance source label

        Returns:
            Knowledge value with propagated uncertainty.
        """
        names = list(inputs.keys())
        knowls = [inputs[n] for n in names]
        nominals = [k.value for k in knowls]
        uncertainties = [k.uncertainty for k in knowls]

        nominal_result = fn(*nominals)

        # GUM: u²(y) = Σ_i (∂y/∂x_i)² * u²(x_i)
        variance = 0.0
        for i, (nom, unc) in enumerate(zip(nominals, uncertainties)):
            h = self.step_factor * abs(nom) + 1e-10
            grad = self._numerical_gradient(fn, nominals, i, h)
            variance += (grad * unc) ** 2

        u_combined = math.sqrt(variance)
        conf = min(k.confidence for k in knowls)

        from sounio._knowledge import Provenance
        return Knowledge(
            value=nominal_result,
            uncertainty=u_combined,
            confidence=conf,
            unit=output_unit,
            provenance=Provenance(
                source=output_source,
                method=f"GUM-first-order({fn.__name__ if hasattr(fn, '__name__') else 'fn'})",
                derived_from=[k.provenance.source for k in knowls],
            ),
        )

    def sensitivity_coefficients(
        self,
        fn: Callable,
        inputs: dict[str, Knowledge],
    ) -> dict[str, float]:
        """
        Compute sensitivity coefficients (∂y/∂x_i) for each input.

        Useful for understanding which input dominates the uncertainty.
        """
        names = list(inputs.keys())
        nominals = [inputs[n].value for n in names]
        coefficients = {}
        for i, name in enumerate(names):
            h = self.step_factor * abs(nominals[i]) + 1e-10
            grad = self._numerical_gradient(fn, nominals, i, h)
            coefficients[name] = grad
        return coefficients

    def uncertainty_budget(
        self,
        fn: Callable,
        inputs: dict[str, Knowledge],
    ) -> dict[str, float]:
        """
        Compute the uncertainty budget: contribution (%) of each input
        to the total output uncertainty.
        """
        names = list(inputs.keys())
        nominals = [inputs[n].value for n in names]
        uncertainties = [inputs[n].uncertainty for n in names]

        contributions = {}
        total_var = 0.0
        for i, name in enumerate(names):
            h = self.step_factor * abs(nominals[i]) + 1e-10
            grad = self._numerical_gradient(fn, nominals, i, h)
            var_i = (grad * uncertainties[i]) ** 2
            contributions[name] = var_i
            total_var += var_i

        if total_var == 0.0:
            return {n: 0.0 for n in names}
        return {n: v / total_var * 100.0 for n, v in contributions.items()}
