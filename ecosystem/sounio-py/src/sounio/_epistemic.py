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

    Two equivalent usage styles:

    Style A — dict (classic)::

        gum = GUMPropagation()
        result = gum.propagate(
            fn=lambda x, y: x * y / (x + y),
            inputs={"R1": measure(100, 2), "R2": measure(200, 5)},
        )

    Style B — add_input (native-API compatible)::

        gum = GUMPropagation()
        gum.add_input("R1", measure(100, 2))
        gum.add_input("R2", measure(200, 5))
        result = gum.propagate(lambda args: args[0] * args[1] / (args[0] + args[1]))
        gum.uncertainty_budget(lambda args: args[0] * args[1] / (args[0] + args[1]))
    """

    def __init__(self, step_factor: float = 1e-5, h: float = None):
        """
        Args:
            step_factor: relative step size for numerical differentiation
            h: absolute step size override (matches Rust API)
        """
        self.step_factor = step_factor
        self._h_override = h
        self._registered_inputs: list[tuple[str, "Knowledge"]] = []

    def add_input(self, name: str, k: "Knowledge") -> None:
        """Register a named input (Style B / native-API)."""
        self._registered_inputs.append((name, k))

    def input_names(self) -> list[str]:
        return [n for n, _ in self._registered_inputs]

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

    def _resolve_inputs(self, inputs_arg) -> tuple[list[str], list["Knowledge"]]:
        """Resolve inputs from either dict (Style A) or registered list (Style B)."""
        if inputs_arg is not None:
            names = list(inputs_arg.keys())
            knowls = [inputs_arg[n] for n in names]
        elif self._registered_inputs:
            names = [n for n, _ in self._registered_inputs]
            knowls = [k for _, k in self._registered_inputs]
        else:
            raise ValueError("No inputs: use add_input() or pass inputs=dict")
        return names, knowls

    def _wrap_fn(self, fn: Callable, style_b: bool) -> Callable:
        """For Style B, fn receives a list; unwrap to positional."""
        if not style_b:
            return fn
        return lambda *args: fn(list(args))

    def propagate(
        self,
        fn: Callable,
        inputs: dict = None,
        output_unit: str = "",
        output_source: str = "GUM-propagated",
    ) -> "Knowledge":
        """
        Propagate uncertainty through fn via GUM first-order method.

        Style A: propagate(fn=lambda x,y: x/y, inputs={"x": k1, "y": k2})
        Style B: propagate(fn=lambda args: args[0]/args[1])  (after add_input)
        """
        style_b = inputs is None and bool(self._registered_inputs)
        names, knowls = self._resolve_inputs(inputs)
        actual_fn = self._wrap_fn(fn, style_b)

        nominals = [k.value for k in knowls]

        nominal_result = actual_fn(*nominals)

        variance = 0.0
        for i, (nom, unc_k) in enumerate(zip(nominals, knowls)):
            step = self._h_override or (self.step_factor * abs(nom) + 1e-10)
            grad = self._numerical_gradient(actual_fn, nominals, i, step)
            variance += (grad * unc_k.uncertainty) ** 2

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
        inputs: dict = None,
    ) -> dict:
        """Compute sensitivity coefficients ∂y/∂x_i (supports both styles)."""
        style_b = inputs is None and bool(self._registered_inputs)
        names, knowls = self._resolve_inputs(inputs)
        actual_fn = self._wrap_fn(fn, style_b)
        nominals = [k.value for k in knowls]
        coefficients = {}
        for i, name in enumerate(names):
            step = self._h_override or (self.step_factor * abs(nominals[i]) + 1e-10)
            grad = self._numerical_gradient(actual_fn, nominals, i, step)
            coefficients[name] = grad
        return coefficients

    def uncertainty_budget(
        self,
        fn: Callable,
        inputs: dict = None,
    ) -> dict:
        """
        Compute and print the uncertainty budget (supports both styles).
        Returns dict of {name: variance_contribution}.
        """
        style_b = inputs is None and bool(self._registered_inputs)
        names, knowls = self._resolve_inputs(inputs)
        actual_fn = self._wrap_fn(fn, style_b)
        nominals = [k.value for k in knowls]

        contributions = {}
        total_var = 0.0
        for i, name in enumerate(names):
            step = self._h_override or (self.step_factor * abs(nominals[i]) + 1e-10)
            grad = self._numerical_gradient(actual_fn, nominals, i, step)
            var_i = (grad * knowls[i].uncertainty) ** 2
            contributions[name] = var_i
            total_var += var_i

        if total_var == 0.0:
            return {n: 0.0 for n in names}
        return {n: v / total_var * 100.0 for n, v in contributions.items()}
