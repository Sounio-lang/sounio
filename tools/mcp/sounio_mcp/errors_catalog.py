"""Compiler error catalog served as MCP resources."""
# ruff: noqa: E501

from __future__ import annotations

import re
from dataclasses import dataclass

ERROR_CODE_RE = re.compile(r"^[E][0-9]{3,5}$")


@dataclass(frozen=True)
class ErrorEntry:
    title: str
    explanation: str
    fixes: tuple[str, ...]
    traces: tuple[str, ...]


ERROR_CATALOG: dict[str, ErrorEntry] = {
    "E001": ErrorEntry(
        "Generic type mismatch",
        "The checker found a value whose inferred type does not match the declared or expected type.",
        ("Check function argument order and annotations.", "Prefer explicit Sounio types on boundary values."),
        ("self-hosted/compiler/lean_single.sio:12003", "tests/compile-fail/annotation_axiom_mismatch.sio:2"),
    ),
    "E003": ErrorEntry(
        "Assignment to immutable binding",
        "`let` bindings are immutable. Reassignment requires `var`.",
        ("Use `var name = value` for mutable locals.", "Keep `let` for single-assignment values."),
        ("tests/ui/type/assign_to_immut.sio:3",),
    ),
    "E004": ErrorEntry(
        "Binary or comparison type mismatch",
        "The operands of a binary/comparison operation are not compatible.",
        ("Align operand types before the operation.", "Use explicit conversions only where Sounio supports them."),
        ("tests/ui/type/invalid_binary_op.sio:3", "tests/ui/type/comparison_type_mismatch.sio:3"),
    ),
    "E005": ErrorEntry(
        "Invalid unary/logical operation",
        "A unary operator was applied to a value of the wrong type.",
        ("Apply logical negation to bool values only.", "Use `0 - x` rather than unary minus where required."),
        ("tests/ui/type/invalid_unary_op.sio:3", "tests/ui/type/logical_not_bool.sio:3"),
    ),
    "E006": ErrorEntry(
        "Condition is not bool",
        "`if` and `while` conditions must be boolean.",
        ("Compare numeric values explicitly.", "Do not rely on truthiness."),
        ("tests/ui/type/condition_not_bool.sio:3", "tests/ui/type/while_cond_not_bool.sio:3"),
    ),
    "E007": ErrorEntry(
        "If branch mismatch",
        "The two branches of an expression-valued `if` produce incompatible types.",
        ("Return the same type from both branches.", "Use an explicit unit-valued block for side-effect branches."),
        ("tests/ui/type/if_branch_mismatch.sio:3",),
    ),
    "E013": ErrorEntry(
        "Value is not indexable",
        "Index syntax was used on a value that is not an array/vector-like type.",
        ("Check the receiver type.", "Use field access for structs and indexing for arrays."),
        ("tests/ui/type/not_indexable.sio:3",),
    ),
    "E014": ErrorEntry(
        "Invalid array index type",
        "Array indexing received a non-integer or otherwise unsupported index.",
        ("Use an integer index.", "Keep index arithmetic typed as an integer."),
        ("tests/ui/type/array_index_type.sio:3",),
    ),
    "E017": ErrorEntry(
        "Not callable or callable mismatch",
        "The checker tried to call a non-function value or a value with an incompatible function type.",
        ("Check the callee name.", "Define helper functions before callers."),
        ("tests/ui/type/not_callable.sio:3", "tools/lsp/test_smoke.sh:63"),
    ),
    "E018": ErrorEntry(
        "Match arm mismatch",
        "`match` arms do not agree on the result type.",
        ("Make each arm return the same type.", "Use explicit unit arms for side-effect-only matches."),
        ("tests/ui/type/match_arm_mismatch.sio:4",),
    ),
    "E020": ErrorEntry(
        "Array element type mismatch",
        "An array literal or assignment contains an element of the wrong type.",
        ("Use homogeneous element types.", "Add an explicit array type to guide inference."),
        ("tests/ui/type/array_elem_mismatch.sio:3",),
    ),
    "E035": ErrorEntry(
        "Recursive type rejected",
        "The checker rejected a recursive type shape.",
        ("Break recursion through an explicit indirection once supported.", "Use a bounded non-recursive representation."),
        ("self-hosted/check/check.sio:4187",),
    ),
    "E040": ErrorEntry(
        "Linear value not consumed",
        "A linear value reached scope exit without being consumed exactly once.",
        ("Pass the linear value to its consuming operation.", "Avoid copying or dropping linear handles."),
        ("self-hosted/check/check.sio:10086",),
    ),
    "E041": ErrorEntry(
        "Unit mismatch",
        "Dimensional units are incompatible in arithmetic or a function call.",
        ("Convert units explicitly.", "Keep API annotations in the same unit family."),
        ("self-hosted/check/check.sio:10721", "self-hosted/check/check.sio:10742"),
    ),
    "E042": ErrorEntry(
        "Refinement violation",
        "A value does not satisfy a refinement predicate.",
        ("Prove or guard the predicate before the boundary.", "Pass a literal/value inside the accepted bounds."),
        ("self-hosted/check/check.sio:4115", "tests/ui/type/refinement_violation.sio:3"),
    ),
    "E044": ErrorEntry(
        "Break outside loop",
        "`break` appeared outside a loop body.",
        ("Move `break` inside a `while` or `for`.", "Use `return` for function exit."),
        ("tests/ui/type/break_outside_loop.sio:3",),
    ),
    "E045": ErrorEntry(
        "Continue outside loop",
        "`continue` appeared outside a loop body.",
        ("Move `continue` inside a loop.", "Use conditional control flow outside loops."),
        ("tests/ui/type/continue_outside_loop.sio:3",),
    ),
    "E048": ErrorEntry(
        "Bitwise op on float",
        "A bitwise operator was applied to a floating-point value.",
        ("Use integer operands for bitwise operations.", "Use arithmetic operations for floats."),
        ("tests/ui/type/bitwise_float.sio:3",),
    ),
    "E049": ErrorEntry(
        "Shift op on float",
        "A shift operator was applied to a floating-point value.",
        ("Shift integer values only.", "Convert through an approved integer representation first."),
        ("tests/ui/type/shift_float.sio:3",),
    ),
    "E050": ErrorEntry(
        "Modulo on float",
        "Modulo was applied to a floating-point value.",
        ("Use integer modulo.", "Implement float remainder through a checked math helper if needed."),
        ("tests/ui/type/modulo_float.sio:3",),
    ),
    "E056": ErrorEntry(
        "Division by zero",
        "The checker detected literal or const-tracked division by zero.",
        ("Guard denominators with a refinement or explicit check.", "Avoid const expressions that reduce to zero."),
        ("tests/ui/type/division_by_zero.sio:3", "tests/ui/type/division_by_zero_const_expr.sio:3"),
    ),
    "E057": ErrorEntry(
        "Modulo by zero",
        "The checker detected literal or const-tracked modulo by zero.",
        ("Guard modulo denominators.", "Use a refinement type for non-zero divisors."),
        ("tests/ui/type/modulo_by_zero.sio:3", "tests/ui/type/modulo_by_zero_const_expr.sio:3"),
    ),
    "E067": ErrorEntry(
        "Causal non-identifiability/confounding",
        "A causal query is not identifiable under the encoded graph assumptions.",
        ("Add a valid adjustment set.", "Use front-door/IV structure only when its assumptions are encoded."),
        ("self-hosted/compiler/lean_single.sio:12092", "tests/compile-fail/causal_no_adjustment.sio:2"),
    ),
    "E070": ErrorEntry(
        "Kernel or singleton constraint violation",
        "A kernel declared an invalid effect, or a singleton/epistemic constraint was violated.",
        ("Remove IO/Mut/Alloc from kernel functions.", "Keep kernel and singleton declarations within their constraints."),
        ("self-hosted/check/check.sio:2625", "self-hosted/compiler/lean_single.sio:23514"),
    ),
    "E071": ErrorEntry(
        "Effect scope leak",
        "An effect is used outside the function or scoped effect row that declares it.",
        ("Add the required `with` effect to the function.", "Push side effects to the boundary."),
        ("self-hosted/check/check.sio:2658", "tests/compile-fail/scoped_effect_escape.sio:2"),
    ),
    "E072": ErrorEntry(
        "Kernel return or causal independence violation",
        "The checker rejected a kernel return type or a causal independence claim.",
        ("Kernel functions should return unit.", "For causal claims, encode the graph conditions that justify independence."),
        ("self-hosted/check/check.sio:2537", "self-hosted/compiler/lean_single.sio:3510"),
    ),
    "E073": ErrorEntry(
        "Potential outcome token consumed twice",
        "A linear potential-outcome token was used after consumption.",
        ("Choose one potential outcome branch per unit.", "Thread linear tokens exactly once."),
        ("self-hosted/check/check.sio:2571", "tests/compile-fail/potential_outcome_both_observed.sio:2"),
    ),
    "E170": ErrorEntry(
        "Knowledge value access needs Epistemic effect",
        "Accessing `.value` on `Knowledge<T>` crosses an epistemic boundary.",
        ("Add `with Epistemic` where the unwrap is justified.", "Prefer propagating `Knowledge<T>` without unwrapping."),
        ("self-hosted/compiler/lean_single.sio:3518",),
    ),
    "E171": ErrorEntry(
        "Knowledge cast rejected",
        "Casting an epistemic value directly to its inner type is forbidden.",
        ("Use an audited epistemic unwrap path.", "Carry `Knowledge<T>` through computations."),
        ("self-hosted/compiler/lean_single.sio:3525",),
    ),
    "E200": ErrorEntry(
        "Unknown identifier",
        "Name resolution could not find an identifier.",
        ("Check spelling and imports.", "Define helpers before their callers."),
        ("self-hosted/compiler/lean_single.sio:3448", "tests/compile-fail/observe_transition_requires_plan.sio:2"),
    ),
    "E201": ErrorEntry(
        "ExactlyPrivate<T> requires ZD",
        "The privacy capability requires the `ZD` effect.",
        ("Add `with ZD` when the operation is justified.", "Move capability-bearing parameters to a ZD-aware boundary."),
        ("self-hosted/compiler/lean_single.sio:23296", "tests/parity/golden/exactly_private_requires_zd.stderr:15"),
    ),
    "E202": ErrorEntry(
        "Editable<T> requires ZD",
        "`Editable<T>` parameters require the `ZD` effect.",
        ("Declare `with ZD`.", "Use non-editable data at pure boundaries."),
        ("self-hosted/compiler/lean_single.sio:23302", "tests/parity/golden/editable_requires_zd.stderr:15"),
    ),
    "E203": ErrorEntry(
        "CapabilityGated<T> requires ZD",
        "`CapabilityGated<T>` parameters require the `ZD` effect.",
        ("Declare `with ZD`.", "Separate gated-capability logic from pure helpers."),
        ("self-hosted/compiler/lean_single.sio:23308", "tests/parity/golden/capability_gated_requires_zd.stderr:15"),
    ),
    "E204": ErrorEntry(
        "Composable<T> requires ZD",
        "`Composable<T>` parameters require the `ZD` effect.",
        ("Declare `with ZD`.", "Keep compositional capability work in a ZD effect row."),
        ("self-hosted/compiler/lean_single.sio:23314", "tests/parity/golden/composable_requires_zd.stderr:15"),
    ),
    "E205": ErrorEntry(
        "Audited<T> requires ZD and Witness",
        "`Audited<T>` parameters require both `ZD` and `Witness` effects.",
        ("Declare `with ZD, Witness`.", "Attach a witness before crossing the audited boundary."),
        ("self-hosted/compiler/lean_single.sio:23321", "tests/parity/golden/audited_requires_witness.stderr:15"),
    ),
    "E206": ErrorEntry(
        "Revivable<T> requires ZD and Temporal",
        "`Revivable<T>` parameters require `ZD` and `Temporal` effects.",
        ("Declare `with ZD, Temporal`.", "Keep temporal revival logic explicit."),
        ("self-hosted/compiler/lean_single.sio:23329", "tests/parity/golden/revivable_requires_temporal.stderr:15"),
    ),
    "E207": ErrorEntry(
        "Interpretable<T> requires ZD",
        "`Interpretable<T>` parameters require the `ZD` effect.",
        ("Declare `with ZD`.", "Move interpretable capability work to a ZD-aware function."),
        ("self-hosted/compiler/lean_single.sio:23336",),
    ),
}


def get_error_doc(error_code: str) -> str:
    """Return a Markdown explanation for a Sounio compiler error code."""

    code = error_code.upper()
    if not ERROR_CODE_RE.fullmatch(code):
        return "# Sounio Error Catalog\n\nInvalid error code. Use a code such as `E070`."
    entry = ERROR_CATALOG.get(code)
    if entry is None:
        known = ", ".join(sorted(ERROR_CATALOG)[:40])
        return (
            f"# {code}\n\n"
            "No catalog entry is promoted for this code yet. The catalog only lists codes "
            "traced to current compiler source or committed compiler-failure fixtures.\n\n"
            f"Known entries include: {known}"
        )
    fixes = "\n".join(f"- {fix}" for fix in entry.fixes)
    traces = "\n".join(f"- `{trace}`" for trace in entry.traces)
    return (
        f"# {code}: {entry.title}\n\n"
        f"{entry.explanation}\n\n"
        "## Fix Suggestions\n\n"
        f"{fixes}\n\n"
        "## Provenance\n\n"
        "This entry is traced to current repository source or committed compiler-failure evidence:\n\n"
        f"{traces}\n"
    )
