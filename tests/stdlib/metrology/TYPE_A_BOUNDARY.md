# Type-A numeric and public-result boundary

Semantic-Lane-ID: metrology-type-a-boundary
Owner: codex-repository-separation
Concept-IDs: none (repair of the existing library's documented GUM quantity)
Intent-Preserved: sample standard deviation and uncertainty of the mean are distinct; invalid numeric evaluation is not valid zero uncertainty.
Transformation: correct s/n to s/sqrt(n); bound count before reads; finite/zero guards and range-reduced square root; read-only result accessors.
Types-Changed: none; TypeAEval representation stays private and unchanged.
Effects-Changed: none for existing public functions; new accessors declare Panic.
IR-Changed: none.
Claims-Introduced: only tested numeric behavior and callable external result access.
Claims-Forbidden: clinical certification, statistical independence inferred from numbers, proof of full f64 accuracy, or repair of other calibration functions.
Assumptions: independent repeated observations; f64 operations; finite representable intermediates.
Write-Set: stdlib/metrology/calibration.sio; stdlib/metrology/mod.sio; this directory; offload log.
Read-Set: package calibration consumer; compiler wrapper; official GUM.
Positive-Witness: symmetric observations, constant observations, small and large representable spread, public getters.
Negative-Witness: invalid count, nonfinite input, intermediate overflow, squared-deviation zero underflow.
Acceptance-Gate: compile and execute test_type_a_boundary.sio with explicit compiler/stdlib selection; no NATIVE_REFUSAL; compare independent decimal expectations.
Integration-Target: core first, then versioned stdlib bundle and sounio-units consumer migration.
Authoritative-Only-If: source/overlay hashes and actual executable results accompany passing CI.

The baseline (69b7fe7546e8) compiled but failed four same-module witnesses on
published Madaros: constant, symmetric, small spread, and n=17. These witnesses
appended a main to unmodified module text to inspect private fields; they were
not public-API evidence.

The standard deviation of the mean is s/sqrt(n), for the stated sampling model:
[JCGM 100:2008, 4.2.2–4.2.3, equations (4) and (5)](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
The source's old comment agreed with this but its calculation divided by n.
This is an intentional numerical correction, not a behavior-preserving refactor.

`valid=false` rejects inputs/intermediates this implementation cannot evaluate.
It does not imply the physical measurement is invalid. Conversely `valid=true`
is not evidence of independent observations or a suitable uncertainty model.
Two-pass summation still has finite-precision cancellation/rounding limitations.
Other functions in calibration.sio, including their Newton loops, remain outside
this repair and must not inherit its validation claim.

Validation: Slurm 12371 completed with exit 0. Four baseline witnesses, the
public-result regression, and an isolated sounio-units calibration consumer
all compiled and executed successfully with no NATIVE_REFUSAL. The compiler
was the published distribution binary; only runtime/stdlib/metrology/{calibration,mod}.sio
were overlaid. calibration.sio SHA-256:
ef6063d4c079e91b6e18d1d0d4aa80be9a6357de92585c9b33a345a3b0711dca.
The regression was renamed with the test_ prefix after execution (identical
source bytes) to match the canonical suite's tests/stdlib/*/test_*.sio discovery.
This is not a new compiler rebuild or a published stdlib release.

Public facade follow-up: Slurm 12379 compiled and executed the complete
regression through `use metrology::mod::{...}`, exercising mod.sio's re-exports
instead of importing calibration.sio directly. The permanent test now uses
this route; its numeric assertions are unchanged. Slurm 12380 separately tried
`use metrology::{...}` and failed module-closure preflight (unresolved=1).
That shorthand is not claimed as supported by this patch.

ADR009 numerical reference: `type_a_reference.cpp` independently evaluates the
same finite Type-A cases with C++23 and Boost cpp_dec_float_100. It computes
symmetric, small-spread and large-spread expected values and asserts the
reference's own algebraic controls before printing. On the development host:
`c++ -std=c++23 -O2 -I/opt/homebrew/include tests/stdlib/metrology/type_a_reference.cpp -o /tmp/type_a_reference && /tmp/type_a_reference`.
This produced `calibration-cpp23-oracle.json` in the task's evidence directory;
C++23 compilation and execution returned zero. The previous decimal script
result is historical only and is excluded from the verification note. The
Sounio regression remains the executable consumer witness. Decimal precision
is finite and does not prove arbitrary-input f64 accuracy.

A subsequent independent math review (Grok 4.5 and Gemini 2.5 Pro) confirmed
the formulas; Grok identified missing controls in the C++ reference. The
reference now asserts zero large-case mean, constant-reading zero uncertainty,
and rejection of 0/1/17 readings. The Sounio test covers nonfinite inputs.
