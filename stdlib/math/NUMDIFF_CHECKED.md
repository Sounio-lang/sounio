# Checked numerical differentiation (SIO-164)

`math::numdiff_checked` exposes a reusable scalar derivative and dynamic gradient, Jacobian and Hessian kernels. Inputs, samples, callback values, staging, outputs and callback context remain caller-owned. The library `_into` bodies contain no heap allocation or result construction. Callback effects and callback numerical accuracy remain the caller's responsibility.

The numerical method uses local three-node quadratic formulas at represented floating-point nodes. For nonuniform finite-difference background see [Fornberg (1988), *Generation of Finite Difference Formulas on Arbitrarily Spaced Grids*](https://www.ams.org/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf). Refinement-based error estimation is also described in the [GSL numerical differentiation documentation](https://www.gnu.org/software/gsl/doc/html/diff.html). This implementation's formulas, contracts and limitations are specified below.

## Scalar API and callback accounting

Create `DiffControls` with `diff_controls(atol, rtol, max_callbacks)` and a reusable `DiffResult` with `diff_result()`. Then call:

```sounio
derivative_into(callback, x, h, max_levels, &controls, &!context, &!result)
```

The scalar callback has type `fn(f64, &!NumericBuffer) -> f64 with Mut, Div, Panic`. Its context can contain model parameters, counters or other caller state. For a derivative interpretation, repeated evaluations must represent the same mathematical function at the same x. Context can update counters or consistent caches; changing the model between sample evaluations is outside that interpretation. The callback is invoked once at the center and twice for each attempted level. **The counter increments immediately before every actual callback, including a callback returning NaN/Inf.** A disallowed call after budget exhaustion is not made or counted. Context changes made by executed callbacks are not rolled back.

`h` must be finite and positive, `max_levels >= 2`, and the callback budget must permit at least five calls. Both the initial `h` and `h/2` node pairs must be finite and distinct from `x`. The center value is reused. Each subsequent level halves the nominal spacing until the requested criterion holds, the level limit is reached, the budget is exhausted, or represented nodes collapse.

On success, `DiffResult` contains `estimate`, `error_estimate`, `step`, `left_step`, `right_step`, `callback_count`, `status=0`, `error_available=true` and `converged=true`. The two step fields report actual represented separations from `x`; `step` is their maximum, rather than a potentially misleading nominal spacing. Failure clears scalar numerical payloads and availability/convergence flags while preserving the number of callbacks actually made. Reuse resets every field at the beginning of a new request.

## Represented stencils and the meaning of error

For `xm = x-h`, `xp = x+h`, let `hm = x-xm`, `hp = xp-x`, `sl = (f0-fm)/hm`, and `sr = (fp-f0)/hp`. The first derivative is the weighted combination `hp/(hm+hp)*sl + hm/(hm+hp)*sr`. Weights are evaluated after normalizing both spacings by their maximum, avoiding overflow of their sum. The second derivative similarly uses `2*(sr-sl)/(hm+hp)` with normalized spacings and reordered division.

This uses actual floating-point geometry and differences relative to the center value, rather than weighted large raw function values or an assumed exactly symmetric denominator. Intermediate arithmetic can still overflow or lose information even when another formulation could represent a mathematical answer; this implementation explicitly rejects nonfinite arithmetic. Finite underflow, cancellation and callback rounding remain limitations.

The reported error is the **absolute difference between consecutive estimates**, not a rigorous truncation/roundoff bound or statistical uncertainty. `converged` means only that this discrepancy satisfied `error <= atol + rtol*abs(estimate)`. Constant or rounded callback outputs can agree even when sampling cannot resolve the true derivative. The library does not assess differentiability, smoothness, determinism, conditioning or model suitability. Callback Panic propagates; it is not converted into an ordinary returned status.

## Dynamic gradient, Jacobian and Hessian

The vector callback has type:

```sounio
fn(&NumericBuffer, i64, &!NumericBuffer, i64, &!NumericBuffer) -> i64
    with Mut, Div, Panic
```

Its arguments are sample, active input count, values, active output count and context. It returns 0 and fills every active callback-output element. Before **each** call the kernel poisons that prefix with NaN, so partial writes cannot silently reuse a previous finite value. Nonzero callback status is preserved in `DiffSummary.callback_status`; nonfinite/partial output is rejected even when the callback returned 0.

Use `diff_summary()` once and reuse it. The APIs are `gradient_into`, `jacobian_into`, and `hessian_into`. Their full signatures are in [numdiff_checked.sio](numdiff_checked.sio); the [executable caller](../../tests/run-pass/math_numdiff_checked.sio) supplies every buffer. They use exactly two step levels, `h[i]` and `h[i]/2`. They do not silently perform additional refinement. Every entry must satisfy the requested discrepancy criterion before any public output is committed.

| Operation | Shape | Caller sample / callback values | Caller staging length | Complete callback count |
|---|---|---|---|---|
| Gradient | n entries, scalar callback output | n / 1 | 4 + 2n | 1 + 4n |
| Jacobian | row-major m by n | n / m | 4m + 2nm | 1 + 4n |
| Hessian | symmetric row-major n by n, scalar callback output | n / 1 | 9 + 2n² | 1 + 4n + 16n(n-1)/2 |

`x` and `steps` need at least n elements. Value and error outputs need the indicated number of entries; suffixes are preserved. Sample, callback values, work, value output, error output and context are distinct buffer owners. The Hessian uses tensor products of local three-node formulas, including center, axis and corner samples. Symmetric entries are assigned from the same computed mixed derivative; symmetry is a construction policy, not an assessment of a function's smoothness.

Positive dimensions, integer products, storage statuses/capacities, finite active inputs/steps, and both represented step levels are globally checked before any callback or scratch mutation. Coordinatewise validation covers the mixed Hessian sample vectors as well. Work-size and callback-budget products are checked before multiplication. There is no fixed 64-variable or Mat8 ceiling. Required payload and callback work still grow with the stated dimensions.

A failed callback or numerical discrepancy can change scratch and context, but leaves **all public value and error outputs unchanged**, including entries staged earlier in the request. `DiffSummary.status` and `converged` describe the new attempt; existing output contents must not be presented as new results after a failure.

Each derivative entry and its error have their own units: output/input for a Jacobian, and scalar-output/(input_i*input_j) for a Hessian. `rtol` is dimensionless. The scalar `atol` is applied numerically to every entry; it is not a universal physical tolerance across heterogeneous units. Callers must choose meaningful scaling/tolerances for their model. No automatic unit conversion, confidence interval or covariance is produced.

## Status and preflight policy

| Status | Meaning |
|---|---|
| 0 | Completed computation and requested discrepancy criterion satisfied |
| 1 | Invalid finite controls, x/h, dimension or scalar level count |
| 2 | Unavailable/short storage or unrepresentable required storage product |
| 3 | Insufficient callback budget |
| 4 | Nonfinite/collapsed represented stencil geometry |
| 5 | Nonfinite scalar callback, vector callback failure, or nonfinite/partial vector output |
| 6 | Nonfinite derivative, discrepancy or tolerance arithmetic |
| 7 | Requested discrepancy criterion not reached |

All returned statuses also appear in the reused result/summary. Scalar failure provides no stale best-estimate payload. Dynamic failure commits no numerical prefix. Active-prefix validation ignores unused suffixes; callbacks must respect the supplied logical counts. A returned computational result does not certify the scientific interpretation of the callback.

## Legacy repairs

The existing private `math/numdiff.sio` signatures remain private. Its square-root helper delegates to public `math::pure::sqrt`; its cube-root helper range-reduces finite magnitudes to [1,8) before bounded Newton steps and rescales by powers of two. Nonfinite root inputs are explicitly rejected. This replaces inaccurate fixed unscaled iterations, including the small epsilon root used by legacy step-size heuristics.

Legacy finite-difference calls assert positive representable step geometry and applicable finite denominators before invoking callbacks. Fixed-array gradient/Jacobian/Hessian/Laplacian paths assert dimensions and active input geometry before reads or writes; empty supported shapes return without callbacks. Invalid void/scalar requests use their existing Panic effect. Other legacy callback/arithmetic behavior does not inherit the public facade's transactional or checked-result contract. The legacy Richardson comment now correctly calls its discrepancy an estimate rather than a bound.

## Replay and execution scope

Run the focused gate on an authorized native Linux worker with an explicit raw Madaros executable and fresh absolute output:

```sh
MADAROS_RAW_BIN=/absolute/selected/madaros.elf \
NUMDIFF_SEED_BIN=/optional/shipped/lean-single.elf \
  bash scripts/stdlib/math_numdiff_checked_gate.sh /absolute/fresh/results \
  /optional/frozen/pre-repair-numdiff.sio
```

The canonical resolver selects the supplied raw compiler directly, avoiding unused prebuilt materialization or bundled compiler copies. The optional baseline exercises actual before/after controls. Optional seed checking uses the seed's raw SRC OUT ABI, compiles the complete executable caller/import closure and does not execute the generated seed ELF.

### Executed acceptance

Base main: `a2a44df195bb8caf3fb07cbc9517275b29325e94`. Final Slurm **17156** completed with gate **STATUS 0**, reusing the explicitly selected source-built Madaros SHA-256 `d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`. Library check, public caller check/compile/native, C++23 build/concordance, seven legacy root values, independent nonconvergence control, shipped-seed caller compilation and input integrity passed.

C++23 independently checks **10 analytic scalar cases** (polynomial, exponential, sine, identity and constant callbacks), **5253 actual matrix entries** (gradient257, Jacobian3x257, Hessian65x65), actual callback counts and completeness, seven root/epsilon-step values, and the one-million-call checksum. Inputs include tiny, huge, adjacent and subnormal sample cases. Failure controls exercise invalid steps and callback budgets, node collapse, nonfinite input/callbacks, short work storage, overflowing requested shape products, actual callback budget exhaustion, arithmetic overflow of a tolerance, and stale-result clearing.

Callbacks returning a nonzero status and callbacks leaving part of their output unwritten are tested on the **ninth** actual callback, after an earlier column or diagonal was already staged. No public prefix is committed. A separate quartic control makes gradient/Jacobian/Hessian all return status7 after five actual callbacks under zero discrepancy tolerance; C++ independently checks the positive paired-step discrepancies and the reported refusals. Wrong-value and missing-completion copies are rejected with exits11 and13.

The unchanged legacy root helper executable exits **1**, while corrected roots execute successfully and agree with C++. Legacy negative h reaches its forbidden continuation and exits **42** before the repair; the corrected helper aborts **1**. Corrected invalid fixed-array count also aborts1. These raw negative statuses are intentionally preserved separately from the gate's acceptance rows.

The full public native program includes **1,000,000 reused scalar calls** plus the wide vector/matrix checks; its one GNU-time observation was **2.34 seconds, maximum RSS1052 KiB**. This is a whole-program observation, not an isolated kernel benchmark, comparison, general memory bound or proof about arbitrary callback allocation. Library hot bodies contain no record construction or heap calls. The shipped seed SHA-256 `53b64a871490a6daf64fb2e56d62482c590b00d9f0ec94f0d621cefc2b341fd2` compiled the main caller/import closure successfully; its generated ELF was not executed. C++ reference build used GCC13.3.0 with `-std=c++23 -O2 -Wall -Wextra -Werror`.

[Compact receipts](../../tests/stdlib/numdiff/evidence/receipts.tar.gz) preserve exact inputs/compiler/executable identities, real stdout, resource records, controls and historical failed attempts. Archive SHA-256: `fe80904b16fd1874ee4b3e338fed2254d62196d6153274045a560805e7b7a7c6`, **246398 bytes**. It contains no ELF, full source tree, compiler payload or cache. Identical receipt files use internal tar hard links. One bounded baseline module is retained as replay text; omitted executable/generated-source hashes are provenance, while `MANIFEST.sha256` verifies the actual extracted files.

`SCOPE.txt` preserves the initial unit/i64 branch-type error and raw-ELF missing executable mode, plus the targeted fixture's rejected simultaneous context read/write. The fixture correction reads the count before the exclusive write; checker and numerical conditions were preserved. Final17156 source identities, rather than historical differing fixtures, govern acceptance. Original worker snapshots/archives remain separate and unmodified.

```sh
sha256sum -c tests/stdlib/numdiff/evidence/ARCHIVE.sha256
mkdir /tmp/sounio-numdiff-receipts
tar -xzf tests/stdlib/numdiff/evidence/receipts.tar.gz -C /tmp/sounio-numdiff-receipts
(cd /tmp/sounio-numdiff-receipts && sha256sum -c MANIFEST.sha256)
```

This increment does not claim a current-main compiler rebuild, full-corpus CI, release, merge or human acceptance.
