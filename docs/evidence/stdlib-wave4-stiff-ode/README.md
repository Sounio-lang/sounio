# Fixed-case stiff ODE / real Monte Carlo / PBox / GUM / Klib contract

Slurm **19566 completed 0** in 00:00:23, maximum RSS 2451856 KiB.
The public CLI checks and compiles the complete test, its native ELF exits 0
within the unchanged **30-second** deadline, and the independent C++23
oracle accepts every reported observable. This is a focused scientific
regression, not a claim about whole-stdlib acceptance or an ODE solver's
global accuracy.

Source base: root `fe4f8be2c6ec1fa57fbf65ce3100a7d208c57e9a`.
Raw compiler:
`/scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros`,
SHA-256
`8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932`.
There is no compiler or library implementation change in this commit.

## Preserved experiment and corrected interpretation

The original forward Euler equations and operation order remain:

```text
dx = y
dy = mu * (1 - x*x) * y - x
x = x + dt*dx
y = y + dt*dy
```

The configuration is defined once in private typed Sounio helpers and shared
by main and the noncapturing callback:

| Parameter | Value |
|---|---:|
| mu / variance | 5.0 / 0.25 |
| x0 / variance | 0.5 / 0.01 |
| y0 | 0.0 |
| dt / Euler steps | 0.02 / 80 |
| Monte Carlo samples / library seed | 5000 / 12345 |

The test calls the real public `monte_carlo(mu_ep, ode_mc_sample, ...)`.
It neither duplicates that sampler nor substitutes a fixed-kernel API.
A function pointer has no captured environment, so `ode_mc_sample` reads
the same private configuration helpers used by main.

The library sampler is a wrapping 64-bit LCG with multiplier
6364136223846793005 and increment 1442695040888963407. It takes the signed
absolute-value residue modulo 2^30, generates twelve uniforms, and subtracts
six from their sum. This is a CLT approximation, not xoshiro or an exact
Gaussian sampler. The MC varies mu only and retains the library estimator
`sum_sq/n - mean*mean`.

`pb_from_knowledge` preserves the supplied variance but sets both mean
endpoints to the supplied value. It does not create a Knightian ambiguity
gap or a support interval. The PBox branch also retains x0 variance, unlike
the mu-only MC/GUM branches, so their variance values are not expected to
coincide. The seven-point square grid and tuple-returning PBox update remain
executed for all eighty steps. The mean band remains a point for this case.
Its variance propagation is independently checked as an implementation
observable, without claiming that it is an exact nonlinear distributional
variance or an enclosure.

GUM remains the original forward difference at h=0.001 of the full
eighty-step Euler map, with `df*df*mu_var` and the retained twelve-step
Newton width helper. It is a local approximation, not an outer bound.

## Independent numerical results

The C++23 reference is written independently of the Sounio implementations.
It uses defined unsigned wrapping for the LCG and explicit IEEE double
operation order for MC, Euler, the point/variance PBox propagation, readout,
and the retained Newton helpers. These reported values agree **bit for bit**:

| Observable | C++23 result |
|---|---:|
| MC mean | 0.128468279712418448346 |
| MC variance | 0.000784875705777753063241 |
| MC standard-deviation helper | 0.0280156332389213183487 |
| Nominal Euler final y / PBox mean endpoints | 0.136738084547140453795 |
| PBox mean width | 0 |
| PBox propagated variance | 0.209273351294278792123 |
| GUM width | 0.0176026975856058420788 |
| Scaled readout endpoints | 0.143574988774497480648 |

The Klib reference uses independent `long double` standard exp/log for the
three-point CARA operator, epsilon 0.25, lambda 1 and alpha 2 versus alpha 0.
Its nominal/support rewards retain MCmean times 0.04 / 0.01 / 0.08:

| Observable | Independent reference |
|---|---:|
| Smooth CE / pragmatic | 0.0041724304477752517598 |
| Precise CE / pragmatic | 0.00417521909065359951487 |
| Active EFE epistemic contribution | 0 |
| Active EFE total | negative pragmatic |

The CE and EFE absolute tolerances are **2e-12**, fixed in the independent
reference before executing the Sounio candidate. They were not adjusted
after observing Sounio output. Strict smooth-vs-precise ordering is verified
for this selected case, together with bridge identity and the EFE formula.
No general monotonicity theorem or interval-enclosure theorem is claimed.

Every f64 observable is printed as bits through a typed helper that first
binds `f64_to_bits(value)` to an i64 local. The parser requires each known
metric exactly once, rejects missing/duplicate/unknown/nonfinite values,
and checks the unique success marker and the counterexample flags.

## Original failures remain real counterexamples

The unmodified original and the prior API-typed captured version remain in
`tests/fixtures/wave4_stiff_ode/{original,typed_capture_refusal}.sio.in`.
The original checks/compiles 1; the typed captured version checks 0 and
compiles 1 with diagnostic #1542 and no ELF. These compiler refusals are
recorded separately from scientific correctness.

All three original scientific predicates evaluate false:

1. MC mean lies in the point mean band.
2. Knightian mean width exceeds 1.01 times the GUM width.
3. Smooth pragmatic is below precise pragmatic by at least 0.001.

The positive test explicitly emits and asserts the selected-case flags
`0 0 0`, while checking the correct point-lift, dispersion and bridge
identities. It does not skip the original predicates or invent physical
intervals to satisfy them.

The separate native fixture `original_predicates_counterexample.sio`
computes the same full MC/PBox/GUM/Klib flows and then applies the old
conjunction. It checks/compiles 0 and **executes with exit 1**, emitting only
the original false flags and the counterexample marker. Thus the original
expectations remain an executable failing scientific counterexample.

The oracle's value, zero-MC, missing-row, duplicate-row and NaN mutations are
all generated successfully and rejected with status 1. Missing oracle
arguments return 64.

## Custody and reproduction

Run `scripts/stdlib/stiff_ode_contract_gate.sh RAW SOURCE ORACLE FRESH_OUTPUT`
inside Slurm. It preserves the native thirty-second deadline and does no
compiler bootstrap. The exact worker, all phase logs/statuses, source and
ELF manifests are in [qualified](qualified/).

The frozen accepted source is
`/scratch/sounio-stdlib-wave3-20261027/core/stiff-contract04/source`.
The complete stdlib file manifest is compared to the isolated worktree's
manifest before execution, and all compiler/input/source/oracle/native ELF
hashes are verified again afterward. The earlier complete gate 19564 passed
before the additional final-PBox-variance observable; 19566 qualifies that
extension too. C++-only job 19561 established the numerical expectations
and tolerances before the first complete native execution 19562.
The initial C++ build 19560 had a global-name collision with the C library
function y0; its failed source/log is preserved separately from numerical
results.

Slurm **19567 completed 0**, created the durable bundle and verified every
file after archive extraction:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/stiff-ode01/stiff-ode-real-mc-source-raw-receipts-v1.tar.gz`

SHA-256:
`4da9c02baa323621d19fb106ca4e0bfcab5fbde2ca8beae1a73797fec3f7cebf`.

It contains the raw compiler, complete stdlib inputs, oracle sources/ELFs,
original failures, accepted program and counterexample, mutation logs,
preliminary/accepted jobs, and the root's original C++ point-band
counterexample. Readback and scheduler receipts are in [durable](durable/).
