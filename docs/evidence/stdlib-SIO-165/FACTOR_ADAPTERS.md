# SIO-165: declared covariance-factor adapters

These adapters extend the independent-input increment in PR #3103. A caller
supplies a finite signed row-major factor L declaring the input covariance
`Sigma = L L^T`. The implementation computes weighted Jacobian columns `H = J L`
and output covariance `H H^T`; it retains negative correlations and rank
deficiency. It does not project a matrix onto the positive-semidefinite cone,
clip correlations, or replace a declared covariance by independent marginals.
This is the first-order correlated-input rule of [JCGM 100:2008,
5.2](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).

## API and units

| Interface | Nominal input/factor-row order | Factor layout |
| --- | --- | --- |
| `classical_force_factor_checked(mass,acc,L)` | mass kg; acceleration m/s2 | 2x2, `[f64;4]` |
| `classical_ke_factor_checked(mass,vel,L)` | mass kg; velocity m/s | 2x2, `[f64;4]` |
| `classical_pos_factor_checked(s0,v0,a,t,L)` | position m; velocity m/s; acceleration m/s2; exact time s | 3x3, `[f64;9]` |
| `boost_e_p_factor_checked(E,p,v,L)` | energy J; momentum kg m/s; boost velocity m/s | 3x3, `[f64;9]` |

Nominals are explicit SI scalar arguments. No Quantity standard-uncertainty
field is silently ignored. Existing independent Quantity interfaces retain
unit checks and their original fast path. Factor entries carry the physical
units of their input row; factor columns refer to standardized latent sources.
L need not be triangular, invertible, or have positive diagonal entries.

Results use the existing scalar/pair layouts with `uncertainty_model=2` and
`first_order=true`, including unavailable results. Model 1 continues to identify
the independent-input interfaces. Status codes and cleared-unavailable payloads
are unchanged. Signed factor entries are valid; they are not negative standard
uncertainties. All three boost-output covariance entries, including the signed
cross-covariance, derive from the declared input factor.

Status 0 records numerical availability under the declared first-order model.
It does not establish that a supplied factor is physically justified or that
linearization is adequate. In particular, exact cancellation in J L gives zero
linearized uncertainty even when higher-order nonlinear terms could matter.
No nonlinear or expanded-uncertainty interpretation is supplied.

## Cancellation and representability

The factor adapter represents each Jacobian coefficient as a product of two
factors. This avoids prematurely overflowing an unweighted Jacobian coefficient
when multiplication by L would bring the result back into range.

Correlated columns need more care than independent norms: nearly cancelling
products can leave a small, scientifically relevant residual. For nonzero
monomial factors in `[1e-50,1e50]` in magnitude, the implementation reuses
`math::dd64::two_prod` and `two_sum`. Each triple product becomes an exact
four-component expansion; a small error-free expansion accumulates all three
terms before rounding to binary64. The bounds avoid splitter overflow and
underflow of the exact product residuals. Exactly zero monomials skip their
otherwise irrelevant factors. Factor pair cross-covariance uses the same
accumulation path.

Outside that range, balanced Lorentz products remain available. A signed
three-term sum with a residual below one eighth of its largest rounded term is
conservatively declined with status 4; the API does not report an unresolved
rounded-product cancellation as a precise covariance result. As in the original
slice, individual nonzero contributions that cannot be represented and required
output covariance entries that cannot be represented also return status 4.

The adapter need not materialize the input Sigma. A finite factor whose formal
Sigma entries exceed binary64 may still produce a representable output when
J annihilates its columns. Tests distinguish this from an unrepresentable
required output covariance. No global conditioning or accuracy certificate is
claimed for arbitrary inputs.

## Fresh executable evidence

Final Slurm job **17130**, UID1000, stack 524288 KiB, same retained source-built
compiler SHA256
`d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`.
Immutable source and result directories:

- `/scratch/sounio-stdlib-inference-scale-20261024/SIO165/factor08-source`
- `/scratch/sounio-stdlib-inference-scale-20261024/SIO165/factor08`

The complete gate passes **49 phase records** (`factor08/RESULTS.tsv`): six
programs check/compile/execute, three C++ concordance modes, six corrupted/missing
oracle rejection controls, and the five independent legacy rejection processes.
The unchanged independent and legacy cases are replayed alongside the new APIs.

The C++23 Decimal400 oracle constructs the full analytic Jacobian independently,
multiplies that matrix by the supplied factor, and compares native bit-decoded
values, standard uncertainties, and all output covariance entries. The factor
mode validates **11 scalar cases, four coupled boost cases, and ten unavailable
cases**, including:

- Negative correlation that annihilates the force Jacobian; rank-deficient KE
  and position factors; all-zero factors; a full-rank correlated boost.
- A nontriangular signed factor with the same covariance as a triangular factor,
  signed velocity, zero-nominal sensitivities, and large factors annihilated by J.
- A near-cancelling KE column whose residual is sensitive to multiplication
  rounding; a wide-exponent available column and a conservatively unavailable
  cancellation outside the expansion range.
- Nonfinite factors/nominals, physical-domain failures, output overflow, and
  verification of cleared numerical payloads with model 2 on every unavailable
  result.

The factor fixture used 1636 KiB maximum RSS and elapsed 0.00 seconds at the
recorded timer resolution, within the 10-second / 128-MiB bounded acceptance gate.
This is not a throughput or speedup claim. Final gate output is 6.0 MiB. New source
staging packets omit shipped compiler binaries and gzip files; factor08-source
is 31 MiB. Historical snapshots and compiler receipts remain unchanged.

Factor fixture SHA256:
`ebc106887f27ab8ed4e2ce32a9ea945ba09feea499288123e7abe88e5d92d2aa`.
Native factor ELF SHA256:
`cdf58e4efa66d1685c4aa439f3bdb2b55fdb7131691b939928def9927e03bd0e`.
Full artifact-manifest SHA256:
`b5f3471d8ffd14fb878b3fe82098f1951660bdbc5ac4074809f6de69650091ae`.

## Native failing-before / passing-after control

The immutable intermediate factor implementation in `baseline08-source` uses
rounded balanced monomials without exact expansion accumulation. It compiles
and executes the same imported production APIs and the same factor fixture.
Its gate exits 1. In the KE case `m=2,v=3,L=[0.4,0,-0.3,0]`, the independent
Decimal400 control rejects the native standard uncertainty:

```text
actual=2.22045e-16 expected=1.66533e-16 abs_error=5.55112e-17
```

The final implementation passes that same comparison. The intermediate candidate
also fails the newly explicit wide-cancellation availability assertion. This
control is a numerical regression against a retained intermediate candidate,
not a claim that origin/main previously exposed a covariance-factor interface.
The first independent increment's distinct original-source failure boundaries
remain documented in [README.md](README.md).

Before-helper source SHA256:
`2ce57bcdfd2ef75b1ecc23dd915fb68ea4981855ad8018a400ed7f28140f8258`.
After-helper source SHA256:
`92d8210deee825910a0435e80e0c77be91d231d7a97a6e74ea43d6266612bbb1`.
Small source/stdout/hash receipts are committed in `before_factor08` and
`factor08`; native ELFs remain in the immutable worker artifact directory.
The gate command remains `scripts/stdlib/physics_uncertainty_gate.sh`, with an
explicit source-built `MADAROS_RAW_BIN` and a new empty absolute output directory.
Stage2/seed full-suite CI, human PR review, and merge are separate checkpoints.
