# SIO-165: checked classical and relativistic uncertainty

Base: `origin/main` at `a2a44df195bb8caf3fb07cbc9517275b29325e94`.
Implementation branch: `devpass/SIO-165-physics-uncertainty`.

The classical and SR interfaces now propagate independent input standard
uncertainties through analytic first-order Jacobians. They retain nonzero
sensitivities at zero nominal mass or velocity and form a Euclidean norm of
weighted sensitivities instead of adding standard uncertainties. This follows
the independent-input first-order propagation rule in [JCGM 100:2008,
5.1](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
The coupled energy/momentum APIs additionally return their output covariance.

## Contract and public interfaces

Every `*_checked` scalar returns `PhysicsScalarResult` with `status`, `value`,
`standard_uncertainty`, `uncertainty_model=1`, and `first_order=true`. Model1
means supplied independent standard uncertainties and a first-order Jacobian.
Neither tag certifies input independence or adequacy of the local linearization.
The pair result adds two values, their standard uncertainties, and a symmetric
row-major 2x2 covariance. It computes `H H^T`, where each column of H is a
Jacobian column multiplied by the corresponding supplied standard uncertainty.

| Module | Checked interfaces |
| --- | --- |
| `physics::classical` | `classical_force_checked`, `classical_ke_checked`, `classical_pos_checked` |
| `physics::sr` | `gamma_checked`, `time_dilation_checked`, `length_contraction_checked`, `relativistic_energy_checked`, `relativistic_momentum_checked`, `velocity_addition_checked`, `invariant_mass_checked`, `invariant_mass_from_4p_checked`, `four_force_checked`, `four_force_long_checked` |
| `physics::sr`, pair outputs | `four_momentum_checked`, `two_particle_total_e_p_checked`, `boost_e_p_checked` |

| Status | Meaning |
| --- | --- |
| 0 | Available under the declared numerical/model contract |
| 1 | A supplied number is nonfinite |
| 2 | Physical/domain restriction fails |
| 3 | A supplied standard uncertainty is negative |
| 4 | Arithmetic or required uncertainty/covariance representation is unavailable |
| 5 | Classical Quantity dimensions do not match the operation |

Unavailable results clear all numerical payloads to zero. Scalar nominal
underflow may round to zero; a nonzero weighted uncertainty term that would
silently disappear is declined. Pair covariance additionally requires its
variances and individual cross-products to be representable. These checks are
conservative: status4 does not claim the exact mathematical result is impossible
to represent using every possible alternative algorithm.

The classical mass must be nonnegative and dimensions must match. SR velocities
satisfy `abs(v)<c`, with exact `c=299792458 m/s`. Invariant mass requires a strictly
timelike future-directed 1D momentum; the light cone has a singular mass
Jacobian and returns status2. Input uncertainties are finite and nonnegative.
Position propagation treats the supplied time as exact.

Legacy Quantity/tuple/FourMomentum result layouts remain available. These
functions call checked implementations and assert status0, so callers must now
admit `Mut, Div, Panic`. Invalid inputs no longer produce a plausible zero tuple.
The `invariant_mass_from_4p` nominal result is corrected from the old extra factor
of c. `compton_shift` continues using its existing validated scattering kernel.

Covariance output is not a correlated-input API: passing E/p marginal standard
uncertainties into `boost_e_p_checked` assumes those inputs are independent.
Callers must retain the returned covariance rather than treating coupled
outputs as independent. This package does not add nonlinear propagation,
coverage intervals, covariance inputs, or multidimensional SR tensors.

## Numerical implementation

`physics::uncertainty` reuses the actual public `physics::lorentz` scalar/vector
kernels for balanced triple products, compensated three-term sums, and scaled
four-term norms. `lorentz.sio` is unchanged. Gamma preserves the representable
SI gap using `c/sqrt((c-abs(v))*(c+abs(v)))`. Velocity addition forms its positive
denominator as a factored sum, avoiding cancellation for opposite near-c inputs.
Invariant mass normalizes its two positive factors before square roots, avoiding
an overflowing `z+abs(p)` when the result is representable.

The SR Jacobians count a shared velocity once. For example,
`dp/dv=m*gamma^3`, and the boost velocity derivatives are
`dE'/dv=-gamma^2*p'` and `dp'/dv=-gamma^2*E'/c^2`. The two-particle pair uses four
independent inputs `(m1,v1,m2,v2)` and preserves the induced E/p cross-covariance.

## Native evidence and practical limits

Final job **17117**, Slurm `cpu-ops`, UID1000, stack limit524288 KiB:
`/scratch/sounio-stdlib-inference-scale-20261024/SIO165/gate05`.
The gate records **42 passing phase checks** in `gate05/RESULTS.tsv`:

- Five independently checked, compiled, and executed programs: checked fixture,
  legacy fixture, classical module main, SR module main, and legacy Compton test.
- C++23 independent analytic reference using Boost Decimal400 and exact binary64
  input reconstruction: 20 scalar cases, five pair cases, and eight legacy rows.
  Pair comparisons include both standard uncertainties and all four covariance
  entries. Relative tolerance is5e-13 plus eight binary64 minimum subnormals.
- Thirteen checked-unavailable cases verify status and cleared payloads, including
  NaN, infinity, c, negative uncertainty/mass, light cone, dimensions, and overflow.
- Four oracle controls reject altered output and missing rows in both modes.
- Five separate processes verify that invalid public legacy calls exit1 after
  their start marker and before their return marker.

The bounded fixture covers zero-nominal sensitivities, signed velocity, the
nearest representable speed below c, opposite near-c velocities, large norms
whose naive squared contributions overflow, compensated cancellation, and an
invariant-mass input with `E/c=1e308` and `p=5e307`.

All five native programs fit the gate's10-second and128-MiB acceptance bounds.
The checked fixture used1080 KiB maximum RSS and legacy fixture1628 KiB. Both
elapsed readings were0.00 seconds at the timer's resolution; this is a bounded
regression measurement and does not establish a speedup. Final gate output is
5.1 MiB, excluding the shared compiler and source staging directories.

Compiler identity:
`d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`, retained at
`/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf`.
This is the retained source-built d74 executable, not a compiler freshly rebuilt
from this branch or current main. The gate rechecks compiler and source hashes
at completion. `gate05/ARTIFACTS.sha256` identifies the retained full artifacts;
this Git evidence directory contains small receipts and stdout, without ELFs.

## Failing-before witness, separated from compiler initialization

Job17099 against unmodified SR source passed compilation but its imported module
constant `C=speed_of_light_approx()` produced zero-gamma/NaN outputs under d74.
The original comparison-only fixture incorrectly accepted NaN. Those first
attempts remain preserved as `base02`; they are not numerical Jacobian evidence.

Job17104's explicit-c control (`base03`) retains the original SR production
functions with exactly one operational substitution:

```diff
-let C: f64 = speed_of_light_approx();
+let C: f64 = 299792458.0;
```

Its real imported gamma and four-momentum functions check0/compile0, then the
same native witness exits1. The negative gamma uncertainty bit pattern is
`-4660324894402989261`, and invariant mass bits are`4852146588476102301` for
`E=c*c,p=0`. The corrected production import (`after03`) check0/compile0/exits0,
with positive uncertainty bits`4563047142451786550` and mass bits
`4607182418800017404`, then prints`SR_BASELINE_PASS`. Final gate05 additionally
uses the independent C++ reference rather than comparison-only acceptance.
The exact original global-initialization failure and the explicit-c numerical
failure are distinct observations.

Original SR SHA256:
`86716e025a609382daa743a0b8b2a1f29a3b88df6683f36c0d4de1619948f88d`.
Explicit-c legacy-control SR SHA256:
`4a329a062aaa08c15390e498d3987354a7e30697fc35b74d7bd8ed86648fa4da`.
The older classical import also failed AST closure; canonical `units::lib` imports
and use of its public `dim_force`/`quantity_new` repair that separate defect.

## Reproduction

On a Linux Slurm worker as UID1000, from an immutable source snapshot containing
launchers, resolver scripts, stdlib, and the listed tests/reference:

```sh
export MADAROS_RAW_BIN=/absolute/path/to/source-built/madaros.elf
export PHYSICS_UNCERTAINTY_OUT=/absolute/path/to/new-empty-output
bash scripts/stdlib/physics_uncertainty_gate.sh
```

Do not include shipped `bin/*gz` in new staging packets: the resolver may
materialize an unused125-MB shipped compiler even with an explicit compiler
override. Historical snapshots are retained unchanged. No quota was altered and
no source snapshot, compiler receipt, or other lane's artifact was removed.

This evidence establishes the focused native contract. Stage2/seed full-suite CI,
PR review, and human merge remain separate checkpoints.
