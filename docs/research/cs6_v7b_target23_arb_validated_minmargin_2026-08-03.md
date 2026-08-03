# Target-23 independent Arb validated minimum-margin center orbit

**Date:** 2026-08-03

**Executed commit:** `62c4deb0d49c1f0457c2b0ea62c745631cff1e13`

**Slurm job:** `8548` (`COMPLETED`, 16 seconds, one task with 4 requested CPUs)

**Verdict:** an independently implemented validated enclosure certifies the
frozen center trajectory and its oriented-return determinant on the retained
minimum-margin leaf. No leaf-wide or global certificate follows.

## Question

The preceding Decimal/RK4 scout found that all 331 computed center points lay
inside the retained CAPD Liouville intervals, but supplied no rigorous error
bound. Could a different interval implementation validate the most demanding
center trajectory without reading CAPD output during integration?

The selected leaf is `U08-0000000223_S09-0000000325`, index 331 in the frozen
cover. It had the smallest observed Decimal-to-CAPD endpoint margin in the
pointwise scout.

## Independent validated route

The worker uses `python-flint 0.8.0` Arb real balls at 256-bit precision, one
thread, an order-40 Taylor polynomial, and fixed step `2^-8`. Every accepted
step proves both

```text
X0 + [0,h] F(B) subset B
h * L_infinity(B) < 1.
```

The first condition contains the step trajectory in a Picard box; the second
gives a strict contraction. The degree-40 polynomial is paired with an
order-41 interval remainder. A separately accumulated state radius includes
the interval polynomial rounding radius and propagates earlier error with the
infinity logarithmic norm.

The worker independently isolates the first two negative-to-positive `w=0`
returns. Its event Picard boxes contain the complete bracket trajectories, and
positive lower bounds for `x*y-zs` establish transversality. It then encloses
the oriented-return determinant using

```text
exp(ell(T)) * normal_velocity(0) / normal_velocity(T) * det(DQ0).
```

All frozen decimal inputs are converted first to exact base-10 rationals. CAPD
data are absent from the worker and enter only in the independent verifier's
post-execution containment comparison.

## Result

- `1793/1793` Picard calls proved containment; the maximum was 5 iterations.
- The largest certified contraction factor was about `0.2078268922496043`,
  strictly below 1.
- The maximum propagated global state radius was about
  `3.354759306896197e-15`.
- Two returns were validated. The second event bracket had width exactly
  `2^-50`; the initial and final section-normal lower bounds were about
  `142.8566` and `54.3578`, respectively.
- The determinant was enclosed in approximately
  `[-2.8398635728038477e-11, -2.8398635728037957e-11]`.
- Its exact interval width was
  `338587319/649037107316853453566312041152512`, about
  `5.216763651615146e-25`.
- The complete validated determinant interval was strictly negative and
  strictly inside both retained CAPD carrier intervals.
- Cluster and local verification passed. The verifier rejected `14/14`
  deliberate semantic and provenance mutations.

The raw worker used Python `3.12.3`; the staged `python-flint` wheel and loaded
native extension are hash-attested in the receipt. The result arrived in a
30,720-byte tar archive whose received SHA-256 exactly matched the framed
header.

## Failed attempts retained

Jobs `8538`, `8539`, `8540`, `8541`, `8543`, and the initial `srun` attempt
exited before the worker. They exposed two execution-path defects: environment-
empty `sbatch` launches died before the batch body, and `pip` rejected a staged
wheel whose safety prefix broke the wheel filename grammar. A traced `srun`
allocation identified the latter exactly. Job `8548` reran the complete frozen
experiment after copying the hash-verified wheel to its canonical private
filename. None of the failed attempts is counted as scientific evidence.

## What this establishes

This is enclosure-level implementation diversity for one exact frozen center
trajectory. The Arb/Taylor route independently validates the trajectory,
events, transversality, and Liouville/oriented-return determinant, and its final
determinant enclosure is compatible with both retained CAPD carriers.

This closes the specific gap left by the Decimal/RK4 point scout on the
minimum-margin center: the conclusion is now interval-certified rather than a
precision comparison.

## What this does not establish

The initial condition is a single exact center. Its small propagated numerical
radius is an integration-error enclosure, not the geometric width of the leaf.
The result therefore does not certify neighboring initial conditions, the full
leaf, the 331-leaf cover, an independent full-leaf interval engine, global
H-PG, V7-B eligibility, promotion, novelty or priority, or a solution to an
open problem. No FPGA was used.

## Next window

The next scientifically meaningful step is to lift this independent Arb route
from a point initial condition to the entire minimum-margin leaf. That requires
propagating the leaf's initial interval geometry, controlling wrapping and
event-time dependence across the box, and comparing the resulting return-map
determinant enclosure with the two CAPD carriers. Only after that succeeds is a
331-leaf replay justified.

## Durable evidence

The receipt directory is
`scripts/research/receipts/cs6_v7b_target23_arb_validated_minmargin_v1/`.
It retains the complete Slurm result, exact worker transcript, cluster and local
verification, dependency and runtime provenance, mutation results, execution
binding, failed-attempt classification, and pre- and post-result orthogonal LLM
reviews. Grok and Z.AI both returned `PASS` on the narrow mathematical claim and
confirmed that it has no leaf-wide or global implication.

## Integration blocker

The scientific gate is green, but the repository-wide docs registry gate is
not. It reports pre-existing stale generated governance files and unrelated
paper metadata. This lane does not own those shared surfaces. The retained B2
blocker requires `docs-governance/A2` to restore
`bash scripts/dev/check_docs_registry.sh`; until then, local evidence can be
published on this branch, but merge and CI readiness are not claimed.
