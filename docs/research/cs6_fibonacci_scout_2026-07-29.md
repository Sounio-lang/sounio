# CS6 Fibonacci covering scout and proof-ready CAPD machine

**Date:** 2026-07-29
**Lane:** `cs6-fibonacci-scout-20260729`
**Base:** `272247874ab882e9ab0603e8a46a96a28d68e001`
**Status:** proof geometry found; bounded interval falsification passed; exhaustive
authorised replay not run

## 1. Result boundary

This wave found a two-h-set candidate for one common map `F=P^6` of the CS6
Poincare map. The sampled graph is

```text
N0 -> N0    degree -1
N0 -> N1    degree -1
N1 -> N0    degree +1
```

The non-rigorous `81 x 81` scout passes all three candidate relations. A
separate CAPD/FILIB program then validated 2,133 adversarial interval boxes
with zero exceptions and strictly positive margins. The smallest observed
margin was `0.135304910`.

That is not the exhaustive covering proof. The authorised heavy-batch surface
failed before submission, so this wave does not promote positive entropy:

```text
numerical_candidate_found = true
bounded_interval_falsification_passed = true
fibonacci_coverings_proved = false
positive_entropy_proved = false
uniform_hyperbolicity_proved = false
chaotic_attractor_proved = false
flow_entropy_bound_proved = false
```

The anti-promotion receipt is
`scripts/research/cs6_capd_fibonacci_covering_certificate_v1.txt`. The
aggregator is the only program in this wave allowed to replace it with a
certificate containing true covering and entropy fields.

## 2. Scientific target

The exact ODE is

```math
x' = 2y^2-xy,
y' = xy-yz/2,
z' = xy-z.
```

The inherited CAPD certificate proves a locally unique hyperbolic saddle point
of prime period six for the upward Poincare map on

```text
Sigma = { z = 22.3274637391 }.
```

The source paper reports numerical chaos and asks for rigorous proofs of the
chaotic attractors in its table. The inherited UPO is a rigorous anchor, but a
hyperbolic UPO alone is not chaos. This wave targets the smallest finite graph
that can add recurrent symbolic dynamics.

Primary context and theorem machinery:

- [Plesa--Sprott CS6 paper](https://doi.org/10.1063/5.0323112)
- [Gidea--Zgliczynski covering relations](https://doi.org/10.1016/j.jde.2004.03.013)
- [Wilczak--Zgliczynski symbolic dynamics](https://doi.org/10.1016/j.jde.2020.06.020)
- [CAPD h-set implementation](https://capd.sourceforge.net/capdDynSys/docs/html/a01922.html)

A bounded title, equation, reference, and CAPD-example search in the inherited
proof-machine wave found no earlier validated CS6 covering certificate. This is
a novelty window, not proof of priority. First-proof wording still requires an
author/specialist literature check.

## 3. Frozen geometry

Coordinates on the section use the proved period-six point as origin and the
numerical eigendirections of `DP^6` as a common affine frame:

```text
origin   = (15.186446520640786, 10.908543194765466)
e_u      = (-0.67430316214199759, -0.73845463335624273)
e_s      = (-0.94170446778164518,  0.33644122125579123)
det      in [-0.92226940685332637, -0.92226940685332570]
```

For `q(u,s) = origin + u e_u + s e_s`, the h-sets are

```text
N0: center (0, 0),                    radii (0.0040, 0.3)
N1: center (0.019771776972779206, 0), radii (0.0015, 0.3)
```

Their unstable-coordinate gap is `0.014271776972779206`, so the two compact
sets are rigorously disjoint in the common invertible frame. The `N1` centre
was found as a secondary zero of the unstable coordinate of `P^6`; its point
image is approximately

```text
(u,s) = (6.8816615067286045e-15, 4.3683658393826571e-6).
```

## 4. Scout evidence

The point-map scout uses CAPD's non-interval `DPoincareMap`; therefore every
number in this section is reconnaissance only. With an `81 x 81` grid:

| Edge | Degree | Entry margin | Left-exit margin | Right-exit margin |
|---|---:|---:|---:|---:|
| `N0->N0` | -1 | `9.6513253770e-5` | `5.6960057344` | `2.9141719078` |
| `N0->N1` | -1 | `3.5447180232e-4` | `3.6748306432` | `22.6189764026` |
| `N1->N0` | +1 | `6.2879764824e-5` | `0.2864199792` | `0.2174122928` |

The replayable receipt explicitly keeps `rigorous_coverings_proved=false`.

## 5. Rigorous predicates

For source `Ni`, target `Nj`, and target coordinates

```math
(U,S)=c_j^{-1} P^6 c_i(u,s),
```

every support tile must prove CAPD's `across` predicate

```math
|U|>1 \quad\text{or}\quad |S|<1.
```

Here the forbidden set is the **target entry boundary**
`[-1,1] x {-1,+1}`, not the whole target h-set. Thus an image strictly
inside the stable strip is allowed and avoids that boundary. This is the same
whole-support test implemented by CAPD `HSet2D::across`; it must not be
confused with asking the source entry face to map outside the entire target.

For degree `+1`, the complete left and right faces must satisfy respectively
`U<-1` and `U>1`. For degree `-1`, the inequalities are reversed. Every
inequality is strict. A CAPD exception, an omitted tile, a non-positive return
time, a non-finite upper return-time bound, or a non-positive initial/final
section-normal velocity makes the shard fail.

Each box is represented by `C0HOTripletonSet` and propagated through exactly
six `MinusPlus` returns. Successful `IPoincareMap` evaluation supplies the
validated existence and transversality checks used internally by CAPD. The
receipt additionally records the total return-time enclosure and both endpoint
normal-velocity enclosures. It does not claim a separate interval proof of the
entire flowpipe's chemical positivity.

The frozen exhaustive partition is

```text
ORDER=8
N0_U=200
N1_U=75
SUPPORT_S=75
EXIT_S=1200
raw interval maps=25425
edge-role ledger records=42825
```

Order 8 is intentional. On this problem, orders 12 through 32 accumulated
more wrapping for the six-return set propagation. Directed interval arithmetic
keeps order 8 rigorous; it is an enclosure-efficiency choice, not a precision
relaxation.

## 6. Bounded interval falsification

Before requesting the full batch, the exact tile dimensions were attacked on
all unstable centres along three stable rows `s=-0.296,0,+0.296`:

```text
N0->N0: 600/600 pass, min margin 0.28087264427002667
N0->N1: 600/600 pass, min margin 0.28087264427002667
N1->N0: 225/225 pass, min margin 0.545998840331
exceptions: 0
```

Faces were tested at 118 adversarial stable centres per edge-face relation,
including both extremes, a dense centre band, and a stride through the full
face:

```text
N0->N0: left 5.67949,  right 2.85903
N0->N1: left 3.63079,  right 22.4719
N1->N0: left 0.264945, right 0.135304910
exceptions: 0
```

These are rigorous enclosures of sampled boxes, but the unvisited boxes forbid
global promotion.

The order-8 sweep used binary SHA-256
`92088f76182c947ea1cc36cd9567dbb07134a20f2ac7a1fb554177f9bb86e649`.
Subsequent proof-pipeline hardening added initial-normal and finite-time guards,
mandatory ledgers, and provenance checks. It does not promote or silently
reinterpret this earlier bounded sample; the current source is frozen
separately by the gate.

## 7. Exhaustive replay and aggregation

On an authorised Foundry/Slurm CPU allocation, the checked-in run driver takes
a source snapshot, builds one retained executable, records source/binary/CAPD
configuration/compiler hashes and their retained inputs, and runs a complete
shard partition. It refuses to run without `SLURM_JOB_ID`:

```bash
bash scripts/research/cs6_capd_fibonacci_covering_run.sh \
  --run-dir <new-artifact-directory> \
  --shards 12 \
  --jobs 12 \
  --capd-config <pinned-capd-config>
```

The driver does not submit a Slurm job; Foundry invokes it inside an authorised
allocation. For shard ordinal `i` of `n`, its verifier interface is:

```text
cs6_capd_fibonacci_covering 200 75 75 1200 8 i n ledger-i.txt > shard-i.txt
```

No C++ execution, including an unsharded one, can print a global proof claim.
It also refuses proof mode without a ledger. After all shards return zero,
aggregate:

```bash
python3 scripts/research/cs6_capd_fibonacci_covering_aggregate.py \
  --run-dir <artifact-directory> \
  --shards <n> \
  --ledger-output <artifact-directory>/ledger-canonical.txt \
  --certificate-output <artifact-directory>/certificate.txt
```

The aggregator checks the retained manifest/source/binary hashes; every frozen
shard preamble; exact expected / seen / passed counts; all `42,825` unique
ledger keys; and the canonical source box for every index. It reparses every
image enclosure and recomputes the `across` or exit predicate, checks strict
margins, finite positive times, positive initial/final normal velocities, and
absence of exceptions. It canonicalises the ledger and binds it, the executable
and the complete shard bundle with SHA-256.

The manifest declares
`EXECUTION_TRUST_MODEL=AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION`.
Those retained inputs and hashes establish artifact identity and
reproducibility; they are not remote attestation against a hostile executor.
The aggregate certificate therefore retains
`REMOTE_ATTESTATION_PRESENT=false` and `INDEPENDENT_REPLAY_REQUIRED=true`.
Scientific acceptance still requires an independent replay in a trusted
CPU/CAPD environment.

Only then may the aggregate state

```text
FIBONACCI_COVERINGS_PROVED=true
POSITIVE_ENTROPY_PROVED=true
```

## 8. Conditional theorem

If the exhaustive certificate succeeds, the common-map graph has adjacency

```math
A=\begin{bmatrix}1&1\\1&0\end{bmatrix},
```

with spectral radius `phi=(1+sqrt(5))/2`. The covering-relations theorem then
gives a compact invariant set carrying Fibonacci symbolic dynamics and

```math
h_{top}(P^6) \ge \log(\phi),
\qquad
h_{top}(P) \ge \log(\phi)/6,
\qquad \log(\phi)/6 \approx 0.08020197084326724.
```

This is a conditional statement in the current artifact, not its result.
Coverings alone do not prove uniform hyperbolicity; that upgrade requires cone
conditions. They also do not prove an attractor; that requires a trapping
region. A flow-entropy-per-time claim requires a suspension/roof argument.

## 9. Execution blocker

The current live audit found idle Slurm CPU resources, but the repository
requires workspace submissions through BeagleCockpit. The authorised surface
failed before any full job was created:

```text
beagle hpc profiles -> HTTP 401, exit 22
auth-bridge desired/available/ready replicas -> 0/0/0
auth-bridge EndpointSlice addresses -> none
```

```text
Blocker-ID: BLK-20260728-cs6-cluster-ops-auth-bridge
Status: classified; reproduced 2026-07-29
Severity: B3
Class: platform-resource
Evidence-Level: E2
Owner: Cluster Ops
Lane: cs6-fibonacci-scout-20260729
Worktree: /tmp/sounio-cs6-fibonacci-scout-20260729
Branch: research/cs6-fibonacci-scout-20260729
Evidence: beagle HTTP 401; auth-bridge replicas 0; no endpoint
Acceptance-Gate: auth-bridge has a ready endpoint and beagle lists an authorised CPU proof profile
Next-Action: Cluster Ops restores the bridge; Foundry runs the frozen 12-shard CPU replay and returns its artifact directory
```

The existing `BLK-20260728-cs6-u250-resource-absent` remains accurate but is not
the cause of this stopped CPU replay. Physical installation can happen later.
U250s are excluded from the proof's trusted computing base and will be useful
only for proposing future tiles or cone candidates whose accepted leaves are
replayed by CPU interval arithmetic.

## 10. Path classification

```text
default Sounio interval path used = false
rebuilt current-source CAPD path used = true
fallback path used = false
legacy numerical reconnaissance kept = true
full authorised CAPD replay completed = false
```

No Sounio compiler semantics, bootstrap path, ontology, or legacy numerical
route was removed or changed in this wave.
