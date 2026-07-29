# CS6 proof machine: from numerical chaos to a rigorous hyperbolic orbit

**Date:** 2026-07-28

**Status:** `RIGOROUS_LOCAL_UPO`; `OPEN_CHAOS_CERTIFICATE`

**Target:** CS6 in Plesa--Sprott, *Simple chemical systems with chaos*

**Concept:** `SOUNIO-SCIENCE-RESEARCH-BOUNDARY`

**Local contract:** `scripts/research/cs6_multiple_shooting_contract.py`

**CAPD proof:** `scripts/research/cs6_capd_periodic_orbit.cpp`

**Frozen CAPD receipt:** `scripts/research/cs6_capd_periodic_orbit_certificate_v1.txt`

**Gate:** `scripts/ci/cs6_proof_machine_gate.sh`

## 1. Verdict

This lane crossed one real boundary and refused the next one.

For the CS6 chemical dynamical system

```math
\dot x=2y^2-xy,\qquad
\dot y=xy-\frac12yz,\qquad
\dot z=xy-z,
```

CAPD 5.3.0 with its FILIB double interval backend proves that the upward
Poincare map on

```math
\Sigma=\{z=22.3274637391\}
```

has a locally unique hyperbolic point of prime period six. The corresponding
flow orbit is positive, has period in

```text
[29.510309125879111, 29.510309313460304],
```

and has one expanding and one contracting transverse multiplier:

```text
mu_u in [-5.9473828066101087, -4.6481361856663446]
mu_s in [-2.1618758145611223e-35, -1.6895952739109806e-35].
```

The exact promotion boundary is:

```text
periodic_orbit_proved = true
hyperbolicity_proved = true
chaos_proved = false
chaotic_attractor_proved = false
```

A hyperbolic unstable periodic orbit is not by itself a proof of chaos. The
next finite certificate must connect recurrent geometry, either through a
transverse homoclinic point or a directed graph of covering relations.

## 2. Why this targets an open problem

Plesa and Sprott introduced CS6 as a six-monomial chemical dynamical system
with a four-reaction, three-quadratic reaction network. Their evidence is
numerical: initial condition `(10,1,10)`, Lyapunov spectrum approximately
`(0.0424,0,-2.7004)`, and Lyapunov dimension `2.0157`. In their discussion they
explicitly ask for rigorous proofs that the systems in their tables have
chaotic attractors.

This artifact does not close that problem. It supplies a new rigorous anchor
inside the numerically reported CS6 invariant set. A bounded search through
the target paper, its references, title/DOI searches, the exact equations, and
CAPD examples found no prior validated periodic-orbit or chaos certificate for
CS6. That absence is not a proof of novelty. Author and specialist review are
required before any first-proof language.

Primary target sources:

- [Plesa--Sprott arXiv record and full text](https://arxiv.org/abs/2601.02787)
- [Published article, DOI 10.1063/5.0323112](https://doi.org/10.1063/5.0323112)

## 3. Semantic lane

```text
Semantic-Lane-ID: cs6-proof-machine-20260728
Owner: codex-1
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: machine evidence must be narrower than the scientific claim
Transformation: numerical recurrence to a validated prime-period hyperbolic orbit
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: one locally unique prime-period-six hyperbolic UPO for exact CS6
Claims-Forbidden: chaos, horseshoe, homoclinic orbit, positive entropy, or attractor
Assumptions: CAPD/FILIB arithmetic and integration implementation are correct
Write-Set: this note, two proof sources, two receipts, and one gate
Read-Set: CS6 paper; CAPD 5.3.0 source/docs; U250/XRT documentation
Positive-Witness: strict CAPD interval-Newton inclusion and separated multipliers
Negative-Witness: single-box long-time propagation is refused by wrapping
Acceptance-Gate: bash scripts/ci/cs6_proof_machine_gate.sh
Integration-Target: research branch only
Authoritative-Only-If: mandatory independent math reviews pass
```

## 4. The CAPD certificate

### 4.1 Section and box

The implementation uses `w=z-z_s`, with the exactly specified rational

```math
z_s=223274637391/10^{10},
```

and the `MinusPlus` crossing of `w=0`. The two-dimensional section box is

```text
X = {
  [15.186446510640785, 15.186446530640787],
  [10.908543184765465, 10.908543204765467]
}.
```

The normal velocity enclosure on the sixth return is strictly positive:

```text
[143.33424557674189, 143.33484263654915].
```

All five preceding return computations also have strictly positive normal
velocity, so the oriented Poincare map is well-defined on the validated sets.

### 4.2 Existence and local uniqueness

Let `P` be the first-return map and `F=P^6-Id`. CAPD rigorously computes
`F(x_0)` at the centre and an interval enclosure `[DF(X)]` from the
variational equations. It then forms the interval-Newton image

```math
N=x_0-[DF(X)]^{-1}F(x_0).
```

The frozen receipt gives

```text
N = {
  [15.186446512748891, 15.186446528535276],
  [10.908543191668944, 10.908543197859878]
}
NEWTON_INTERIOR=true
```

Since `N` lies strictly inside `X`, the interval-Newton theorem proves one and
only one zero of `F` in `X`. This is local uniqueness in the displayed box,
not global uniqueness of all CS6 period-six orbits.

The code follows CAPD's official Rossler periodic-orbit pattern: a rigorous
centre image in `C0HOTripletonSet`, first variational enclosures in
`C1HORect2Set`, conversion from flow derivative to Poincare derivative with
`computeDP`, interval Gaussian elimination, and `subsetInterior`.

- [CAPD rigorous Poincare maps and derivatives](https://capd.sourceforge.net/capdDynSys/docs/html/a05237.html)
- [CAPD Rossler interval-Newton example](https://capd.sourceforge.net/capdDynSys/docs/html/a05240.html)

### 4.3 Prime period

For each `k=1,...,5`, CAPD encloses `P^k(X)` and proves it disjoint from `X`.
The smallest normal-velocity lower bound among those returns is
`9.9863422730990301`, at `k=4`. Therefore the fixed point of `P^6` cannot have
any smaller Poincare period, and its prime Poincare period is six.

### 4.4 Hyperbolicity without a cancellation error

Direct interval evaluation of the determinant of the `2 x 2` enclosure for
`DP^6` gives `[-2.9814,2.9814]`, which is useless because dependency causes
severe cancellation. The proof does not use that interval.

Instead, a fourth ODE state integrates the divergence

```math
\operatorname{div}f=x-y-z/2-1.
```

The validated six-return integral is

```text
[-78.28303797907661, -78.28303565833528].
```

For a three-dimensional flow returning to the same section,

```math
\det DP(x)=\exp\left(\int_0^{\tau(x)}\operatorname{div}f(\phi_t x)dt\right)
\frac{n\cdot f(x)}{n\cdot f(Px)}.
```

At the proved fixed point of `P^6`, the normal-velocity ratio is exactly one.
Liouville's identity therefore gives

```text
det(DP^6) in
[1.0048669882187863e-34, 1.0048693202578456e-34].
```

Combining this with

```text
trace(DP^6) in [-5.9473828066101078, -4.6481361856663455]
```

separates both real roots of the characteristic polynomial into the
multiplier intervals in Section 1. One lies strictly below `-1`; the other
lies strictly between `-1` and `0`. The period-six point is a hyperbolic
saddle, hence an unstable periodic orbit.

## 5. The self-falsifying local scaffold

Before CAPD was available, the lane built a dependency-free directed-decimal
Taylor/Picard integrator. Its purpose is both reconnaissance and an independent
falsifier of easy numerical stories.

For a local initial box `X`, step `h`, and candidate tube `Y`, every accepted
step establishes

```math
X+[0,h]f(Y)\subset\operatorname{int}Y.
```

For Taylor order 18, meaning polynomial terms `0,...,18`, the endpoint uses

```math
\phi_h(X)\subset
\sum_{k=0}^{18}A_k(X)h^k+h^{19}A_{19}(Y).
```

The coefficient recurrences are exact polynomial identities. For example,

```math
(k+1)A_{k+1,x}
=2\sum_{i=0}^k A_{i,y}A_{k-i,y}
-\sum_{i=0}^k A_{i,x}A_{k-i,y}.
```

Every Decimal operation uses a 70-digit `ROUND_FLOOR` lower context or
`ROUND_CEILING` upper context. Python `decimal.Context` is part of the trusted
computing base and is not formally verified.

### 5.1 The negative control worked

Propagating one axis-aligned interval box without resetting shooting nodes
caused wrapping growth. Even after adaptive halving, the proof was refused at
approximately

```text
t = 0.924881992340087890625
smallest attempted dt = 1.9073486328125e-8.
```

This is a useful negative result: long-time interval arithmetic does not
become a proof merely by increasing precision.

### 5.2 The 296-segment witness

Resetting at shooting nodes produced 296 local enclosures over
`T=29.510309219673534`:

```text
295 segments * 0.1       = 29.5
final segment            = 0.010309219673534
295 segments * 20 steps  = 5900
final segment steps      = 3
total accepted steps     = 5903
rejected steps           = 0
```

The executable ledger checks every start/end time, exact duration sum, and
absence of gaps or overlaps. Results:

```text
max local endpoint width = 2.7970799962504979e-16
minimum Picard margin    = 1.1781453369552920e-8
max closure residual     = 6.7238508696836924e-13
records SHA-256          = 1533d7dfe80792c471d9fc5d8fa29e87
                              d62ab802ca1d5bc6ed027882b3e2fa92
```

The earlier fixed-step RK4 event-shooting residual near `1.5e-12` was a
discretised self-consistency result. The first directed-Taylor replay exposed
a true fixed-time discrepancy around `2.5e-6`; two numerical Newton
corrections produced the witness above. Those reconnaissance values are not
the proof. The frozen local witness explicitly keeps
`local_enclosures_glued=false` and all scientific promotion bits false.

The gap is deliberate: a closure residual of `6.8e-13` is larger than the
local endpoint widths. Midpoint resets prove 296 separate initial-value
problems, not the existence of one trajectory passing through every node.
CAPD's interval-Newton inclusion, not the local witness, proves the UPO.

## 6. Trusted-computing-base boundary

The proof path is external CAPD 5.3.0 built from pinned source, not the default
Sounio interval library and not a silent fallback.

`stdlib/data/interval.sio` correctly warns that its `f64` operations are not
outward-directed. `stdlib/verify/interval.sio` currently describes its
arithmetic as ISO/outward-rounded but implements ordinary `f64` endpoint
operations without directed rounding. Neither file is in this proof's trusted
computing base. `stdlib/data/interval_rat.sio` has exact rational endpoints but
does not provide a validated ODE solver.

Path classification:

```text
default Sounio interval path used = false
rebuilt current-source CAPD path used = true
fallback path used = false
legacy numerical reconnaissance kept = true
```

## 7. The next theorem-sized target

The smallest plausible certificate is not a full two-shift. Take two h-sets
`N0,N1` and validate only three covering relations:

```text
N0 -> N0
N0 -> N1
N1 -> N0
```

Their adjacency matrix is

```math
A=\begin{bmatrix}1&1\\1&0\end{bmatrix},
```

whose spectral radius is the golden ratio `phi>1`. Standard covering-relation
theorems then supply a compact invariant set with Fibonacci symbolic dynamics
and positive topological entropy. Cone conditions are not required for this
minimal C0 existence/entropy statement; they are the upgrade that proves
uniform hyperbolicity and sharper coding properties.

More precisely, if the validated relation graph is for `F=P^m`, then the
subshift gives `h_top(F) >= log(phi)` and hence
`h_top(P) >= log(phi)/m`. The first U250 chart target is `F=P^6`, but a smaller
iterate is admissible if reconnaissance produces tighter sets. All three
edges must use the same `F`; variable-duration edges must first be uniformised
with intermediate nodes or analysed as a weighted shift.

A transverse-homoclinic route remains conceptually direct:

1. Construct validated local parameterisations for the stable and unstable
   manifolds of the proved period-six point.
2. Use non-rigorous multiple shooting to locate a candidate homoclinic return
   on the Poincare section.
3. Validate the connection with interval Newton/Krawczyk or a Chebyshev
   boundary-value/radii-polynomial argument.
4. Prove transversality by separating the determinant of stable and unstable
   tangent directions from zero.
5. Invoke Smale--Birkhoff to obtain a horseshoe for an iterate.

The multiplier near `1e-35` makes backward propagation along the stable
direction exceptionally ill-conditioned. That makes the homoclinic route
higher-risk unless reconnaissance finds a clean crossing. It should use a
parameterisation method rather than naive backwards boxes.

This is established machinery rather than a new proof principle. The novelty
window is its first rigorous application to CS6, with a compact, replayable
certificate:

- [Gidea--Zgliczynski, covering relations for multidimensional systems](https://doi.org/10.1016/j.jde.2004.03.013)
- [CAPD covering-relation and h-set implementation](https://capd.sourceforge.net/capdDynSys/docs/html/a01922.html)
- [Wilczak--Zgliczynski, covering relations to symbolic dynamics](https://doi.org/10.1016/j.jde.2020.06.020)
- [Zgliczynski, cone conditions as the uniqueness/hyperbolicity upgrade](https://doi.org/10.1016/j.jde.2008.12.019)
- [Murray--Mireles James, validated transverse connections for periodic ODE orbits](https://arxiv.org/abs/2405.12446)

The Fibonacci covering route is certificate-minimal and implementation-friendly
because its tile tests are finite, independent, and naturally accelerated. The
homoclinic route gives the stronger classical horseshoe conclusion if its
geometry is numerically clean. Both may be explored; only a small
CPU-replayable certificate can promote either route.

### 7.1 A higher-risk Shilnikov scout

CS6 also has the positive equilibrium `(4,2,8)`. Its Jacobian has the exact
spectrum

```math
J(4,2,8)=
\begin{bmatrix}-2&4&0\\2&0&-1\\2&4&-1\end{bmatrix},
\qquad
\det(\lambda I-J)=(\lambda+4)(\lambda^2-\lambda+2),
```

and therefore

```math
-4,\qquad \frac12\mathbin{\pm}i\frac{\sqrt7}{2}.
```

For the reversed flow this becomes a saddle-focus with real unstable exponent
`4`, stable real part `-1/2`, and favourable Shilnikov saddle quantity
`4-1/2=3.5`. A homoclinic loop to this equilibrium would open a second route
to rigorous chaos. The missing loop is a codimension-one geometric event and
there is currently no evidence that it occurs at the fixed CS6 coefficients.
This merits a cheap one-dimensional unstable-manifold scout, not a promoted
claim or the primary proof budget.

- [Capinski--Wasieczko-Zajac, CAPD proof of Shilnikov homoclinics](https://doi.org/10.1137/16M1079956)

Neither route alone proves a chaotic *attractor*. Closing the paper's exact
open problem additionally requires a trapping/absorbing region and a precise
argument that the certified chaotic invariant set belongs to the attracting
dynamics. The first defensible promotion is therefore:

```text
CS6 contains a compact invariant set with symbolic dynamics and positive entropy.
```

not:

```text
the numerically displayed CS6 attractor has been proved chaotic.
```

Add `hyperbolic` only after cone conditions or a transverse-homoclinic
horseshoe are certified.

## 8. Why two U250s matter after, not inside, the proof

The U250 is not in the trusted computing base. CPU CAPD remains responsible
for rigorous flow integration, return times, chart remainders, and final replay.

The useful FPGA boundary is a deterministic search accelerator:

1. CPU CAPD emits degree-12 bivariate Taylor charts for `P^6` and `DP^6`, with
   rigorous remainder intervals, in coordinates aligned to stable/unstable
   directions.
2. Eight compute units, four per U250 and one per SLR/DDR bank, evaluate those
   charts over millions of dyadic subtiles.
3. Each tile returns only `COVER`, `DISJOINT`, `UNDECIDED`, `OUTSIDE`, or
   `OVERFLOW` plus a deterministic manifest.
4. A fixed fraction of batches is duplicated across cards and must be
   bit-identical.
5. Every accepted tile is replayed independently by CPU CAPD before entering
   a certificate.

For degree 12, two map components contain 91 monomials each and four Jacobian
components contain 78 each, about 494 interval fused multiply-add evaluations
per tile. The proposed datapath uses scaled dyadic `Q8.56` endpoints,
`ap_int<128>` exact products, explicit floor/ceiling, and sticky overflow.
Coefficient scaling must be emitted and checked per chart; default HLS
truncation or wraparound is forbidden.

Before synthesis, `0.5--2 million tiles/s` across eight compute units is only
a sizing estimate. It becomes evidence only after `csynth`, place-and-route,
measured Fmax/II/resources/DMA, cross-card duplication, and CPU-oracle replay.

Hardware references:

- [AMD U250 product details](https://docs.amd.com/r/en-US/ds962-u200-u250/Alveo-Product-Details)
- [AMD U250 Gen3x16 XDMA platform](https://docs.amd.com/r/en-US/ug1120-alveo-platforms/U250-Gen3x16-XDMA-4_1-Platform)
- [Vitis HLS arbitrary-precision fixed point](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types)
- [XRT native API](https://xilinx.github.io/XRT/2024.2/html/xrt_native_apis.html)

## 9. Hardware blockers

The cards may exist physically, but no authorised or observable execution
surface currently enumerates them. These blockers do not weaken the CPU UPO
proof. They block only the accelerated search for the next certificate.

The live audit used the following surfaces:

```bash
for d in /sys/bus/pci/devices/*; do cat "$d/vendor" 2>/dev/null; done
command -v xbutil; command -v xrt-smi
kubectl get nodes -o custom-columns='NAME:.metadata.name,XILINX:.status.allocatable.xilinx\.com/fpga,AMD:.status.allocatable.amd\.com/fpga'
scontrol show config | awk -F= '/^GresTypes/ {print}'
sinfo -N -o '%N|%P|%t|%G|%f'
beagle hpc profiles
kubectl -n beagle get deploy auth-bridge -o json
kubectl -n beagle get endpointslice -l kubernetes.io/service-name=auth-bridge -o json
```

Observed on the current pod and three running Slurm workers: zero PCI vendors
`0x10ee`/`0x1002`, zero FPGA device nodes, no XRT/Vitis commands, no Kubernetes
FPGA resource or device plugin, and `GresTypes=gpu` only. `beagle hpc profiles`
exited 22 on HTTP 401. The bridge deployment had zero configured, available,
and ready replicas; its EndpointSlice had zero addresses. The fourth Kubernetes
node had no FPGA resource label but no running Slurm worker for direct `/sys`
inspection.

```text
Blocker-ID: BLK-20260728-cs6-u250-resource-absent
Status: classified
Severity: B3
Class: platform-resource
Owner: Hardware/Cluster Ops
Lane: cs6-proof-machine-20260728
Worktree: /tmp/sounio-cs6-proof-machine-20260728
Branch: research/cs6-proof-machine-20260728
Files-Owned: none; external PCI, Kubernetes, Slurm, XRT and device-plugin surfaces
Files-Read-Only: this proof-machine artifact
Do-Not-Touch: primary dirty /workspace/sounio checkout
Repro: enumerate PCI vendor 0x10ee, FPGA devices/resources, XRT, and Slurm GRES on the authorised workers
Observed: no U250 PCI function, device node, XRT tool, Kubernetes FPGA resource, or Slurm fpga GRES is exposed
Expected: two U250 BDFs and an allocatable two-card Kubernetes or Slurm resource
Acceptance-Gate: xbutil examine sees both BDFs/shells and xbutil validate passes on both cards inside an allocated job
Evidence-Level: E2
Evidence: this note, Section 9, live environment audit on 2026-07-28
Fallback-Path: CPU CAPD only; no hardware claim
Legacy-Kept: n/a
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: Hardware/Cluster Ops installs XRT/device plugin and exposes a two-U250 allocation
```

```text
Blocker-ID: BLK-20260728-cs6-cluster-ops-auth-bridge
Status: classified
Severity: B3
Class: platform-resource
Owner: Cluster Ops
Lane: cs6-proof-machine-20260728
Worktree: /tmp/sounio-cs6-proof-machine-20260728
Branch: research/cs6-proof-machine-20260728
Files-Owned: none; Cluster Ops auth-bridge and profile surfaces
Files-Read-Only: this proof-machine artifact
Do-Not-Touch: workspace credentials and bootstrap paths
Repro: beagle hpc profiles; inspect auth-bridge deployment and endpoints
Observed: profiles returns HTTP 401; auth-bridge has zero replicas and no ready endpoint
Expected: an authorised U250 profile returned by a healthy bridge
Acceptance-Gate: auth-bridge has at least one ready replica and endpoint, and beagle hpc profiles lists the U250 profile
Evidence-Level: E2
Evidence: this note, Section 9, live Cluster Ops audit on 2026-07-28
Fallback-Path: CPU CAPD only; restoring auth alone does not create the missing FPGA resource
Legacy-Kept: n/a
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: Cluster Ops restores auth-bridge, then Hardware/Cluster Ops closes the independent resource blocker
```

The missing chaos certificate is an explicit research obligation, not a
platform blocker: locate and validate a homoclinic/covering witness, or publish
the negative result if exhaustive bounded searches rule out the chosen route.

## 10. Reproduce and provenance

Fast receipt and local-enclosure gate:

```bash
bash scripts/ci/cs6_proof_machine_gate.sh
```

Full 296-segment directed-decimal replay:

```bash
CS6_FULL=1 bash scripts/ci/cs6_proof_machine_gate.sh
```

CAPD was built from the official 5.3.0 source archive:

```text
URL: https://sourceforge.net/projects/capd/files/5.3.0/capd-5.3.0.tar.gz/download
tarball SHA-256: e4100959a5409d330f8907d050f101a0485489075b4ce0d5eb2e349a2f8bf228
```

The exact build used GCC/G++ 13.3.0 and:

```bash
../capd-5.3.0/configure --prefix=/tmp/capd-install --cache-file=/dev/null
make -j2 lib
g++ -O2 -std=c++17 scripts/research/cs6_capd_periodic_orbit.cpp \
  -o /tmp/cs6_capd \
  $(/tmp/capd-build/bin/capd-config --cflags --libs)
```

`capd-config` supplied the FILIB headers, static CAPD/FILIB libraries,
`-frounding-math`, `-D__USE_FILIB__`, and `-ffloat-store`. Enabling fast-math
or omitting those generated flags invalidates the receipt.

For a local CAPD installation, replay and byte-compare the frozen receipt:

```bash
CS6_CAPD_REPLAY=1 \
CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config \
bash scripts/ci/cs6_proof_machine_gate.sh
```

Two original runs and one later run exited zero and were byte-identical. The
frozen stdout SHA-256 is:

```text
5ae7a2154204870170639f6075f5292edd00059e095a650726ea4ea1a6c44054
```

The hard-coded `CAPD_VERSION=5.3.0` output line is descriptive, not runtime
discovery. The version anchors are the pinned source-archive hash, compiler
version, and generated backend flags recorded above. Repetition on one machine
checks determinism, not independence.

## 11. Falsifiers and stop rules

| Falsifier | Consequence |
| --- | --- |
| Rebuilding the pinned CAPD source with the generated flags changes a bound enough to fail strict Newton inclusion. | UPO certificate does not reproduce; stop promotion. |
| Any `P^k(X)`, `1 <= k < 6`, intersects `X` in a rigorous replay. | Prime-period-six claim fails; retain only a period dividing six. |
| The Poincare normal velocity enclosure reaches zero. | Return-map orientation/transversality is not proved. |
| Independent review finds the Liouville normal-velocity factor was cancelled away from the fixed point. | Hyperbolicity claim fails; recompute determinant in an aligned basis. |
| A multiplier interval reaches the unit circle. | Hyperbolicity is unproved. |
| FPGA and CPU disagree on one duplicated or accepted tile. | Reject the full FPGA batch; hardware cannot promote evidence. |
| A homoclinic candidate cannot pass interval Newton or transversality. | It is reconnaissance only; subdivide or abandon that candidate. |
| A horseshoe is proved without an attracting neighbourhood. | Claim positive entropy, not a chaotic attractor. |

## 12. Review ledger

- CAPD adversarial audit: `PASS`. It independently checked interval Newton,
  Poincare iteration semantics, prime period, Liouville's normal-velocity
  factor, local-only uniqueness, and the hyperbolicity boundary.
- `xai/grok-4.3`, `math-review`: `PASS`. All eight audited mathematical and
  executable claims were marked `[OK]`.
- `zai/GLM-5.2`, `math-review`: `PASS`. It independently rederived the
  divergence, determinant identity, multiplier bounds, Taylor recurrences,
  Picard inclusion, Fibonacci spectral radius, and FPGA monomial count.
- Raw dual-provider review: `/tmp/llm-offload-VnzU67/`.
- Focused second round: `xai/grok-4.3` and `zai/GLM-5.2` both passed the
  Fibonacci entropy bound, reversed-flow spectrum/saddle quantity, and replay
  normalisation. Raw: `/tmp/llm-offload-4AmKqu/`.
- Focused spectrum round: both providers directly expanded the displayed
  Jacobian characteristic polynomial and passed the eigenvalues, reversed-flow
  saddle quantity, and no-attractor boundary. Raw: `/tmp/llm-offload-VzvKN4/`.

## 13. AI disclosure

The numerical reconnaissance, interval contracts, CAPD adaptation, hardware
partition, and research note were produced under human direction on
2026-07-28. The CAPD Newton pattern derives from the official periodic-orbit
example. Mathematical claims require the repository's mandatory independent
LLM-offload review before commit; the dual-provider review above passed. No
clinical content is present.
