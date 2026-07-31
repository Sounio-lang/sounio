# CS6 full-source C1 cone and Liouville proof machine

**Date:** 2026-07-30
**Lane:** `cs6-c1-cone-20260730`
**Base:** `21c29183c396288b79ef4cc93fe966f5b3b6d63b`
**Status:** all four proof components are executable and fail closed; bounded
CAPD probes and adversarial gates pass; the exhaustive C0+C1 replay has not run

## 1. Result boundary

This wave implements the four requested components for the frozen CS6
Fibonacci construction:

1. a CAPD C1 enclosure of the derivative of exactly six `MinusPlus` Poincare
   returns over a complete, sharded source partition;
2. an exact-rational global-hull Sylvester test for all three edges;
3. a separate augmented-divergence Liouville certificate that retains the
   section-normal velocity ratio away from the fixed point;
4. a combined aggregator that refuses promotion unless retained exhaustive C0
   raw records freshly reconstruct the supplied Fibonacci certificate.

The machinery exists and its bounded tests pass. The full-source C1 partition
has not been calibrated to a tractable size, the authorized Foundry/Slurm
submission surface remains unavailable, and no exhaustive C0 certificate has
been produced. The repository certificate therefore remains:

```text
full_source_c1_derivative_enclosure_proved = false
global_full_source_hull_tested = false
pairwise_chord_cone_condition_proved = false
liouville_invertibility_proved = false
combined_c0_c1_mathematical_evidence_complete = false
combined_c0_c1_execution_provenance_attested = false
uniform_hyperbolicity_proved = false
chaotic_attractor_proved = false
```

The prepromotion receipt is
`scripts/research/cs6_capd_c1_cone_certificate_v1.txt`.

## 2. C1 contract

Let `D=[-1,1]^2`, `F=P^6`, and

```math
chi_i(xi) = p_* + B(c_i + R_i xi),
```

where `R_i=diag(r_ui,r_si)`. For edge `i->j`, the normalized derivative is

```math
A_ij(xi) = R_j^{-1} B^{-1} DF(chi_i(xi)) B R_i.
```

The worker uses `C1Rect2Set` with a structured C0 source parallelogram. Its C1
initial matrix carries `B R_i` in the two tangent columns and zero in the
irrelevant normal column. CAPD propagates those columns through all six returns.
The raw variational matrix is not accepted: `computeDP` must apply the return
time correction before the top-left section derivative is normalized.

Every accepted ledger record also requires:

- finite C1 and C0 enclosures;
- six strictly ordered return-time enclosures;
- positive normal velocity at crossings `0,...,6`;
- a positive Liouville determinant enclosure;
- overlap between the C1 total return time and Liouville's sixth return;
- consistency of the serialized `exp(ell_6) nu_0/nu_6` operands;
- overlap between the normalized C1 determinant and the radius-scaled
  Liouville determinant;
- exact source indices and an outward enclosure of the canonical source tile.

The worker never promotes a global claim. It emits tile enclosures and keeps all
proof fields false. Only the aggregator can verify that every source tile is
present exactly once for every required edge.

## 3. One global hull, not tilewise wishful thinking

For each edge, the aggregator forms one entrywise hull containing every
normalized tile derivative:

```math
[A_ij] = hull_T [A_ij(T)].
```

This is stronger than checking the cone separately on each tile. The domain
`D` is convex, so the derivative average along a segment between points in
different tiles lies in the global convex entrywise hull. A tilewise pass alone
does not control those cross-tile chords.

For

```math
A = [[p,q],[r,s]],
Q_i = diag(a_i,-b_i),
Q_j = diag(a_j,-b_j),
```

the independent entries of `M=A^T Q_j A-Q_i` are

```math
m00 = a_j p^2 - b_j r^2 - a_i,
m01 = a_j p q - b_j r s,
m11 = a_j q^2 - b_j s^2 + b_i.
```

The straightforward interval determinant `m00*m11-m01^2` repeats dependency
terms and was a major source of false negatives. This wave uses the exact
expanded identity

```math
det(M) = a_j b_i p^2 - b_j b_i r^2
       - a_i a_j q^2 + a_i b_j s^2
       - a_i b_i - a_j b_j (p s-q r)^2.
```

The identity cancels `p^2 q^2` and `r^2 s^2` symbolically before interval
evaluation. On the bounded N0 center witness, the naive lower bound is
`-5.5192522934`, while the exact expanded form gives `+5.2100640574` for the
same outward derivative box. This is not weaker arithmetic or a numerical
approximation; it is a different interval extension of the same polynomial.

The aggregator parses the frozen decimal Q weights as exact rational numbers
and every CAPD endpoint as exact binary64. The C++ worker expands endpoints one
ULP outward, serializes them in hexadecimal, and the Python process converts
them to exact `Fraction` values. The authoritative predicate is

```text
inf(m00) > 0 and inf(det_expanded) > 0
```

for the single full-source hull of each edge. Zero fails. The naive determinant
is retained as a diagnostic. A positive expanded determinant and first leading
minor prove positive definiteness and hence the pairwise chord condition in
Definition 11 and Lemma 8 of Zgliczynski's construction.

Primary references:

- [Zgliczynski, covering relations, cone conditions and the stable manifold theorem](https://doi.org/10.1016/j.jde.2008.12.019)
- [Author manuscript with Definition 11 and Lemma 8](https://ww2.ii.uj.edu.pl/~zgliczyn/papers/invman/cncv.pdf)
- [CAPD rigorous Poincare derivative interface](https://capd.sourceforge.net/capdDynSys/docs/html/a05237.html)

## 4. Liouville without cancelling the wrong factor

In `w=z-zs` coordinates,

```math
div f = x-y-(w+zs)/2-1,
nu(x,y,w) = n dot f = xy-w-zs.
```

The second CAPD system appends

```math
ell' = div f,  ell(0)=0.
```

For a general source point, the oriented determinant of the sixth-return map
is

```math
det DP^6 = exp(ell_6) * nu_initial / nu_final.
```

The velocity ratio equals one at a fixed point and can also equal one by
coincidence elsewhere, but it is generally non-unit away from the orbit.
Omitting it over the complete h-sets would be wrong. The ledger records
`ell_6`, `nu_0,...,nu_6`, all six return times, `exp(ell_6)`, and the final
determinant interval. It requires every normal velocity and the determinant
lower bound to be strictly positive.
The exact-rational aggregator reconstructs the interval expression
`E_6 nu_0/nu_6` from the separately outward-expanded operands, where `E_6` is
the CAPD-produced enclosure of `exp(ell_6)`, and requires it to contain the
serialized determinant. It does not independently recompute the transcendental
map `ell_6 -> exp(ell_6)`; that operation remains inside the retained, hashed
CAPD worker and is stated as such in the certificate.

This path resolves determinants near `10^-34` without subtracting the
cancellation-dominated entries `ad-bc` in ordinary double precision. In
normalized charts,

```math
det A_ij = det DP^6 * det(R_i)/det(R_j),
```

because the frame determinant cancels and all radii are positive.

For every ledger record, the aggregator also requires the interval determinant
`A00*A11-A01*A10` to overlap that scaled Liouville enclosure. Separately, the
C1 total return-time enclosure must overlap Liouville's sixth-return enclosure.
These two checks bind the independently integrated C1 and Liouville systems to
the same branch and determinant; positivity alone is not accepted. The
determinant overlap is a necessary consistency check between independent
enclosures, not an independent proof of equality or of execution provenance;
neither enclosure is assumed to contain the other.

## 5. Combined C0+C1 evidence

The C1 runner performs early rejection unless it receives both a retained raw
C0 run directory and its aggregate certificate of kind
`CAPD_RIGOROUS_COVERING_AGGREGATE_V1`. It reruns the canonical local C0
aggregator and requires byte-identical certificate output before compiling C1.
The combined aggregator repeats that reconstruction from the copied raw bundle,
requires the retained C0 aggregator to equal the canonical code byte for byte,
and checks the freshly reconstructed canonical-ledger hash. A forged certificate
text is therefore insufficient.

Only then does it accept the full C0 contract: all `42,825` ledger records, the
three directed coverings, Fibonacci adjacency, map, section, `zs`, origin,
frame, h-sets, grid, order, legacy declared trust metadata, and exact local C0
proof-source hash.
The C1 shards independently emit their dynamical-system preamble from the same
constants used to construct the CAPD maps. Those values, including both
vector-field strings, are compared exactly before C0 and C1 may be combined.

The C1 manifest hashes the retained C0 certificate and aggregator. It also binds
the explicit CAPD 5.3.0/FILIB artifacts observed by this runner: pkg-config
record, all regular files below emitted `-I` roots, every existing library path
emitted by `--libs`, the compiler-driver bytes checked immediately around
compilation, and every runtime object reported by `ldd`. Each checksum manifest
must equal the corresponding argument or linkage set, not merely contain
hash-shaped values. This is an explicit artifact set, not a claim to capture the
whole host trusted computing base.

The runner also requires `scontrol` to report an active same-UID allocation
whose node list includes the current execution node. This is a consistency
check on locally observed scheduler records. It does not attest process/cgroup
membership, Foundry or Beagle submission origin, remote hardware, or the bytes'
physical provenance. Authorized submission remains an operational precondition,
and independent replay remains required.

The inherited C0 certificate's
`AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION` value is legacy declared
metadata. Fresh reaggregation proves the mathematical consistency of its raw
bytes, not the authorization assertion. The combined certificate therefore
emits it only as `C0_EXECUTION_TRUST_MODEL_DECLARED` and keeps
`C0_EXECUTION_PROVENANCE_VERIFIED=false`. Synthetic gate data can exercise this
boundary but cannot turn it into execution provenance.

After those bindings, the aggregator validates the exact C1 partition, global
cone hulls, and Liouville records before it may emit:

```text
fibonacci_coverings_proved = true
positive_entropy_proved = true
pairwise_chord_cone_condition_proved = true
tangent_cone_condition_proved = true
liouville_invertibility_proved = true
combined_c0_c1_mathematical_evidence_complete = true
combined_c0_c1_execution_provenance_attested = false
```

Even then it deliberately emits

```text
uniform_hyperbolicity_proved = false
chaotic_attractor_proved = false
flow_entropy_bound_proved = false
```

Lemma 8 supplies the two-point cone inequality, not by itself a named compact
invariant cone-field theorem. A later promotion of uniform hyperbolicity must
bind that theorem or include its proof, including invertibility of the return
map on the invariant set. A trapping region and basin argument remain separate
obligations for any attractor claim.

## 6. Executable evidence

The static and synthetic adversarial gate passed:

```bash
bash scripts/ci/cs6_capd_c1_cone_gate.sh
```

It exercises:

- exact rational equality of naive and expanded determinant forms on singleton
  matrices;
- a dependency witness where only the expanded interval form succeeds;
- every key in a complete synthetic two-shard source partition;
- global rather than tilewise Sylvester promotion;
- fresh reconstruction from a self-contained, symlink-free C0 raw bundle, plus
  exact C0/C1 dynamical-system and C0 source binding;
- exact CAPD header, static-library, and runtime-linkage artifact sets;
- per-shard mathematical counts and modulo ownership;
- output-overwrite refusal;
- rejection of a widened global hull, relabelled source tile, zero Liouville
  determinant, disjoint C1/Liouville final returns, inconsistent Liouville
  formula operands, inconsistent normalized determinants, false or foreign-map
  C0 certificate, tampered or symlink-backed C0 raw record, drifted C1 `zs`,
  forged shard counts, swapped shard records, non-binary64 hexadecimal input,
  and an unrelated library manifest;
- refusal to run outside Slurm or with an unverified `SLURM_JOB_ID`;
- all inherited C0, UPO, Fibonacci, and numerical-cone gates.

The bounded live CAPD replay also passed:

```bash
CS6_CAPD_C1_SAMPLE_REPLAY=1 \
CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config \
bash scripts/ci/cs6_capd_c1_cone_gate.sh
```

Selected order-8 probes with the current tangent initialization:

| Probe | Result |
|---|---|
| N0->N0 center, absolute grid `40000 x 30000` | C1 valid; `m00_lo=2.33037`; naive `det_lo=-5.51925`; expanded `det_lo=5.21006`; Liouville positive |
| N0->N0 right-top, `40000 x 30000` | C1 valid; `m00_lo=-1.00703`; cone rejected and probe exits nonzero |
| N0->N0 right-top, `80000 x 40000` | cone diagnostic passes; expanded `det_lo=3.19603` |
| N1->N0 center, `15000 x 30000` | C1 valid; `m00_lo=-0.02137`; cone rejected |
| N1->N0 center, `30000 x 60000` | cone diagnostic passes; expanded `det_lo=1.25523` |
| inherited C0 cell scale `N0 200 x 75` | CAPD rejects a possible nontransversal return after wrapping |

The live gate now exercises all three directed edges. Its N0->N0 probe writes
a real worker ledger record; the aggregate module parses the exact hexadecimal
endpoints, checks the complete schema, and revalidates its canonical source
tile. The selftest checks the exact one-ULP neighbors of binary64 `1.0`, rather
than merely looking for a hexadecimal marker.

These are bounded falsification/calibration points, not a cover of either
source h-set. They show both that the exact determinant identity changes
feasibility and that a naive uniform regular grid remains computationally
unacceptable. Representative worst-location resolutions extrapolate to
billions of tiles; no such run was attempted locally.

Orders `12`, `16`, and `20` did not improve the representative order-8 C1
enclosures. `C1HORect2Set`, six separate calls on the same set, and reboxing
between return blocks also failed to produce a tractable complete partition.
The reboxing experiment isolated the loss: a tiny source tile grew to roughly
`2.47e-4 x 8.98e-5` in local radii after three `P^2` blocks.

## 7. Next scaling experiment

The next high-value implementation is a chart-chain that preserves the C0 set
and resets only the C1 matrix doubleton after each return:

1. propagate the unstable chart direction forward;
2. propagate the stable chart direction backward from return six;
3. call one rigorous return on the same mutable C0 set;
4. record the corrected local Poincare derivative;
5. recondition only the C1 matrix representation in the next chart;
6. compose the six small interval matrices externally.

Point probes in backward-stable `P^2` charts reduced the off-diagonal derivative
enclosures to approximately `|q|<=4.19e-9` and `|r|<=3.51e-6`. Reboxing C0
destroyed that gain, so the next prototype must preserve the predictor/corrector
state and synchronize every internal C1 enclosure member. Resetting only one
matrix field would be unsound.

The two AMD U250 boards could later schedule independent accepted leaves or
evaluate preconditioner candidates. They are not required, not installed for
this replay, and not part of the proof trusted computing base. Outward-rounded
CPU CAPD and exact aggregation remain authoritative.

## 8. Execution paths

```text
default Sounio compiler path used = false
rebuilt current-source CAPD path used = true
fallback path used = false
legacy numerical cone scout kept = true
legacy C0 proof machine kept = true
same-UID active Slurm record including execution node required by runner = true
Slurm process membership attested by runner = false
Beagle authorization inferred from Slurm environment = false
remote exhaustive C0 replay run = false
remote exhaustive C1 replay run = false
remote attestation present = false
```

The primary checkout remained untouched. All writes were isolated in
`/tmp/sounio-cs6-c1-cone-20260730`.

## 9. Semantic lane declaration

```text
Semantic-Lane-ID: cs6-c1-cone-20260730
Owner: codex-1
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: machine evidence remains weaker than the scientific claim until every global proof obligation is executable and green
Transformation: add a fail-closed CAPD C1, global cone-hull, Liouville, and C0-composition proof machine
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a replayable machine now exists for full-source chord cones and branch-bound Liouville invertibility
Claims-Forbidden: completed C1 proof; uniform hyperbolicity; chaotic attractor; trapping region; flow entropy; first-proof priority
Assumptions: CAPD 5.3.0 FILIB outward intervals; frozen CS6 map/frame/h-sets/Q forms; exact complete ledger required
Write-Set: scripts/research/cs6_capd_c1_cone.cpp; scripts/research/cs6_capd_c1_cone_run.sh; scripts/research/cs6_capd_c1_cone_aggregate.py; scripts/research/cs6_capd_c1_cone_certificate_v1.txt; scripts/ci/cs6_capd_c1_cone_gate.sh; docs/research/cs6_capd_c1_cone_2026-07-30.md; .claude/llm_offload_log.md
Read-Set: prior CS6 UPO, Fibonacci C0, and numerical cone artifacts; CAPD source/examples; Zgliczynski cone reference
Positive-Witness: static adversarial gate and bounded live CAPD replay pass
Negative-Witness: C0-scale C1 tile wraps to nontransversality; singular map passes the cone but is rejected as invertibility evidence
Acceptance-Gate: CS6_CAPD_C1_SAMPLE_REPLAY=1 CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config bash scripts/ci/cs6_capd_c1_cone_gate.sh
Integration-Target: review branch only; no main merge requested
Authoritative-Only-If: real exhaustive C0 raw bundle reconstructs its certificate and a complete full-source C1 ledger passes the aggregate gate; execution provenance remains independently unaudited

Semantic-Outcome: all four proof components are executable without promoting an unrun theorem
Concept-Status-Before: numerical cone forms existed; full-source C1 and general-point invertibility machines did not
Concept-Status-After: proof machinery and exact predicates exist; full replay remains an evidence gap
Distinctions-Added: tile cone diagnostic vs global chord condition; naive vs dependency-reduced interval determinant; fixed-point vs general Liouville determinant; mathematical byte consistency vs execution provenance
Distinctions-Preserved: numerical candidate != theorem; local probe != full-source cover; cone condition != invertibility; hyperbolic saddle set != attractor
Distinctions-Erased: none
Evidence-Run: synthetic exact-partition aggregate; mutation negatives; CAPD selftest; bounded C1/Liouville probe; inherited CS6 gates
Fallback-Path: none
Legacy-Kept: yes
Conflicting-Lanes: none reported by bin/sounio-coord brief
Next-Semantic-Interface: C1-only chart reset preserving the mutable C0 set, then authorized full replay
```

## 10. Open blocker records

```text
Blocker-ID: BLK-20260730-cs6-c1-full-source-scaling
Status: classified
Severity: B3
Class: evidence-gap
Owner: cs6-rigorous-c1-cone
Lane: cs6-c1-cone-20260730
Worktree: /tmp/sounio-cs6-c1-cone-20260730
Branch: research/cs6-c1-cone-20260730
Files-Owned: scripts/research/cs6_capd_c1_cone.cpp; scripts/research/cs6_capd_c1_cone_run.sh; scripts/research/cs6_capd_c1_cone_aggregate.py; scripts/research/cs6_capd_c1_cone_certificate_v1.txt; scripts/ci/cs6_capd_c1_cone_gate.sh; docs/research/cs6_capd_c1_cone_2026-07-30.md; .claude/llm_offload_log.md
Files-Read-Only: prior CS6 C0 and numerical cone artifacts
Do-Not-Touch: prior promoted certificates or claim fields without complete aggregate evidence
Repro: /tmp/cs6_capd_c1_cone probe N0 N0 99 37 200 75 8
Observed: the inherited C0 cell scale wraps until CAPD cannot prove a transverse sixth return; bounded passing C1 cells imply an intractable uniform-grid extrapolation
Expected: a complete source partition produces finite C1 enclosures whose single global edge hull passes expanded Sylvester
Acceptance-Gate: python3 scripts/research/cs6_capd_c1_cone_aggregate.py --run-dir <real-run-with-retained-c0-raw> --shards <n> --source scripts/research/cs6_capd_c1_cone.cpp --ledger-output <fresh-ledger> --certificate-output <fresh-certificate>
Evidence-Level: E3
Evidence: scripts/research/cs6_capd_c1_cone_certificate_v1.txt and the named bounded replay gate
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: dual-pass-addressed
Next-Action: prototype a synchronized C1-only chart reset that preserves the C0 predictor/corrector set
```

```text
Blocker-ID: BLK-20260728-cs6-cluster-ops-auth-bridge
Status: classified
Severity: B3
Class: platform-resource
Owner: Cluster Ops
Lane: cs6-c0-c1-authorized-replay
Worktree: /workspace/sounio
Branch: research/self-falsifying-compilation-line-20260726
Files-Owned: none
Files-Read-Only: scripts/research/cs6_capd_c1_cone_run.sh; scripts/research/cs6_capd_fibonacci_covering_run.sh
Do-Not-Touch: auth-bridge replicas, tokens, contexts, or manual cluster YAML from the workspace
Repro: beagle hpc profiles
Observed: HTTP 401; beagle/auth-bridge is 0/0 with no endpoints or pods
Expected: an authorized HPC profile is returned
Acceptance-Gate: beagle hpc profiles exits zero, followed by both frozen aggregate gates
Evidence-Level: E4
Evidence: current Cluster Ops deployment/endpoints/pods queries
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: Cluster Ops restores auth-bridge; replay owner submits through the authorized profile
```

```text
Blocker-ID: BLK-20260730-cs6-uniform-hyperbolicity-theorem
Status: classified
Severity: B3
Class: evidence-gap
Owner: cs6-dynamics-theorem
Lane: future-cs6-uniform-hyperbolicity
Worktree: /tmp/sounio-cs6-c1-cone-20260730
Branch: research/cs6-c1-cone-20260730
Files-Owned: none in this lane after handoff
Files-Read-Only: future aggregate C0+C1 certificate; Zgliczynski cone reference
Do-Not-Touch: UNIFORM_HYPERBOLICITY_PROVED until a named theorem or complete proof is bound
Repro: grep -Fx UNIFORM_HYPERBOLICITY_PROVED=false scripts/research/cs6_capd_c1_cone_certificate_v1.txt
Observed: the current aggregate intentionally stops at covering, chord/tangent cone, and invertibility evidence
Expected: a compact invariant cone-field theorem is explicitly instantiated, including return-map invertibility on the invariant set
Acceptance-Gate: future theorem-specific review and executable certificate gate
Evidence-Level: E2
Evidence: this result-boundary document and mandatory math review
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: dual-pass-addressed
Next-Action: bind a precise strong-cone theorem only after the real C0+C1 aggregate exists
```

## 11. LLM-offload reviews

The mandatory M1 review ran on the final C++ worker, exact-rational aggregator,
and result-boundary document with xAI/Grok 4.3 and Z.AI/GLM-5.2. Both providers
confirmed the cone algebra, Liouville formula, chart determinant scaling,
Sylvester predicate, and non-promotion boundary. Z.AI caught the overstrong
claim that a unit normal-velocity ratio occurs only at a fixed point; section 4
now states the correct general boundary. Its suggestion to require containment
between the independent C1 and Liouville determinant enclosures was not adopted:
intersection is necessary, while neither containment direction follows from two
sound enclosures. The executable source binding and unattested-provenance limit
remain explicit.
