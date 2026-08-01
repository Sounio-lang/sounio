# CS6 fixed-chart H-APG full53: a positive bounded population result

**Date:** 2026-08-01
**Lane:** cs6-affine-projective-full53-20260801
**Contract freeze:** 58905019754cf66f077a0db228f1a99a4a7612eb
**Implementation:** c5e99db178711081f158020b239488dc7f7f8e02
**Status:** the predeclared 53-attempt fixed-chart H-APG criterion passes
on every parent-computable coordinate, with zero losses on the 28 parent-affine
obligations and 20 rescues among 24 parent-nonaffine coordinates; this is local
bounded CAPD CPU evidence for a candidate validated numerical primitive, not
full-source coverage, hyperbolicity, chaos, world-priority, or an open-problem
solution

## 1. Frozen question and answer

The four-leaf pilot in
docs/research/cs6_affine_projective_cocycle_2026-08-01.md supported the
shared-source affine-projective carrier H-APG, but it used selected witnesses.
Before this run, commit 589050197 froze a manifest containing all 53
coordinates from the retained parent H-PG experiment:

~~~text
selection                       all parent coordinates, no post-hoc filter
attempts                        53
parent-computable pairs         52
expected root exception          1
parent-affine obligations       28
parent-nonaffine opportunities  24
chart policy                    parent chart/sign tuple fixed per leaf
~~~

The acceptance criterion was also frozen before implementation results:

1. attempt all 53 coordinates;
2. reproduce the exact interval-domain class at U00/S00;
3. obtain 52 paired-valid non-root computations with no new unresolved leaf;
4. produce a valid APG computation on every paired leaf;
5. lose no parent-affine obligation, where a loss means failure to obtain a
   valid APG certificate on one of the 28 parent-affine leaves;
6. rescue at least one of the 24 parent-nonaffine leaves;
7. replay every receipt exactly and reject every frozen mutation;
8. keep promotion false without independent attestation.

The retained answer is:

~~~text
coordinate attempts                         53
paired-valid computations                    52
root interval-domain class match           true
new unresolved leaves                         0
valid APG computations                       52
parent-affine obligations                     28
parent-affine obligations preserved           28
parent-affine obligation losses                0
APG certificates                              48
parent-nonaffine opportunities                24
APG rescues on parent-nonaffine leaves        20
full53 predeclared criterion                true
promotion eligibility                      false
~~~

The full53 criterion therefore passes. This wording is deliberately narrower
than saying that every leaf is orientation-certified: four parent-nonaffine
U08 x S08 leaves remain APG-uncertified, while all four APG computations are
structurally valid and strictly narrower than the three comparison carriers.

## 2. What the machine actually computes

The worker is the pilot's common-source order-2 Taylor machine extended to the
entire parent population. It retains the same two source symbols through the
event-1 state, local return, projective normalization, tangent action, and
signed exterior reconstruction. The primary result is

~~~math
D_{APG} = (p_{10}p_{20})(p_{11}p_{21})
          det(u_{20},u_{21}).
~~~

For each of the four event/ray positions, the chart and expected pivot sign are
copied from the anchored parent receipt before execution. The child must replay
that exact ordered tuple. A pivot whose complete dependent range does not have
the expected strict sign makes the leaf unresolved; the worker cannot select
another chart or fall back to an interval box.

Chart freezing is a protocol invariant for this falsifier, not an algebraic
theorem that the parent chart is optimal for the child. It prevents favorable
runtime reselection from changing the hypothesis after coordinates are known.
The worker and verifier support all four frozen charts X, Y, PLUS, and MINUS,
both pivot signs, and positive E2 pivots.

The root is outside the paired leaf verifier. Its exact two-line CAPD/FILIB
division-by-zero signature, empty receipt, and return code 1 are checked
separately. Any format drift, unexpected successful root computation, timeout,
new chart failure, or other worker failure is retained as a negative result and
makes the full53 support field false.

## 3. Numerical result

The certificate counts separate the old affine carrier from the new H-APG
carrier:

| Population | Affine certificate | APG certificate | APG rescue |
|---|---:|---:|---:|
| 28 parent-affine obligations | 28 | 28 | 0 by definition |
| 24 parent-nonaffine leaves | 0 | 20 | 20 |
| 52 paired leaves | 28 | 48 | 20 |

The four APG-uncertified coordinates are:

~~~text
U08-0000000064_S08-0000000064
U08-0000000064_S08-0000000192
U08-0000000192_S08-0000000064
U08-0000000192_S08-0000000192
~~~

All 52 APG intervals are strictly narrower than both the boxed H-PG and affine
intervals. Thirty-seven are strictly narrower than the shared-source
nonprojective TM2 control.

| APG width divided by | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| boxed H-PG | 2.10109e-5 | 5.06276e-5 | 1.33566e-4 | 1.32698e-3 |
| affine | 0.0260800 | 0.0682823 | 0.0782715 | 0.195040 |
| shared nonprojective TM2 | 0.527074 | 0.739269 | 0.888516 | 1.59927 |

The minimum certified APG pivot margin is
1.7966009779136807e-4. Aggregate worker time is 430,521 ms;
mean paired-valid worker time is 8,261.673 ms; mean receipt size is
79,750.35 bytes.

The sharp bounded conclusion is:

1. retaining common state/tangent source symbols through the fixed-chart
   projective composition preserves every affine-positive parent obligation;
2. the same machine certifies orientation on 20 of the 24 affine-negative
   parent coordinates;
3. projective factorization is not uniformly narrower than the same
   shared-source nonprojective composition, so the result does not isolate
   projective geometry alone as the cause.

## 4. Evidence envelope

The retained bundle contains 237 run payloads and two enclosing files:
files.sha256 and retained-manifest.txt. It includes:

- source, runner, leaf verifier, coordinate contract, compiler and CAPD flags;
- compile dependencies, link inputs, runtime libraries, and worker-binary hash;
- 53 inputs, 53 receipts, 53 stderr payloads, and 52 exact verifications;
- one mutation result per paired leaf and the 52-row mutation index;
- exact summary, run contract, clean Git state, and implementation commit.

The leaf-verifier suite contains 112 receipt/contract/challenge mutations.
Every suite was run on every paired leaf:

~~~text
mutation-audited leaves    52
mutations per leaf        112
mutation tests          5,824
mutations rejected      5,824
~~~

The retained verifier independently:

- reconstructs all 53 coordinates and the 28-leaf obligation set from the
  anchored parent bundle;
- re-derives all 208 chart/sign fields from parent homogeneous receipts;
- checks the frozen challenge preimage and all manifest bindings;
- replays all 52 leaf verifications byte for byte;
- reruns all 5,824 mutations and checks each audit digest;
- recomputes every summary statistic, including honest negative outcomes.

The gate then rejects nine coordinated retained-tree mutations: chart, sign,
chart plus challenge rebinding, parent-affine obligation, leaf method, root
class, mutation aggregate, extra file, and symlink path. Its fresh mode rebuilds
the CAPD worker and exactly replays two leaves covering all four charts and a
positive E2 pivot, plus the canonical root failure. All checks pass.

## 5. Integrity anchors

~~~text
predeclaration commit       9dcf1fca964d7e54e1109f9210689809666b2a54
contract-freeze commit      58905019754cf66f077a0db228f1a99a4a7612eb
implementation commit       c5e99db178711081f158020b239488dc7f7f8e02
coordinate manifest SHA256  61b2b0649983a332b5abb530443a3ff14a19e62514ef9b1d3175d8e9a6bbfd9c
worker source SHA256         57547b10911354fac05f35cbc301125c30f6b88e5e85559449964ed32bf0f3a7
leaf verifier SHA256         8407301a0aaff85005347193543e65c906043f940f2327afa75747c29af3e910
runner SHA256                3dc7c1d5fc96e487913718c83cdbc09a0d736b08c73f51b9d027fa51ead989ab
retained verifier SHA256     37a00a22bbb247c42f49bee87865f31266ca841118fc89d33f1b4ccb4e2ffce0
run manifest SHA256          f00976d9efbb6119793f78bb680620aa93dbb27a371c661643c5e6ec4bc6d53a
files index SHA256           6c43f37a2b9dfd6bc0de4abafcfe21dedc7329866ea2a95202e4255bb0fc181f
retained manifest SHA256     b6cfb0d70f1d8d609362259fb4170aee129b0939dbdaccefa22638e0c6132b84
summary SHA256               c0232382d019a8124fb58845753a4bb8708d099fd40e57c25e6786dbc1bf8442
leaves SHA256                db4b9bb2637cccec7ee9f4912007d84aeddbee02bba80422e4ca6d3c5faabe3a
mutation audits SHA256       81faed35d8c40d772974d679790cb035e1c9d3ea9997cd344b703ca06b76aeb9
~~~

The run challenge is
d1562dd661a6a925cb250d1635d4b20ef8cfcfe2bd60442a346c25740f0037d8.

## 6. Claim boundary

This run establishes a bounded machine result on the exact frozen parent
population. It does not establish:

- a partition or exhaustive cover of the CS6 source domain;
- a continuous Riccati or general Grassmann-Plucker integrator;
- a validated augmented ODE carrying projective rays as CAPD state;
- hyperbolicity, a chaotic attractor, or a Shilnikov connection;
- independent execution provenance or remote attestation;
- a world-novel method, priority claim, or solution of an open problem.

The 53 parent coordinates are a retained diagnostic population and include
nested leaves. Their areas cannot be summed into certified source coverage.
The literature boundary remains the one documented in the pilot report:
validated Poincare derivatives, affine dependency, projective/Grassmann chart
flows, and exterior-scale separation exist separately. This run supplies
stronger evidence for the candidate integrative novelty window, but no new
literature search or priority claim is inferred from a positive corpus.

No U250 was installed or used. The evidence class is
LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION; promotion remains false.

## 7. Reproduction

The fresh run was:

~~~bash
python3 scripts/research/cs6_affine_projective_cocycle_full53_run.py \
  --capd-config /tmp/capd-build/bin/capd-config \
  --run-dir /tmp/cs6-affine-projective-full53-run-c5e99db1 \
  --root-challenge d1562dd661a6a925cb250d1635d4b20ef8cfcfe2bd60442a346c25740f0037d8 \
  --jobs 8 --timeout-seconds 300 --keep-failed
~~~

Durable verification and adversarial fresh replay are:

~~~bash
python3 scripts/research/cs6_affine_projective_cocycle_full53_retained_verify.py \
  scripts/research/receipts/cs6_affine_projective_cocycle_full53_retained_53_v1

CS6_AFFINE_PROJECTIVE_COCYCLE_FULL53_REPLAY=1 \
CS6_CAPD_CONFIG=/tmp/capd-build/bin/capd-config \
bash scripts/ci/cs6_affine_projective_cocycle_full53_gate.sh
~~~

Both commands pass, with 52/52 exact leaf replays,
5,824/5,824 mutation rejections, two fresh paired replays, and the fresh
root-class replay.
