<!-- docs:meta
topic_id: repo.docs.spec.s17-conformance-suite
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s17-conformance-suite
-->

# §17 — Conformance suite

Spec-Section: `SOUNIO-SPEC-17`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **undefined.** No normative statement has been ruled. This records the
measured state, which the frame previously marked `contested` on the strength of
a single fact. Three more have since been measured and they change the shape of
the problem.

## 17.1 The suite does not run the canonical compiler

`bin/souc` routes to Madaros. The Full Test Suite runs `souc-stage2` — that is
**lean_single**, the bootstrap seed (`#1964`, `#1978`).

`#1978` measured the whole workflow, job by job:

| jobs | engine |
|---|---|
| 2 | Madaros, built from the PR's own source in-job |
| 4 | lean_single — **including the Full Test Suite** |
| 1 | the **stale tracked prebuilt** `bin/madaros-linux-x86_64` |
| 5 | no Sounio engine at all |

The wrapper's lean_single-fallback notice appears in **0 of 12** job logs, so
this is not silent degradation. Each job's engine was chosen.

## 17.2 The price, measured

`#1985`, both arms over the same archived source tree, same harness, same 3,012
file selection:

| lean_single | Madaros | tests |
|---|---|---:|
| green | green | 1,090 |
| **green** | **not green** | **437** |
| not green | green | 27 |
| not green | not green | 132 |
| skipped by both | skipped by both | 1,326 |

**437 of the 1,527 tests CI counts as green fail under source-built Madaros —
28.6%.** The inverse is larger by execution outcome: 55 tests fail beneath an
`xfail`/`vxfail` on lean_single but execute successfully on Madaros; the strict
table shows 27 because an `xpas` is deliberately not green in this harness.

## 17.3 Part of the gap is not disagreement — it is absence

`#2016`: `self-hosted/compiler/lean_single.sio` contains `Knowledge` 118 times,
`measure` 66, `variance_of` 5, `ExactlyPrivate` 3 — and `KnowledgeTypeInfo`,
`EpsilonBound`, `ValidityCondition` and `AstProvenanceKind` **zero times each**.

The seed knows `Knowledge<T>` as a **bare** constructor. It has no
annotation-component machinery at all.

So a test exercising an annotated `Knowledge` type **cannot disagree between the
engines**. It can only fail to parse on one. Part of what the suite does not test
is not a difference in result; it is a **language surface one engine does not
have** — and it is the surface the epistemic core of the language is expressed
in.

## 17.4 Why this is not merely an implementation lag

A conformance suite is the artefact that makes a specification falsifiable. A
suite that runs an engine other than the canonical one produces greens that are
true statements about a different object. Under `SOUNIO-GATING-ENGINE` a green
must name its engine; §17 is where that requirement becomes structural rather
than editorial.

Until it does, every other section of this specification inherits the problem:
a conformance test written for §6, §7 or §8 and run by CI is, today, a test of
the seed.

## 17.5 Rulings owed

- **Which engine is normative for conformance?** Madaros is the canonical
  compiler by `CLAUDE.md`; the suite runs the seed. One of the two must change.
- **What is the status of a test that only one engine can parse?** It is neither
  pass nor fail nor divergence. The suite has no vocabulary for it, and 17.3
  shows the class is not empty.
- **Does the fixed point bind conformance?** `make build` verifies the fixed
  point over `lean_single.sio`, not over Madaros. If conformance moves to
  Madaros, the suite's authority stops resting on the fixed point and must rest
  on something stated here.

## Claims forbidden

- Quoting a suite result without naming the engine that produced it.
- Reading 17.2's 437 as fully explained by 17.3. The annotation surface accounts
  for part of it; the remainder is unmeasured.
- Describing the suite as covering Sounio. It covers what the seed can parse.
