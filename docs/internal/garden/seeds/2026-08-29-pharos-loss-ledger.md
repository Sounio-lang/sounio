<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-29-pharos-loss-ledger
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-29-pharos-loss-ledger
-->

# Pharos — a compiler that publishes its own erasures

> **Status**: Garden seed | **Last validated**: 2026-08-29 | **Source**: kimi-cli2 conversation on compilers, CHERI, and the Pireus lineage (branch `lane/cursor-1/20260826`)

## Butterfly

> um compilador que publica as próprias perdas

The phrase arrived during a late-night conversation that ran from compiler
philosophy to machine language to CHERI to the Pireus operator-genome lineage
(v2's frozen `B=1128` discovery; the ten materialization obligations). The felt
pressure: Pireus keeps meaning intact *across materials* — Xeon, Apple Silicon,
DGX, dual U250 — but nothing yet keeps the compiler's own erasures visible
*across passes*. The founder's recurring question — "what real information was
made invisible so that this system could appear simpler?" — has not yet been
turned inward onto the compiler in executable form.

The name follows the project's geography: Sounion is the cape that watches,
Pireus is the port that ships semantics to silicon. Pharos is the lighthouse —
the piece that illuminates what would otherwise sink silently.

## Core Idea

EISA separates `val`, `err`, and `u` — it tracks what a value *is*. Pharos
would be the dual: it tracks what compilation *did* to the program's meaning.

- Every compilation emits, alongside the artifact, a **loss certificate**: a
  machine-readable ledger, hash-bound to the artifact lineage, of every
  semantic erasure the pipeline performed — reassociations, precision demotions
  (`f256` → `f64`), fused rounding (FMA contraction), reordered reductions,
  collapsed provenance. Each entry is typed: what was erased, where, by which
  pass, and a bound marked proven, measured, estimated, or declared-unknown.
- **Undeclared erasure is a compile error, not a warning.** A pass that cannot
  account for itself is *silence*: an unclassifiable loss, therefore an error.
  The promise is not "nothing was lost" but "nothing was lost in silence."
- **Loss algebra**: provenance semirings (Green–Karvounarakis–Tannen, database
  literature) ported from query evaluation to compiler passes. Sequential
  passes compose losses; dominance rules prevent double counting (an `f32`
  demotion subsumes a later `f64` demotion of the same value).
- **Calibration witness**: plant a known crime. Compile the frozen `B=1128`
  genome with and without one reassociation pass; the certificate diff must
  show exactly that reassociation — nothing more, nothing less. A missed
  erasure, or a hallucinated one, falsifies the instrument. The instrument is
  calibrated by a deliberate, reproducible offense.
- **Budget interface**: optimization level stops being `-O2` and becomes an
  epistemic budget — e.g. `--max-loss=reassociation:none, precision:f64-floor`.
  Compilation becomes constrained optimization: maximize speed subject to a
  declared meaning budget. Today's `-ffast-math` is a confession without
  accounting; Pharos is the accounting.

Open question, fixed at planting: **is silence detectable by construction?**
Can every pass be given a closed-world loss account such that the compiler
statically refuses an unaccountable pass — or is there always room for a
transformer that does not know it should be confessing? If detectable, this
becomes a type system; if not, the impossibility result is itself the
deliverable.

## Connections

- [`FOUNDER_INTENT.md`](../../../FOUNDER_INTENT.md) — the recurring question
  and the erasure list this seed turns inward; also the evidence-label ladder
  this seed is subordinate to.
- [`stdlib/eisa/core_v2.sio`](../../../../stdlib/eisa/core_v2.sio) — the
  `val`/`err`/`u` separation; Pharos is its compiler-side dual.
- Pireus Operator Genome v3 (branch
  `lane/codex/pireus-u250-dual-card-admission-20260828`,
  `stdlib/hardware/pireus/operator_genome.sio`,
  `tools/pireus/GARDEN_PIREUS_OPERATOR_GENOME_V3.md`) — obligations 8–10
  (`LineageBinding`, `ReplayBinding`, `ClaimBoundary`) are the lineage chain a
  loss certificate would ride; the frozen `B=1128` genome is the calibration
  corpus.
- [`docs/decisions/adr-008-claim-oracle-semantic-clock.md`](../../../decisions/adr-008-claim-oracle-semantic-clock.md) —
  the certificate must be emitted by Sounio's own compiler; no foreign tool may
  define what was lost.
- [`2026-07-11-the-zero-of-encounter.md`](2026-07-11-the-zero-of-encounter.md) —
  kindred seed: surface sameness hiding real distinctions.
- External prior art to confront, not ignore: CompCert (verified semantic
  preservation), Herbie (numerical rewriting with error analysis),
  CADNA/Verificarlo (stochastic arithmetic), provenance semirings (databases).
  None of these enforces fail-closed loss accounting *inside* the compiler,
  hash-bound to artifact lineage; that enforcement is the only novelty
  candidate, and it remains unproven.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Captured: the butterfly phrase, the dual-of-EISA framing, the loss-algebra sketch, the calibration witness, the budget interface, the silence question. |
| `Hypothesis` | Every Madaros pass can be given a closed-world loss account such that undeclared erasure is statically refusable. Whether silence is detectable by construction is open. |
| `Executable` | None. No code, no gate, no census exists yet. |
| `Claim-ready` | No. |

## What This Is Not

- Not a compiler feature, not an implementation commitment, not a roadmap.
- Not a claim that semantic information loss is measurable in general;
  Kolmogorov-style undecidability bounds that. The scope is closed-world:
  losses the compiler has concepts for.
- Not a substitute for CompCert-style correctness proofs; preservation of
  semantics and accounting of losses are different obligations.
- Not a novelty claim over provenance semirings, Herbie, or stochastic
  arithmetic; per the Pireus `ClaimBoundary` obligation, this seed promotes no
  algorithmic, scientific, or priority claim.
- Not clinical guidance, not user-facing documentation, not a statement about
  what Sounio's compiler does today.

## Next Executable Bridge

A **pass census**, not machinery: enumerate Madaros's optimization and lowering
passes and classify each as (a) loss-free, (b) loss with known bound, or
(c) loss uncharacterized. The census is itself the first executable artifact
and measures today's silence surface. It requires no new semantics — only
honesty about existing passes.

## Session Record

Planted 2026-08-29 (UTC) from a kimi-cli2 conversation. The repository's live
coordination protocol was honored: the exact write set was claimed via
`bin/sounio-coord scope` before the first write. The seed intentionally
contains no code and no expected values; per Garden discipline, the butterfly
is preserved, not forced into a deliverable.
