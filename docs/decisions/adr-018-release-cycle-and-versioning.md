<!-- docs:meta
topic_id: repo.docs.decisions.adr-018-release-cycle-and-versioning
authority: repo_only
audience: users
last_validated: 2026-10-24
validated_by: docs/RELEASE_POLICY.md; scripts/ci/changelog_fragment_gate.sh
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-018-release-cycle-and-versioning
-->

# ADR-018: Release trains, editions and the compatibility contract

**Status**: accepted (implementation in progress)
**Date**: 2026-10-24
**Supersedes**: the versioning and cadence sections of `docs/RELEASE_POLICY.md` before this date

## Context

Between `v2.1.0-native-v2` (2026-06-04) and 2026-10-24, `main` received 6,319 commits from 2,164 merged pull requests. Over that period there was no release, `CHANGELOG.md` had no new entry, and the stdlib reliability snapshot that the release checklist reads (`artifacts/stdlib/stdlib_reliability_status.v1.json`) stayed at 2026-05-12. Four agents and several human contributors now merge in parallel through a merge queue. Three different version strings were in public use: 2.1.0 (`CITATION.cff`), "Madaros v0.80.0" (`bin/souc --version`), and dated `madaros-YYYYMMDD` tags.

Several of the changes merged in that period alter what existing programs do, as listed under *3.0.0* below. The project had no rule for deciding whether such a change is a minor or a major release. A scientific language also needs a rule that ordinary language projects do not: whether a change to a printed number is a fix, a feature or a break.

## Decision

### 1. Two version axes

1. **The toolchain**, comprising `souc`, Madaros, the stdlib and the runtime, carries one Semantic Versioning 2.0.0 number, `MAJOR.MINOR.PATCH[-PRE]`. `CITATION.cff` is its single source of truth. At a release tag, `souc --version` must report the same string; the engine build identifier follows it, as in `3.0.0-rc.1 (madaros 4f1c2e9a)`, and never replaces it.
2. **The language** carries an **edition**, named by year: `edition = "2026"` in `sounio.toml`. `souc new` already writes this key. An edition fixes the lexical and syntactic surface and the meaning of programs. A toolchain release may add an edition but never changes an existing one.
3. **Edition 2026** is the language as shipped in 3.0.0. **Edition 2027** will carry the thirteen lexical rulings of ADR-015. It becomes available as an opt-in in a 3.x minor, and becomes the default only in 4.0.0, because changing the default edition is a breaking change.
4. A file outside any package, or a package that does not declare an edition, gets the default edition of the toolchain's major (2026 for 3.x), with a warning that names the edition chosen. Every edition remains supported for at least two majors after its successor becomes the default.

### 2. Feature trains

1. Each minor or major release is a **train**: a named milestone (`3.1`, `3.2`, …) with a scope written down at kickoff and a **freeze date**. The milestone lives in GitHub Milestones and in the Notion *Milestone* field.
2. At freeze, the release manager cuts `release/X.Y` from `main`. From then on the branch receives only fixes, by cherry-pick of PRs that carry the label `backport:X.Y`. `main` immediately becomes the next train.
3. A feature that is not merged by freeze moves to the next train. A train never waits for a feature. This keeps "release by feature" from turning into "release when everything is done".
4. The branch is tagged `vX.Y.0-rc.1`. An RC soaks for at least **7 days**. Any fix merged into the branch produces `rc.N+1` and restarts the soak.
5. **Gates for an RC tag:**
   - an unfiltered `Release Gate` run is green on the branch head;
   - there are zero open `release-blocker` issues;
   - the stdlib reliability and science-pipeline snapshots were regenerated on the branch head;
   - the compatibility check of §3 passes against the previous release's baseline;
   - the CHANGELOG section was assembled from the fragments of §4;
   - `CITATION.cff` and `souc --version` agree.
6. **Gates for the stable tag:** the soak ends with no new `release-blocker`, the Zenodo version DOI is minted and recorded in `CITATION.cff`, and the GitHub Release carries the CHANGELOG section verbatim.
7. **Patch releases** `vX.Y.Z` are cut from `release/X.Y` and need the RC gates without a soak.
8. Pre-release identifiers are `alpha.N` (API unstable), `beta.N` (feature-complete), `rc.N` (frozen), and nothing else. Dated or engine-named tags (`madaros-YYYYMMDD`, `v2.1.0-native-v2`) are frozen history. Engine prebuilts ship as artifacts of a toolchain release, not as releases of their own.

### 3. The compatibility contract

Compatibility is judged per edition. A change is **breaking** if, under the same edition, a program that compiled with release X.Y.Z either:

1. **source** — no longer compiles, or compiles with a new error;
2. **semantics** — compiles but takes a different path: name resolution, dispatch, effects, evaluation order or refusal behaviour;
3. **numerics** — produces a recorded output that fails the epistemic consistency test below.

New warnings, new items and new modules are not breaking.

**Soundness exception.** Making a program refuse instead of printing a fabricated or wrong value is allowed in a minor release, even though it is a source break. Such a refusal is announced in a `refusal` fragment that names the diagnostic code and the PR. A wrong number that prints is worse than a refusal; this is the same exception Rust makes for soundness fixes.

**Epistemic numerical consistency.** Each recorded output in the compatibility corpus has a value `x`, an optional certified enclosure `e`, and an uncertainty budget `B` whose components are keyed by provenance (ADR-016, decision 6). Call the baseline from the previous release `(x₀, e₀, B₀)` and the candidate `(x₁, e₁, B₁)`.

1. **Correlation is accounted for, not assumed away.** The two outputs come from the same program run on the same inputs, so every budget component they share by key is fully correlated (ρ = 1) and cancels in the difference. Applying `E_n = |x₁ − x₀| / √(U₀² + U₁²)` to the whole budget would assume independence, which ADR-016 decision 5 forbids. It would also let a regression of nearly 2U pass as noise. Only the component that is not shared can absorb a difference: the numerical method error, key `numerics` (Type B, GUM §4.3).
2. **The certified layer:** if both outputs carry enclosures, `e₀ ∩ e₁ ≠ ∅`. Rigorous enclosures need no independence assumption.
3. **The statistical layer:** `(x₁ − x₀)² · den² ≤ num² · (U₀,ₙᵤₘ² + U₁,ₙᵤₘ²)`, decided exactly on squares of rationals as in `formal/EpistemicStat.lean`. The default policy is `num/den = 1` on expanded uncertainties (k = 2, ISO 13528). If an output declares no `numerics` component, its `U_num` is zero and the test reduces to exact equality of the recorded value. Bit-exactness is therefore the default, and not an exception that has to be declared.
4. **The budget must not move silently.** The non-`numerics` components of `B₀` and `B₁` must match key by key:
   - an uncertainty that shrinks with no new provenance key is **breaking**, because ADR-016 decision 4 allows narrowing only by new evidence, never by recomputing the same data;
   - an uncertainty that grows, or that shrinks because a new provenance key was added, is a **semantics** change, allowed in a minor release with a `semantics` fragment.
5. **The correction exception:** a change that fails the test is still allowed in a patch release if it moves the value toward an exact oracle (an ADR-009 reference or a C++23 exact oracle) and the oracle value lies in `e₁`. It must be recorded as a `numerical-correction` fragment that cites the oracle receipt.
6. A change that passes rules 2 and 3 but is not bit-identical is compatible and needs a `numerical` fragment.

The compatibility corpus starts with the corrected pinned 200-program corpus and the stdlib golden outputs. Its baseline is recorded at each stable tag. A major release is the only release allowed to fail the test without an exception, and it lists every failure under *Breaking*.

### 4. Changelog fragments

1. Every PR that touches the compiler, runtime or stdlib, as classified by `scripts/ci/classify_ci_impact.sh`, adds one file `changelog.d/<slug>.<kind>.md`. The kinds, in release-note order, are: `breaking`, `refusal`, `semantics`, `numerical-correction`, `numerical`, `added`, `changed`, `deprecated`, `removed`, `fixed`, `performance`, `security`, `internal`.
2. `internal` is the explicit, auditable way to say that a change has no user-visible effect. Its file still names the change.
3. Fragments never conflict in the merge queue, because each PR adds its own file and none edits `CHANGELOG.md`. At an RC, `scripts/release/changelog_assemble.sh` writes the section and removes the consumed fragments. Only the release manager edits `CHANGELOG.md` directly.
4. The gate (`scripts/ci/changelog_fragment_gate.sh`) runs on pull requests in the Impact job. PRs opened before this ADR merged are exempt, so open work by other agents is not failed after the fact.

### 5. Support window

1. The current minor receives every kind of fix.
2. The previous minor receives only `numerical-correction` and `security` fixes, until the next minor ships. A paper that cites 3.1.x can then rely on 3.1.x being corrected, not merely superseded.
3. Older releases receive nothing.

### 6. The 3.0 train

3.0.0 is a major release because these changes have already shipped. All of them are in edition 2026:

- module-local name resolution (#2784), a semantics change;
- refusal propagation and the end of fabricated stub returns (#1801), struct field visibility (KL-4, #2488), unit-typed arguments refused into bare `f64` (KL-13, #2499), unknown `Knowledge` components refused (E241, #2102), and the other diagnostics listed in `CHANGELOG.md`, all source changes;
- numerical corrections that fail bit-exactness against 2.1.0, among them #1024, #1630, #2165 and #2385.

Freeze happens at the merge of this ADR. `release/3.0` is cut from that commit, and `v3.0.0-rc.1` is tagged there once the RC gates of §2.5 pass. The compatibility check of §3 has no baseline before 3.0.0; it starts enforcing at 3.0.1, against the 3.0.0 baseline.

ADR-015 and ADR-016 are decided but not implemented. They are not part of 3.0.0: ADR-015 lands as edition 2027, and ADR-016 lands in 3.x minors as additive API.

## Consequences

- A release becomes something the project can do repeatedly rather than an event, and its notes cost nothing at release time.
- A numerical change is no longer simply "a fix". It is classified, and the classification is checked. A paper's numbers can be tied to a release and an edition.
- The compatibility corpus must record budgets by provenance key. Until an output does, it is held to bit-exactness, which is strict but honest.
- `souc --version`, the website's getting-started pages and `README.md` must stop presenting "Madaros v0.80.0" as a version. They follow the toolchain version from 3.0.0 on.
- The edition machinery is new compiler work: parsing the manifest key, the default-edition warning, and per-edition dispatch in the lexer. Until it exists there is only one edition, so nothing depends on it.

## Implementation

1. This ADR, `docs/RELEASE_POLICY.md`, the 3.0.0-rc.1 `CHANGELOG.md` section, `changelog.d/`, the assembler and the fragment gate. **This PR.**
2. The compatibility checker, `scripts/release/compat_check.cpp` (C++23, exact rational decision), with a Lean theorem `formal/ReleaseCompat.lean` stating that the test reduces to equality when `U_num = 0` and is symmetric in the two versions. Then the corpus baseline format. **Next, before 3.0.1.**
3. `souc --version` reporting the toolchain version, plus a gate that compares it with `CITATION.cff` at tag. **Before the 3.0.0 stable tag.**
4. The edition key, the default-edition warning, and edition 2027 as an opt-in. **The 3.x trains.**

## Grounded in

- ADR-016 (decisions 4, 5 and 6, on resolution by new evidence only, independence of sources, and budget keys) and `formal/EpistemicStat.lean` (the `E_n` policy decided on squares).
- ADR-009 (verified foreign reference oracles).
- ISO 13528 (`E_n`); JCGM 100:2008 (GUM) §4.3 (Type B evaluation).
- Semantic Versioning 2.0.0; the Rust edition model (RFC 2052) and its soundness exception (RFC 1122); the release-train model of Rust and Firefox; towncrier's news fragments.
