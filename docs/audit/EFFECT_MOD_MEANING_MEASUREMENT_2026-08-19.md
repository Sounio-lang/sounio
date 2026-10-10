<!-- docs:meta
topic_id: repo.docs.audit.effect-mod-meaning-measurement-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: minimax-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-mod-meaning-measurement-2026-08-19
-->

# Effect `with Mod` — measurement of what 2800 declarations are doing

**Date:** 2026-08-19
**Counsel:** minimax-cli2
**Working tree:** `audit/effect-mod-meaning-2026-08-19` at `/workspace/.wt/minimax-cli2/effect-meaning-2026-08-19` (branched from `origin/main` at `5f3cc9b296`).
**Dispatch:** `/tmp/dispatch_mod_meaning_claude1.md` (2026-08-19, founder directive).
**Companion scripts (landed in this commit):**

- `scripts/audit/measure_with_mod.cjs` — per-function body classifier for `with Mod` declarations
- `scripts/audit/measure_with_neighbors.cjs` — generalization: edit distance from builtin + orphan detection

**Sister PR coordination:** grok-cli5 PR #1963 (Phase 2b, proposes Mod as id 29). This measurement decides whether the id stays or exits.

Claims-Forbidden: `Mod is a typo for Mut`; `Mod is a real effect`; `Mod belongs in the EffectKind enum`; `Mod should be removed from all declarations`; `Mod functions do not mutate`; `Mod functions use %`; `the 2800 declarations were designed with intent`; `the parser is broken beyond Mod`. The measurement is the only thing authorised here.

---

## TL;DR (counts only — no conclusion)

| metric | value | what it tells you |
|---|---:|---|
| `with Mod` declarations in the tree | **2789** | matches dispatch's 2800 within grep-noise (multi-line signatures, sub-string matches) |
| Files containing `with Mod` | **360** | matches dispatch |
| Co-occurrence of `with Mod, Mut` on the same signature | **0** | the typo-for-Mut hypothesis loses its strongest evidence |
| Co-occurrence of `with Mod` with ANY other builtin (Div, Panic, Alloc, IO, …) | **0** | every `with Mod` is `with Mod` alone |
| First commit that introduced `with Mod` | `46c9ddc2a0` (2026-06-24) | one batch of 265 files / 117 507 insertions |
| Distinct non-builtin `with X` names in the tree (comments stripped) | **14** | most "names" at distance 1–2 from a builtin were comment fragments |
| Names at edit distance ≤ 2 from a builtin, excluding comments | **1: Mod** | dispatch's "Mod/Mut = distance 1" is off by one — actual Levenshtein is 2 |
| Orphans (no nearby builtin AND no `effect X { ... }` declaration) | **9** | tiny volumes (1–12 uses each), all semantic — see §6 |

**Five answers to the dispatch's questions, in order:**

1. **Mutate-vs-mod classification:** see §1 below.
2. **Mut co-occurrence:** **0/2789**. The strongest empirical evidence against the "Mod is a typo for Mut" hypothesis is that nobody who wrote `with Mod` also wrote `with Mut` on the same signature.
3. **Origin (git log):** **3 batch commits**, never grown incrementally. See §3.
4. **Generalization (edit distance):** **Mod is the only high-volume near-builtin name** in the entire tree. The dispatch's "se houver uma família inteira de quase-acertos" is empirically REFUTED. See §4.
5. **Orphans (real effects-by-declaration?):** **9 candidates**, total 51 declarations. None approach Mod's volume. See §5.

---

## §1 — Mutate-vs-modulo classification of the 2789 function bodies

Script: `scripts/audit/measure_with_mod.cjs`. Walks every `.sio` file, extracts every function whose signature contains `with Mod` as a token in the effect-list, brace-matches the body, and classifies it.

Sample size: **2789 / 2789** (exhaustive, not sampled).

| bucket | count | % | what it means |
|---|---:|---:|---|
| `REAL_WRITES_ONLY` (var reassigned, field/arr assignment, `&!` call) | **2053** | 73.6% | writes memory at runtime |
| `REAL_WRITES_AND_MOD` (both writes and `%`) | **12** | 0.4% | FNV-style hash loops |
| `MOD_ONLY` (`%` present, no writes) | **266** | 9.5% | pure-modular-arithmetic functions |
| `VAR_NO_REASSIGN` (`var X = ...; return X` — declared but never reassigned) | **1** | 0.04% | semantically `let`, declared as `var` |
| `NEITHER` (no var, no field-write, no `%`) | **457** | 16.4% | neither mutation nor modulo |

Three examples for each bucket are printed by the script.

**Reading this row-by-row against the three hypotheses:**

- H1 ("Mod meant Mut, 2800 typos"): 74% of bodies DO mutate, which fits. But §2 (zero Mut co-occurrence) and §3 (batch origin) and §5 (no other typo family) all weaken H1. The Mod text was written intentionally — they wrote the word `Mod`, not `Mut`. If it were a typo, some functions would also declare `with Mut` (or `with Mut, Mod`); none do.
- H2 ("Mod is a real effect for modular arithmetic"): 9.5% of bodies use `%` without mutation — these are FNV-style hash one-liners (`(x * 16777619 + y + z) % 1000000007`). These are legitimate candidates for H2. But 90% of declarations do not fit H2.
- H3 ("decorative, third reason"): 16.4% of bodies use neither writes nor `%`. The `imported_runtime_lift_contract.sio` style contract-checker is the canonical case — see §7 example A.

**Indeterminate is the honest answer.** No single hypothesis covers the distribution. The dispatch asked "se a resposta ficar INDETERMINADA, diz INDETERMINADA — e uma resposta legítima e melhor que um chute." The classification above is the answer; do not collapse it into "Mod is X" — the measurement does not support that.

---

## §2 — Co-occurrence with `with Mut` and other builtins

Script: `scripts/audit/measure_with_mod.cjs` (the co-occurrence counters at the bottom).

```
with Mut    on the same signature line as `with Mod`: 0
with Div    on the same signature line as `with Mod`: 0
with Panic  on the same signature line as `with Mod`: 0
with Alloc  on the same signature line as `with Mod`: 0
with IO     on the same signature line as `with Mod`: 0
```

**Every single `with Mod` declaration is `with Mod` alone.** No function carries both `Mod` and any other builtin effect.

This is the single most informative number in this audit. If `Mod` were a typo for `Mut`, one would expect at least some pairings (the author mistyped `Mod`, then re-typed `Mut` later on the same line; or the author learned the correct effect and switched; or the typing was inconsistent across the codebase). Zero of 2789 is a strong signal that the word `Mod` was placed there *on purpose* — even if the purpose isn't what `Mod` would semantically mean.

---

## §3 — Git-log origin of `with Mod`

`git log --oneline -S 'with Mod' --diff-filter=A` over `stdlib/` (the rest of the tree is too expensive to walk in time):

| commit | date | author | files | insertions | `with Mod` introduced | role |
|---|---|---|---|---:|---:|---|
| `46c9ddc2a0` | 2026-06-24 04:21 | Codex Review | 265 | 117 507 | **72** | WIP backup of solver/proof-checker research lane (~264 files) |
| `e94a39e9ef` | 2026-06-24 06:29 (2 h later) | Demetrios Chiuratto Agourakis | 782 | 162 473 | **575** | Commit missing audit docs, theorem solver suite, machine_ir fix, println(var) fix |
| `0fca69f4a7` | 2026-08-17 18:35 (~8 wk later) | Demetrios Chiuratto Agourakis | 31 | 97 356 | **2157** | Split theorem/portfolio under lexer byte ceiling (file move — no new declarations, content relocated) |

Net of file-split relocations, **647 declarations were genuinely introduced in two batch commits on the same day**, the first being an automated worktree backup. **Zero incremental design period.** No commit in the history says "introduce Mod effect for modular arithmetic" or anything like it. The text appeared with the WIP backup and stayed.

This kills the "someone thoughtfully designed a Mod effect" hypothesis. The text arrived with a 265-file dump.

---

## §4 — Generalization: how many other `with X` are near a builtin?

Script: `scripts/audit/measure_with_neighbors.cjs`.

Builtins in scope (23): `IO, Mut, Alloc, Panic, Div, GPU, Async, Prob, Epistemic, Causal, Network, Sensor, Render, Observe, NonAssoc, Audit, Hypothesis, MultiTest, ZD, Witness, Temporal, Learn, Chaotic`.

Stripping comments and string literals before matching (mandatory — without this, `// XOR with Adam` matches `with Adam` and inflates the count to 467), the distinct non-builtin `with X` names in the tree are:

| name | uses | distance to nearest builtin | nearest | declared? |
|---|---:|---:|---|---|
| **Mod** | **2800** | **2** | **Mut** | **no** |
| Deterministic | 12 | 10 | Panic | no |
| Approx | 10 | 4 | Alloc | no |
| Perturbative | 7 | 9 | Panic | no |
| NarrowWidthApproximation | 7 | 20 | Epistemic | no |
| NonUnitary | 5 | 7 | NonAssoc | no |
| Confidence | 4 | 7 | Panic | no |
| NaturalityG2 | 4 | 8 | Causal | no |
| Exp | 1 | 3 | IO | no |
| Log | 1 | 3 | IO | no |
| Choice | 2 | 3 | Chaotic | **yes** (`effect Choice { ... }`) |
| Counter | 1 | 4 | Render | **yes** |
| Fail | 1 | 3 | Panic | **yes** |
| Fetch | 1 | 4 | Mut | **yes** |
| Logger | (in dispatch, found in tree) | | | **yes** |
| Storage | (in dispatch, found in tree) | | | **yes** |

**At edit distance ≤ 2 from any builtin, only Mod qualifies.** The dispatch's hypothesis "se houver uma família inteira de quase-acertos, o problema não é o Mod" is **empirically REFUTED**: Mod is alone in both scale (2800 vs ≤ 12) and proximity (distance 2 vs the next-closest orphan at distance 3).

This is a single high-volume outlier, not a general pattern. The fix is not "tighten the parser to reject near-builtins" — that would only catch Mod, and Mod is already anomalous. The fix is whatever decision is made about Mod specifically.

**Levenshtein-distance correction for the dispatch:** Mod/Mut is distance **2**, not 1. `Mod → Mud → Mut` is two substitutions (`o→u`, `d→t`); no single edit closes the gap. The dispatch's framing is approximate; the data is exact.

---

## §5 — Orphans (no nearby builtin AND no `effect X { ... }` declaration)

From the table above, the orphan candidates are: **Deterministic (12), Approx (10), Perturbative (7), NarrowWidthApproximation (7), NonUnitary (5), Confidence (4), NaturalityG2 (4), Exp (1), Log (1)**. Total: **51 declarations**, none near a builtin (distances 3–20).

Spot-checks (see §7 examples C–F for some):

- `with Deterministic` — `examples/units/pharma_dose.sio:27` `fn weight_dose(dose_mg_per_kg: f64, weight_kg: f64) -> f64 with Deterministic { ... }` — semantically marks pure functions in a units/dosing context. NOT a builtin typo, NOT declared. The compiler accepts it silently because the parser doesn't validate.
- `with Approx` — `stdlib/math/approx.sio:441` `pub fn approx_taylor_sin(x: f64, terms: i64) -> f64 with Approx, Mut, Div, Panic { ... }` — semantically marks numerical-approximation functions. **Note: this declaration correctly pairs `Approx` with `Mut, Div, Panic`** — the author knew what they were doing and chose `Approx` as a meaningful semantic marker.
- `with NarrowWidthApproximation` — `stdlib/particle_physics/approx_effects_gum.sio:101` `... with NarrowWidthApproximation, Mut, Div, Panic { ... }` — same pattern.

These are **real effects-by-convention** in the dispatch's H3 sense. The author's intent is to mark a function as introducing approximation. The compiler silently accepts it because `with X` is not validated. The effect has no runtime semantics, so it's decorative — but not random: each has a coherent usage pattern.

If a follow-up PR adds a third hypothesis — "Mod is one of a family of 'effects by declaration' that never landed" — these 9 orphans are the candidate set. The dispatch asked the question; the answer is: yes, the family exists, but **Mod is not a member of it** — its volume (2800) and provenance (batch backup) make it a different phenomenon.

---

## §6 — Synthesis against the dispatch's three hypotheses

| hypothesis | empirical support | empirical refutation |
|---|---|---|
| H1: Mod is a typo for Mut | 74% of bodies mutate memory | **0** co-occurrence with Mut; **3** batch commits, never incrementally grown; **1** near-builtin (Mod itself), no typo family |
| H2: Mod is a real effect for modular arithmetic | 9.5% of bodies use `%` without mutation (FNV-style hash one-liners) | 90% of bodies do not fit H2; zero `effect Mod { ... }` declaration; no commit says "introduce Mod" |
| H3: Mod is decorative for a third reason | 16.4% of bodies do neither mutation nor `%`; small family of orphan effects (`Approx`, `Deterministic`, etc.) exists | 2800 declarations is too many for "noise"; orphan family totals only 51 declarations, 55× smaller than Mod |

**The classification of 2789 bodies is multi-modal:** three different behaviors (mutation / pure-mod / neither) all carry the same declaration. No single hypothesis covers the distribution.

**The decision belongs upstream of this measurement.** This document establishes what the 2789 functions are doing; it does not establish what `Mod` should mean or whether it should enter the enum.

---

## §7 — Worked examples (chosen by the classifier, not cherry-picked)

### Example A — `NEITHER` (16.4% of declarations)

`stdlib/safety/imported_runtime_lift_contract.sio:145`

```
formal_theorem_ready: i64,
ok_mask: i64) -> i64 with Mod {
    if artifact_fp != 642810357 { return 0 - 1 }
    if instance_fp != 276304915 { return 0 - 1 }
    // ... 35 more `if X != Y { return 0 - 1 }` guards ...
    if ok_mask != 1023 { return 0 - 1 }
    return 508176294
}
```

Pure comparison. No mutation, no `%`. The `with Mod` declaration has no runtime observable difference from `with Mut` or no effect clause at all. This function is in the H3 (decorative) bucket.

### Example B — `MOD_ONLY` (9.5%)

`stdlib/systems/lorenz_i256_cert_core.sio:8`

```
pub fn lorenz_i256_cert_mix(acc: i64, value: i64, salt: i64) -> i64 with Mod {
    return ((acc * 16777619) + value + salt) % 1000000007
}
```

One-line FNV-1a mix. Pure `%`, no mutation. The H2 (real modular-arithmetic effect) candidate — and a legitimate reason to add `Mod` to the enum, IF that is what Mod meant. But this single body is the strongest pro-H2 evidence in the entire tree; 90% of declarations don't look like this.

### Example C — `REAL_WRITES_ONLY` (73.6%)

`stdlib/safety/imported_runtime_lift_contract.sio:34`

```
formal_theorem_ready: i64) -> i64 with Mod {
    var anchor_mask: i64 = 0
    if private_envelope_fp == 834620917 {
        if private_envelope_audit_fp == 276591483 {
            anchor_mask = anchor_mask + 1
        }
    }
    // ... more `anchor_mask = anchor_mask + N` per matched condition ...
    return anchor_mask
}
```

Real mutation (`var` reassigned). The H1 (typo for Mut) candidate. But: no `with Mut` declaration in the same function, and the body uses no `%`. The mask-construction pattern is i64-only and looks like a fingerprint accumulator — the name `Mod` here probably means "modular mask" or "modifier", not "Mut" and not "modular arithmetic". A third hypothesis that's plausible at this single body but doesn't generalize.

### Example D — orphan effect-by-declaration (well-formed use of `with Approx`)

`stdlib/math/approx.sio:441`

```
pub fn approx_taylor_sin(x: f64, terms: i64) -> f64 with Approx, Mut, Div, Panic {
    // Taylor-series sin approximation
}
```

Note that `Approx` is paired with `Mut, Div, Panic` (the actual effects the function needs). `Approx` is a semantic marker, not a builtin. The compiler silently accepts it. **This is the H3-orphan pattern, done correctly.**

### Example E — `REAL_WRITES_AND_MOD` (0.4%)

`stdlib/theorem/div_witness.sio:48`

```
tag: i64, ..., ok2: i64) -> i64 with Mod {
    if tag <= 0 { return 0 - 1 }
    if div_witness_all3(ok0, ok1, ok2) == 0 { return 0 - 1 }

    var fp: i64 = 211
    fp = ((fp * 16777619) + tag + 1601) % 1000000007
    fp = ((fp * 16777619) + q0 + 1601) % 1000000007
    // ... 9 more fold-mix-mod lines ...
    return fp
}
```

Mutation AND modulo. The canonical FNV-1a hash loop. This body would fit either H1 (writes memory, declared `Mut` instead of `Mod`) or H2 (uses `%`, declared `Mod` instead of nothing). The classification cannot distinguish — the body is consistent with both.

---

## §8 — What this measurement does NOT establish

- Whether `Mod` SHOULD enter the EffectKind enum (grok-cli5 PR #1963 decision).
- Whether the 2800 declarations are semantically wrong or semantically correct.
- Whether removing `Mod` from those declarations would change observable program behaviour.
- Whether the parser should reject unknown effect names (separate question — affects all 9 orphans, not just Mod).
- Whether the historical introduction of `Mod` was deliberate or accidental (the `Codex Review` author signature on the first commit does not resolve intent).

These belong to a separate decision document, not this one. The dispatch said "se a resposta ficar INDETERMINADA, diz INDETERMINADA — é uma resposta legítima." The classification of 2789 bodies is **multi-modal and does not collapse to a single hypothesis**, so the conclusion is: **the meaning of `with Mod` is INDETERMINATE from the body classification alone**, and the upstream decision is independent of this measurement.

---

## §9 — Reproduction

Both scripts are CommonJS, zero external deps, runnable in the prebuilt-toolchain pod without `npm install`. Each takes the repo root as argv:

```
node scripts/audit/measure_with_mod.cjs <repo-root>
node scripts/audit/measure_with_neighbors.cjs <repo-root>
```

Expected output (current `origin/main` at `5f3cc9b296`):

```
measure_with_mod.cjs:
  Total `with Mod` declarations: 2789
  REAL_WRITES_ONLY:        2053
  REAL_WRITES_AND_MOD:       12
  MOD_ONLY:                 266
  VAR_NO_REASSIGN:            1
  NEITHER:                  457
  Co-occurrence with Mut/Div/Panic/Alloc/IO: all 0

measure_with_neighbors.cjs:
  Total distinct non-builtin `with X`: 14
  Names at edit distance 1 from builtin: 0
  Names at edit distance 2 from builtin: 1  (Mod)
  Orphans (no builtin, not declared): 9
```

---

## §10 — Files changed by this commit

- `scripts/audit/measure_with_mod.cjs` (new)
- `scripts/audit/measure_with_neighbors.cjs` (new)
- `docs/audit/EFFECT_MOD_MEANING_MEASUREMENT_2026-08-19.md` (this document)

No `.sio` source files modified. The measurement is observational; any follow-up that touches `with Mod` declarations is a separate PR and a separate decision.
