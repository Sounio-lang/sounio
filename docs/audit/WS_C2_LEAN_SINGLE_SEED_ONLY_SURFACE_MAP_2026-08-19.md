<!-- docs:meta
topic_id: repo.docs.audit.ws-c2-lean-single-seed-only-surface-map-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ws-c2-lean-single-seed-only-surface-map-2026-08-19
-->

# WS-C2 — lean_single as seed only: surface map (semantic declaration)

**Date:** 2026-08-19  
**Lane:** grok-cli4 / wsc2-seed-only-map  
**Directive (founder):** Madaros is the **canonical compiler**; lean_single becomes
**bootstrap seed** only.  
**This document:** design / measurement. **No code changes** in this turn.  
**Related:** SeedReceipt + refresh recipe (provenance of the seed ELF); not a
substitute for this map.

---

## Claims-Forbidden

| claim | status |
|---|---|
| “Madaros replaces lean_single” | **FORBIDDEN** without the debt list below measured and closed |
| “lean_single is already only a seed” | **FALSE** on main today — it is still three roles at once |
| “Full Test Suite runs Madaros” | **FALSE** — CI pins `SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2` built from the lean_single self-host chain |
| “Madaros is fixed-point verified like the seed” | **FALSE** as default claim — `madaros_fixed_point_gate.sh` exists as a ladder; CLAUDE.md / gate header: do not describe Madaros as fixed-point-verified without gen2==gen3 evidence on that tree |

**Semantic clock (product claims):** Madaros under `bin/souc` is already the
*user-facing* default and the intended claim oracle surface. That is a **product**
declaration. It is **not** the same as “every gate, script, and suite already
runs Madaros.” This map separates those.

---

## 0. Three roles lean_single still holds (the problem)

| role | what it means | still true on main? |
|---|---|---|
| **A. Bootstrap seed** | ELF/source that *builds* Madaros (`main.sio`) and the self-host ladder | **Yes — necessary** |
| **B. Invokable engine** | People/CI run programs with `SOUNIO_SOUC_ENGINE=lean_single` or the raw ELF | **Yes — widespread** |
| **C. Sole fixed-point trust anchor in daily CI** | Byte-identical self-repro of *itself* is what Contracts hard-checks by default (`canonical_compiler_gate`) | **Yes — structural** |

While B and C remain load-bearing for “what the language is,” there are
**two languages** in practice: Madaros-default and lean_single-oracle/suite.

Making lean_single **only A** means retiring B as a *product* engine and
narrowing C to “seed integrity,” not “semantic reference for science/CI.”

---

## 1. Surfaces that still use lean_single as an **engine** (or build it as one)

Measured by ripgrep over `scripts/ci`, `bin/souc`, `Makefile`, `.github/workflows/ci.yml`
(not an exhaustive archive/ sweep). Counts are order-of-magnitude inventory, not
a claim that every hit is wired into default Contracts.

### 1.1 Wrapper and resolution

| surface | behaviour |
|---|---|
| `bin/souc` | Default `SOUNIO_SOUC_ENGINE=madaros`. Explicit `lean_single` forces raw ELF. **Fallback** to lean_single if Madaros ELF missing (stderr notice). Raw positional `SRC OUT` **requires** lean_single ELF. |
| `scripts/lib/resolve_souc.sh` | Resolves to `bin/souc` (wrapper → Madaros by default). |
| `scripts/lib/resolve_madaros.sh` | Madaros-only resolution. |
| `scripts/ci/souc-native-wrapper.sh` | Verb CLI over a raw ELF (often lean_single). |
| `scripts/ci/souc-seq-leansingle.sh` | Shim “Seq&lt;T&gt;-capable lean_single ELF” for gates that need it. |

### 1.2 Bootstrap / seed integrity (engine *of the seed*, not of user code)

| surface | role |
|---|---|
| `Makefile` `build` | gen1→gen2→gen3 of **lean_single.sio** via `bin/souc-linux-x86_64` |
| `scripts/ci/build_modular_madaros.sh` | Derives a **source-tracking seed** from current `lean_single.sio`, then compiles `main.sio` → Madaros. Comment: committed seed may lag; bootstrap ELF preferred. |
| `scripts/ci/canonical_compiler_gate.sh` + `seed_receipt_provenance_gate.sh` | Seed self-repro + optional SeedReceipt provenance |
| `scripts/ci/verify_lean_seed.sh` | Fixed point + optional DDC of committed seed |
| `scripts/ci/souc_v2_gate.sh`, `selfhost_host_gate.sh`, `reproduce_artifact.sh` | Self-host ladder compiling **lean_single.sio** |
| `scripts/dev/refresh_lean_seed.sh` / SeedReceipt | How to refresh the seed ELF honestly |

These are **seed machinery**. They do not go away when lean_single is “seed only”;
they *are* the seed role.

### 1.3 CI jobs that run user/tests under a lean_single-derived binary

| surface | measured fact |
|---|---|
| **Full Test Suite** (`.github/workflows/ci.yml`) | `SOUNIO_TEST_SOUC_BIN: /tmp/souc-stage2` — artifact from **native self-host of lean_single**, not Madaros |
| Native / source-bootstrap self-host jobs | Produce and prove `souc-stage2` from lean_single chain |
| Contracts step “Canonical lean_single fixed point” | Always on Contracts when job runs |

So the **largest** behavioural corpus in CI is still judged by a **lean_single-lineage**
compiler, while Madaros has its own witness/full gates in parallel.

### 1.4 Gates that **force** lean_single as run engine (habit or necessity)

Representative set (force pin via `SOUNIO_SOUC_ENGINE=lean_single` or raw
`souc-lean-single-x86_64`). Full force-ish count from inventory script: **~40+**
`scripts/ci/*.sh` files. Clusters:

| cluster | examples | why the script says lean_single |
|---|---|---|
| **Science / package import** | `package_pbpk_gum_gate.sh`, `package_import_science_gate.sh`, `sounio_package_support_gate.sh`, `pediatric_pbpk_gate.sh` | Explicit: Madaros multimodule thin-link still fails (“effect not declared GPU” etc.); lean_single resolves imports + effects |
| **Particle / physics vertical** | `particle_exp10_*`, `particle_exp123_*`, `particle_exp7_*`, … many default `SOUNIO_SOUC_ENGINE=lean_single` | Historical “full vertical on seed”; some dual-run Madaros |
| **Epistemic dual-engine** | `epistemic_fabrication_detect_gate.sh`, `f64_bitcast_sitofp_boundary_gate.sh` | lean_single as **healthy reference** when Madaros is the defect under test |
| **Engine split oracles** | `e219_engine_oracle_gate.sh`, `engine_parity_gate.sh` | lean_single is the **other** engine by definition |
| **Witness / zero / honesty** | `zero_event_gate.sh`, `zero_provenance_witness_gate.sh`, `self_falsifying_compilation_line_r28_gate.sh` | Pinned lean_single execution |
| **Unit-types current-source** | `unit_types_derived_gate.sh`, `unit_types_clinical_current_source_gate.sh` | Rebuild lean_single from source then run |
| **Launcher / CLI shape** | `souc_launcher_portability_gate.sh`, `souc_invoke_selftest.sh` | Must exercise lean_single argv/`SRC OUT` path |
| **Seq / special ELF** | `souc-seq-leansingle.sh` consumers | Capability pinned to a lean_single build |

### 1.5 Docs / narrative (authority without execution)

| surface | message |
|---|---|
| `CLAUDE.md` §4 | Madaros default; lean_single seed + fixed-point ELF; force via env |
| `docs/MADAROS_STATUS.md` | Same split; “canonical fixed-point ELF” still attached to lean_single |
| `AGENTS.md` | `bin/souc` → Madaros; lean_single bootstrap seed |
| Gate comments / audits | Widespread “lean_single reference”, “Track A = lean_single” |

Docs already **want** Madaros as product default. They still **teach** lean_single
as fixed-point and escape hatch — correct for role A/C, confusing if read as
role B forever.

### 1.6 What is *not* required for “seed only”

- Deleting `lean_single.sio` or the seed ELF  
- Stopping gen2==gen3 / SeedReceipt  
- Stopping `build_modular_madaros.sh` from deriving a seed  

Those are the seed role.

---

## 2. Necessity vs habit

Classification rule used here:

- **NECESSITY (seed):** cannot build or attest the bootstrap without lean_single source/ELF.  
- **NECESSITY (oracle):** gate’s *theorem* is an engine split or “Madaros is wrong, seed is healthy.”  
- **NECESSITY (gap):** script documents Madaros cannot run this workload yet (import/effects/path).  
- **HABIT:** default env is lean_single with no measured Madaros blocker in the file header; dual would be enough or Madaros-only is plausible.  
- **UNKNOWN:** needs a re-run under Madaros to classify — listed as debt until measured.

### 2.1 Real necessity — keep under “seed only”

| item | class |
|---|---|
| `make build` / `souc_v2_gate` / selfhost_host_gate / reproduce_artifact | seed ladder |
| `build_modular_madaros.sh` seed derivation | seed builds Madaros |
| `canonical_compiler_gate` + `verify_lean_seed` + SeedReceipt | seed integrity / provenance |
| `souc` raw `SRC OUT` and `SOUNIO_SOUC_ENGINE=lean_single` escape | bootstrap/debug CLI (may stay as **non-product** escape) |
| `souc_invoke_selftest` / launcher portability lean arm | must keep testing the escape CLI shape |
| `e219_engine_oracle_gate` (seed side of split) | oracle about seed vs Madaros disagreement |
| Dual-engine **reference** legs (fabrication F1 lean non-zero var; f64 KCONF lean healthy) | oracle: Madaros defect only holds if seed is healthy |

### 2.2 Necessity by **Madaros gap** (debt — see §3)

| item | stated or structural gap |
|---|---|
| `package_pbpk_gum_gate` / package import science | Madaros multimodule: spurious missing GPU effect; lean_single enforces full import/effect surface |
| `sounio_package_support_gate` (lean pin) | same family |
| Full Test Suite on `souc-stage2` | Suite corpus and vacuous baselines calibrated on lean lineage; Madaros has separate corpus/witness gates — **not** the same bar |
| Many `particle_exp*` default lean_single | historically seed path; dual gates exist for some — residual need **unmeasured** here |
| `pediatric_pbpk_gate` lean pin | likely package/import family — confirm under Madaros |
| `unit_types_*_current_source` rebuild lean | testing **seed** current-source semantics, not Madaros |
| Seq lean_single shim | capability not claimed on Madaros path without evidence |

### 2.3 Habit / dual-by-design (not “Madaros can’t”)

| item | note |
|---|---|
| Particle exp dual gates (`exp9`, `exp11`, …) | Already run both; lean default is habit for “primary” |
| `sounio_science_flex_gate` | Primary lean, secondary flexible — product should flip primary to Madaros when green |
| Docs saying “force lean_single for X” without a live Madaros fail | Narrative habit |
| Local developers using `SOUNIO_SOUC_ENGINE=lean_single` because Madaros not built | Wrapper fallback — ergonomics, not semantic authority |

### 2.4 Wrapper fallback (special)

When Madaros is missing, `bin/souc` falls back to lean_single. That preserves
**buildability** of a fresh clone. For “seed only” product semantics, fallback
should remain a **loud degraded mode** (already notices), never silent authority
for claims. Claims stay Madaros-or-fail (aligns with claim-oracle intent).

---

## 3. Debt list — where Madaros is not yet “canonical de facto”

Each row is a place the seed still **does work Madaros does not** (or CI does
not trust Madaros to do). Closing the row is required before “Madaros replaces
lean_single as engine” is sayable.

| ID | surface | seed still does | Madaros gap (as documented or structural) | evidence level |
|---|---|---|---|---|
| D1 | Multimodule package import + PBPK/GUM workflow gates | Full import graph + effect check + run | Thin-link / effect false positive (GPU); gate pins lean_single | **script-stated** (`package_pbpk_gum_gate.sh`) |
| D2 | Full Test Suite corpus | ~all run-pass/compile-fail under stage2 | Suite not run under Madaros in default CI; baselines lean-calibrated | **CI-measured** (`ci.yml` `SOUNIO_TEST_SOUC_BIN`) |
| D3 | Fixed-point of the *product* compiler | lean_single self-repro is Contracts-hard | Madaros fixed-point is optional ladder, not the daily trust anchor | **gate + docs** |
| D4 | Engine-split oracles (E219, fabrication F1/F2, f64 KCONF) | Healthy reference / disagreement pole | Not a “gap” to delete — need a **named reference** even after seed-only (could become “frozen seed oracle ELF”, not “product engine”) | **by design** |
| D5 | Particle / science long verticals | Default lean full run | Dual exists for some; Madaros-only green not claimed here | **partial / unmeasured** |
| D6 | Building Madaros itself | Seed compiles `main.sio` | No alternative bootstrap on main without a seed-class ELF | **necessary seed** (not debt to remove — debt is only if Madaros could self-host without seed) |
| D7 | Positional `SRC OUT` / some GPU-less CLI shapes | lean_single CLI | Madaros verb CLI; some options missing (`--show-ast` etc. per CLAUDE.md) | **docs-measured** |
| D8 | Current-source unit-type clinical gates | Rebuild+run lean_single | Tests seed semantics of units in monolith | **script-stated** |
| D9 | Parity / corpus tools | lean side of agree/diverge | Madaros-only would hide LEAN-ONLY / DIVERGE classes | **by design** until parity debt → 0 |

**Map of “what Madaros lacks to be canonical de facto”** = rows where class is
**gap** (D1, D2, D5, D7, D8) plus deciding the fate of **oracle** rows (D4, D9):
keep as frozen seed-oracle, not as product engine.

D3/D6 stay as **seed obligations**, not Madaros feature debt.

---

## 4. What is required for lean_single to be **only** seed

### 4.1 Must keep (seed role)

1. `lean_single.sio` + committed seed ELF + SeedReceipt provenance path  
2. `build_modular_madaros.sh` (or successor) deriving Madaros from seed  
3. Self-host / fixed-point gates on the **seed**  
4. Optional: frozen “oracle ELF” for dual-engine gates (may be the seed, but
   **not** advertised as the language)

### 4.2 Must change (engine role retirement)

1. **Full Test Suite** runs Madaros (or a Madaros-built stage artifact), with
   baselines re-derived — until then Madaros is not the suite authority (D2).  
2. **Package/PBPK/science gates** that pin lean_single for multimodule (D1)
   either go green on Madaros or become explicitly `requires: seed-oracle` with
   a sunset.  
3. **Defaults** in particle/science scripts: Madaros primary, lean only if
   dual-oracle.  
4. **Docs:** “canonical fixed-point” language attaches to **seed integrity**,
   not “the compiler users mean.” Product canonical = Madaros.  
5. **Wrapper:** fallback to lean_single remains degraded bootstrap, never claim
   path.  
6. **Claim surfaces:** no science/dissertation claim may cite
   `SOUNIO_SOUC_ENGINE=lean_single` as the semantic clock (already the intent;
   enforce by inventory of residual foreign judges).

### 4.3 Explicit non-goals of “seed only”

- Madaros bit-identical self-host before D1–D2 close (nice, not the definition
  of seed-only).  
- Deleting dual-engine **oracles** (D4/D9) — they need a reference pole; rename
  the pole from “the other language” to “bootstrap oracle.”

---

## 5. Suggested sequencing (design only)

1. **Publish this map** (this doc).  
2. **Close D1** (multimodule package/PBPK on Madaros) — unlocks science gates.  
3. **Re-home Full Test Suite** (D2) — largest semantic authority shift.  
4. **Retarget habit defaults** (D5) once dual matrices are green Madaros-first.  
5. **Doc/wrapper vocabulary** — seed vs oracle vs product.  
6. Only then: forbid product claims under lean_single engine env.

No step may claim “Madaros replaces lean_single” before D1+D2 have measured
green evidence on main CI.

---

## 6. Relation to SeedReceipt work

SeedReceipt answers: *is this seed ELF the fixed point of this source, with a
paper trail?*  

WS-C2 answers: *who is allowed to define the language when both engines exist?*

They compose: a honest seed is required to **build** Madaros; a honest Madaros
is required to **be** the language. Confusing the two is how two languages persist.

---

## 7. One-line summary

> lean_single is still seed + engine + fixed-point oracle. Seed-only means keep
> the first, retire the second as product authority, and rename the third to
> bootstrap oracle. The debt list is mainly multimodule/package science on
> Madaros (D1) and Full Test Suite still on souc-stage2 (D2). Without those,
> “Madaros is canonical” is decree, not fact.

*End of semantic declaration. No implementation in this document.*
