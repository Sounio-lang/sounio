<!-- docs:meta
topic_id: repo.docs.architecture.mir-port-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.mir-port-plan
-->

# MIR Port Plan — WS-C route study

**Status:** Route B **approved** (founder, 2026-08-16). This document is the
port plan + preflight amendments from
`docs/architecture/WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md` (fable-1). Route B is
**not** re-litigated; the amendments make PR1–PR2 land at the stated cost.  
**Author:** grok-cli1 (`ws-c-mir-study` / `amend-mir-port-plan`), 2026-08-16.  
**Plan context:** `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` §WS-C.  
**Frontier worktree:** `/workspace/.wt/mir-study` → `origin/canon/madaros-v2-sota` @ `97b5259497`.  
**Constraint (study):** did **not** edit `self-hosted/` on main; this file is the write product.

**Preflight amendments folded (2026-08-16):** C1 (payload census: **63**
files, not ~23), C2 (BASE_REF re-anchor), C4 (Madaros parse + dual receipt),
C5 (`loop_closed` semantic fix), C6 (WS-B ordering vs ENIR serialisation).
C1 authority: `docs/architecture/WS_C_PR1_PAYLOAD_CENSUS.md` (§4.1, §6.1).

---

## 1. Executive summary

A real ENIR→MIR pipeline already exists on the frontier tip and **still builds** at `97b525949`:

| Check | Result |
|---|---|
| Tip SHA | `97b525949765980406a4fefa7f533e9db89721e1` (`feat(c2): emit first divergence receipt (#837)`) |
| `self-hosted/enir/` | 14 modules, **7310** LOC |
| Seed compile `enir/driver.sio` | ELF produced under `souc-lean-single` + build lock; **reconfirmed under current main seed** by fable-1 preflight |
| `madaros-enir emit` | Exit 0; canonical ENIR artifact ~2053 bytes |
| Production wiring | Explicitly **not** imported by production codegen (`enir/mod.sio` banner) |
| Madaros v0.80.0 check | **Fails** on `mir_join.sio` (C4) until rewrite; other 13 enir files parse clean |

**Measured divergence** (do not use the swapped wording in an early draft of the focus plan):

```text
merge-base(origin/main, frontier) = a930c8ac72
commits only on frontier (main..frontier):  189
commits only on main     (frontier..main): 2086+  (main still moving)
```

| Route | Effort (order) | Risk | Recommendation |
|---|---:|---|---|
| **B · `enir/` subtree cherry-pick + resume lettered plan** | 1–2 weeks to green gates on main **after** PR1/PR2 preflight items below; then multi-week letter resume | **Lowest** | **Approved default** |
| **C · Fresh re-derivation from MIR_* + EISA docs** | multi-month | Medium–high scope | Only if B fails bounded repair budget |
| **A · Wholesale rebase of frontier onto main** | multi-week conflict storm | **Highest** | Reject as primary |

**Route B isolation claim survives adversarial re-measurement** (fable-1, 2026-08-16): all 36 `use` lines in enir are `enir::*`; both-sides conflict surface unchanged; driver builds under *current main* seed.

---

## 2. What the frontier actually carries

### 2.1 ENIR/MIR source tree (`self-hosted/enir/`)

| File | Role (from imports / gates) |
|---|---|
| `ir.sio` | ENIR core model |
| `parser.sio` / `canonical.sio` / `verify.sio` / `hash.sio` | Strict parse, canonicalize, verify, hash |
| `shadow_fixture.sio` | E1 fixture corpus surface |
| `source_lower.sio` | Source → ENIR (E2*) |
| `interpreter.sio` | Compiler-owned interpreter |
| `qd.sio` | qd128 arithmetic (E2E/E2F+) |
| `mir.sio` | Semantic MIR (E3A+) |
| `mir_cfg.sio` | CFG / Memory-SSA (E3C) |
| `mir_join.sio` | Multi-pred join MIR (E3D) — **Madaros parse break, §4.3** |
| `driver.sio` | CLI: emit / verify / roundtrip / lower / run / MIR cmds |
| `mod.sio` | Public boundary; **not** production codegen |

**Import topology (critical for port cost):** every `use` inside `enir/` is **`enir::*` only**. There is no `use ir::`, `use native::`, or `use compiler::` edge. ENIR is a **shadow lane** by design: gates assert zero-diff against production lowering/codegen/ABI surfaces.

### 2.2 Lettered plan state (implementation, not the four MIR_* research PDFs)

Normative narrative: `docs/architecture/MADAROS_V2_EISA_SEMANTIC_IR.md` (frontier-only; **absent** from `origin/main` at study time).

| Stage | Status at 97b525949 | Gate script |
|---|---|---|
| E1 shadow foundation | FULL | `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh` |
| E2A–E2H source→ENIR slices | FULL (corpus slices) | `madaros_v2_e2*_enir_*.sh` |
| E3A ENIR→MIR qd128 arith | FULL | `madaros_v2_e3a_enir_mir_qd128_gate.sh` |
| E3B MIR memory/move | FULL | `madaros_v2_e3b_enir_mir_memory_gate.sh` |
| E3C CFG / Memory-SSA (canonical loop) | FULL | `madaros_v2_e3c_cfg_memory_ssa_gate.sh` |
| E3D multipred scalar + multi-slot join | FULL | `madaros_v2_e3d_multipred_scalar_memory_ssa_gate.sh` |
| E3E equal-value distinct-event | present | `madaros_v2_e3e_equal_value_distinct_event_gate.sh` |

**Explicit non-claims after E3D:** general N-way SSA, nested diamonds, loops-in-same-schema, alias analysis, ABI selection, MachineIR, production codegen.

### 2.3 The four `docs/architecture/MIR_*` files

Present on **both** tips (with main-side churn). They are **research / Cranelift / optim strategy** docs, not the lettered ENIR implementation contract:

- `MIR_RESEARCH_STRATEGY.md`
- `MIR_OPTIMIZATION_STRATEGY.md`
- `MIR_OPTIMIZATION_PASES_DOCUMENTATION.md` (typo in filename preserved)
- `MIR_CRANELIFT_INTEGRATION_REPORT.md`

A “re-derive from the 4 MIR_* docs” route would **not** reconstruct the E1–E3D executable gates without also re-deriving `MADAROS_V2_EISA_SEMANTIC_IR.md` and the gate/Python verifiers.

### 2.4 Build evidence

```text
worktree: /workspace/.wt/mir-study @ 97b525949
seed:     bin/souc-lean-single-x86_64 (frontier and current-main seeds both produce ELF)
cmd:      scripts/dev/souc-build-lock.sh <seed> self-hosted/enir/driver.sio <out>
result:   ELF written; driver emit OK (~2053 B artifact)
note:     seed path logs E200 `loop_closed` — see §4.4 (semantic, not noise)
Madaros:  souc check fails on mir_join.sio lines 476–489 — see §4.3
```

Full E1–E3D gate matrix was **not** re-run end-to-end in wave 1 (Python-heavy + multi-hour). Build+emit is the hard existence proof for “still builds at 97b525949”.

---

## 3. Divergence census (measured 2026-08-16)

### 3.1 Commit counts

| Direction | Count | Meaning |
|---|---:|---|
| `origin/main..frontier` | **189** | Frontier-only commits (main lacks) |
| `frontier..origin/main` | **2086+** | Main-only commits (frontier lacks; tip still moves) |
| Merge-base | `a930c8ac72` | `docs(governance): register tuple-let-desugar audit doc` |

**Correction to early focus-plan wording:** those drafts swapped these numbers. **Main is the long tip; frontier is a short divergent lobe (~189 commits) with the ENIR lane.**

### 3.2 Tree-level

| Set | Count |
|---|---:|
| Paths differing tip-to-tip (`git diff --name-only HEAD...origin/main`) | ~4202 |
| of which under `self-hosted/` | ~197 |
| Frontier-only paths (not on main at all) | **276** |
| of which `self-hosted/enir/*` | **14** (entire tree missing on main) |
| Main `self-hosted/enir/` | **0** |

### 3.3 Both-sides edits since merge-base (true rebase conflict surface)

Files modified on **both** frontier and main since merge-base: **115** total, **41** under `self-hosted/`.

Highest main-side churn on shared hot files (numstat lines since merge-base):

| Path | Main churn | Frontier churn |
|---|---:|---:|
| `self-hosted/ir/lower.sio` | ~14178 | ~1880 |
| `self-hosted/compiler/main.sio` | ~6737 | ~2453 |
| `self-hosted/compiler/lean_single.sio` | ~5014 | ~763 |
| `self-hosted/compiler/module_frontend.sio` | ~4778 | ~1677 |
| `self-hosted/native/codegen_x86_linux.sio` | ~3906 | ~1211 |
| `self-hosted/ir/ir.sio` | ~2121 | ~122 |

**None of these are required imports of `enir/`.** They matter only for routes that replay frontier *compiler* history or re-open production IR. They **do** matter for gate frozen-surface checks (§4.2).

### 3.4 Frontier-only ENIR-related commit set

~**15** commits touch `self-hosted/enir`, `scripts/dev/madaros_v2_e*`, and `MADAROS_V2_EISA_SEMANTIC_IR.md` (linear lettered delivery E1→E3D/E3E). Representative:

```text
3795ac03a0 feat(madaros): add compiler-owned ENIR shadow
… E2A–E2H …
6129a9f990 feat(enir): add translation-validated semantic MIR   # E3A
73542db065 compiler: add E3B semantic MIR memory
9f4ec85a21 compiler: add E3C explicit CFG and Memory SSA
ce2f94407f compiler: add E3D multipred scalar and memory SSA
b4142c83cb test(enir): distinguish equal values by event receipt  # E3E
```

### 3.5 Main-only commits on IR/compiler/native (port friction if you merge wholesale)

`git rev-list --count frontier..main -- self-hosted/ir self-hosted/compiler self-hosted/check self-hosted/native` ≈ **467** commits. These include Lane B stack peels, IR arena/SoA, module frontend multi-mod walls, f32 coercion, etc. They do **not** delete or redefine `enir/` (it never existed on main).

---

## 4. Preflight amendments (Route B, approved — implementation detail)

Findings from `docs/architecture/WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md`. None overturn Route B.

### 4.1 C1 · Gate payload files (PR1/PR2 inventory)

**Authority:** `docs/architecture/WS_C_PR1_PAYLOAD_CENSUS.md` (codex-2,
2026-08-16). Source refs in that file: `origin/main` @ `03416657fa`,
`origin/canon/madaros-v2-sota` @ `97b525949`.

**Headline (preflight was low by ~3×):** the preflight’s “~23 files” figure
counted only the `tools/eisa` delta. Following every reference the gate
scripts actually make — shell references, Python verifier references,
verifier `PROGRAMS` fixture expansions, recursive regression-gate calls, and
the `self-hosted/enir/driver.sio` import closure each gate build needs — the
requested **E1/E2/E3** gate stack needs **63 files absent from `origin/main`**:

| Class | Count | Role |
|---|---:|---|
| Gate scripts (`scripts/dev/madaros_v2_e*`) | **14** | E1, E2A–E2H, E3A–E3E shells |
| Python verifiers | **13** | Pair with gates (E3E has shell only) |
| `self-hosted/enir/` driver-closure | **14** | Common build closure for every gate |
| `tools/eisa/` oracle + fixture | **22** | First needed at E2B; grows through E3E |
| **Total add-list** | **63** | |

**Excluded (not in the 63):**

- `tools/eisa/eisa_enir_c2_rump.eisa` — frontier-only, but owned by the **C2**
  gate (`madaros_v2_c2_v0_gate.sh`), not E1/E2/E3.
- `$TMP_DIR` negative fixtures generated inside gate scripts — not repository
  payload.

**Per-gate transitive counts** (from the census; each includes the shared 14
enir files + inherited regression chain):

| Gate | Transitive missing on main |
|---|---:|
| E1 shadow | 16 |
| E2A lowering | 18 |
| E2B CFG | 21 |
| E2C fuel/blockargs | 24 |
| E2D rump DD | 27 |
| E2E qd128 | 35 |
| E2F rump qd | 38 |
| E2G fuel/control/frail | 43 |
| E2H memory/move/poison | 48 |
| E3A MIR qd128 | 50 |
| E3B MIR memory | 52 |
| E3C CFG memory SSA | 56 |
| E3D multipred SSA | 60 |
| E3E equal-value distinct event | 17 (no regression chain; +2 fixtures) |

Exact path lists and “which PR carries what” are decided by the **per-gate
transitive lists** in `WS_C_PR1_PAYLOAD_CENSUS.md` — not by this summary table
alone. Consolidated add-list is the union (63).

**Present-but-changed watchlist** (on main already; content differs from
canon — not add-list, but PR2+ may need behavioural accounting / C3):
`bin/souc-lean-single-x86_64`, `scripts/dev/souc-build-lock.sh`,
`self-hosted/compiler/main.sio`, `stdlib/math/qd128.sio`,
`tools/eisa/eisa_evm_run.sio`.

Also flag (preflight C3): shared oracles above plus `stdlib/eisa/` have
**drifted** on main; WS-F (EISA Madaros port) collocates on the same surface —
coord claim boundary before PR2+.

### 4.2 C2 · BASE_REF re-anchoring decision (PR2 cost — gate scripts)

**Problem.** Every E-gate opens with approximately:

```bash
BASE_REF="${E1_BASE_REF:-origin/canon/madaros-v2-sota}"   # or E2*_BASE_REF / E3*_BASE_REF
git diff --quiet "$BASE_REF" -- \
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native \
  self-hosted/wasm self-hosted/gpu stdlib/runtime \
  [stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio tools/eisa/eisa_evm_run.sio …]
```

On main that diff against the **frontier branch name** is never empty
(thousands of lines of main-side IR/native drift). **Every gate fails its
shadow-discipline check before doing any work** if left as-is.

**Options and judgment**

| Anchor | Behaviour on main | Verdict |
|---|---|---|
| Keep `origin/canon/madaros-v2-sota` | Always fails on main | **Unusable** |
| Default `origin/main` | Clean tree OK if PR is main-based and only adds enir; **spuriously fragile** when collocated lanes leave dirty `ir/lower.sio` (constant on this pod) *or* when comparing a lagging PR branch to a racing main tip that already moved production surfaces | **Reject as default** |
| Default `HEAD` | Checks worktree cleanliness vs this commit; empty for a clean checkout of a PR that only added enir. Weak as a “did the PR modify production IR?” check alone, but matches E3D’s internal `E3C_BASE_REF=HEAD` precedent for cascade regressions | **Accept for local/cascade** |
| **PR-range pin (recommended CI default)** | `MERGE_BASE=$(git merge-base HEAD origin/main)` then `git diff --quiet "$MERGE_BASE" HEAD -- <prod surfaces>` — fails iff the **PR commit range** touches production IR/codegen/ABI. Independent of frontier tip and of other agents’ dirty trees on other worktrees | **Required for PR2+ CI** |

**Decision (document for PR2 implementers):**

1. **Primary CI mode (E1 standalone and every top-level E-gate on main):**  
   Replace the frontier default with a **PR-range frozen-surface check**:
   ```bash
   MERGE_BASE=$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")
   git diff --quiet "$MERGE_BASE" HEAD -- \
     self-hosted/compiler/main.sio self-hosted/ir self-hosted/native \
     self-hosted/wasm self-hosted/gpu stdlib/runtime
   # plus any extra pins (eisa/qd) as today, same MERGE_BASE..HEAD range
   ```
   Meaning: *this PR must not edit production lowering/codegen/ABI surfaces*.
   Env override `E1_BASE_REF` / `ENIR_GATE_BASE` remains for emergency pins to a
   specific SHA (e.g. last ENIR-green tag).

2. **Cascade regressions (E3D → E3C → … → E1):** keep and generalise the E3D
   pattern: `E3C_BASE_REF=HEAD` + `E3C_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1` (and
   siblings) **after** the parent gate has already protected production
   surfaces and rebuilt historical witnesses. Do not invent a second discipline.

3. **Collocated dirty worktrees:** gates are **CI / clean-tree tools**. A dirty
   `self-hosted/ir/lower.sio` in the same worktree will still fail
   `git diff HEAD -- self-hosted/ir` — that is correct. The fix is not
   `BASE_REF=origin/main`; it is run gates on clean checkouts or dedicated
   worktrees (Attention Charter: control is not bench).

4. **Boundary honesty (overrides Route B step 3 as originally worded):**  
   “Fix **inside `enir/` only** until driver ELF green” remains true for
   **enir source** typecheck/parse/semantic repairs in **PR1**.  
   It **does not** hold for **gate scripts**: PR2 **must** edit
   `scripts/dev/madaros_v2_e*_*.sh` (BASE_REF re-anchor, possibly payload
   paths). That is an **unbudgeted PR2 cost item** relative to the original
   “days 4–10 cascade” estimate — budget **+1–2 days** for gate surgery and
   CI wiring, not “search-replace the branch name.”

### 4.3 C4 · Madaros parse break + dual PR1 receipt

**Receipt (fable-1):** default Madaros `souc check self-hosted/enir/mir_join.sio`
→ parse errors at lines **476–489**; `driver.sio` / `mod.sio` fail transitively.
Other **13** enir files parse clean under Madaros v0.80.0. Seed accepts the
file.

**Named PR1 tasks (enir-local rewrites, seed-only constructs):**

| Site | Construct Madaros rejects | Fix shape |
|---|---|---|
| `mir_join.sio:476` | Parenthesised **`if`-expression** used as a comparison operand inside an `\|\|` chain (`b.trap_kind != (if opcode >= … { … } else { … })`) | Hoist to a `let trap = if … { … } else { … }` **before** the condition, then compare `b.trap_kind != trap` |
| `mir_join.sio:481` | **Semicolon-joined `let` statements** on one line (`let a = …; let b = …`) | Split into two successive `let` lines (no `;`) |

**PR1 acceptance (amended):**

1. Seed: `souc-lean-single` (or locked seed path) **compiles** `enir/driver.sio` to ELF; optional `emit` smoke.  
2. **Madaros:** `bin/souc check self-hosted/enir/mod.sio` (or `driver.sio`) **green**, **or** a dated FAIL_HONEST list of residual codes with owners — not “seed-only green.”  
3. No silent merge of seed-only syntax that the default engine cannot parse.

Without (2), PR1 lands code the primary compiler cannot load and the failure
shows up as a mystery later.

### 4.4 C5 · `loop_closed` is a semantic fix, not a syntax papercut

**Facts:**

- **Use** at `source_lower.sio:529`:  
  `if loop_closed && !is_load { return enir_lower_fail(module, line, 15) }`  
  (rule: no non-load `let` after loop close.)
- **Declaration** at `source_lower.sio:645`: `var loop_closed = false` in a
  **different** function (the CFG/`while` lowerer), with assignments at 672+ in
  that same second function.
- Seed path compiles the use-site as a **guarded** gate (`gates[… guarded=…]`),
  so lower-fail-15 is **silently unenforced or trap-prone** under seed today.
  Gate green under that seed proves less than it appears to.
- Madaros will **hard-error** (unknown identifier / E200 class) once C4 is fixed
  and the driver is checked under Madaros.

**Budgeting:** treat as a **semantic repair** in PR1 (or a blocking PR1.1 before
any E2 gate claim): thread `loop_closed` into the correct function scope, or
delete/replace the check with an equivalent well-scoped rule, then **re-run
affected E2 gates** (at least E2B/E2C/E2G class that exercise loop close). Do
**not** classify as a one-line syntax papercut or “non-fatal noise.”

### 4.5 C6 · WS-B SOIR sequencing does not protect ENIR serialisation

The focus plan’s “SOIR gate before the port lands” rationale catches
serialisation drift on **`self-hosted/ir/*` (SoIR)**. ENIR has its **own** text
format, with canonical/roundtrip/hash checks **inside E1**.

| Premise | Holds? |
|---|---|
| SOIR-first is still fine fleet sequencing (WS-B before heavy PR2+) | Yes, as hygiene |
| SOIR green mechanically protects ENIR serialisation | **No** |
| A WS-B slip must hold **PR1** hostage | **No** |

**PR1** (enir sources + dual check receipts + docs + payload if censused) may
proceed on the SOIR track in parallel. **PR2+** (gate matrix) should still
prefer a green SOIR gate when both compete for CI attention, but PR1 is not
blocked on a false ENIR-safety premise.

---

## 5. Route costings

### Route A — Wholesale rebase of frontier onto current main

**What it is:** take `canon/madaros-v2-sota` (189 commits) and rebase onto
`origin/main` (or merge main into frontier and resolve).

**Pros**
- Preserves full historical lettered receipts and S1–S5 / C2 scaffolding scripts if desired.
- One-shot “bring the sota branch home.”

**Cons / cost**
- Must resolve **115 both-sides files**, including the worst IR/compiler/native hotspots above.
- ~2086 main commits of semantic drift in the same files frontier also touched lightly.
- Frontier also carries **276** paths main never had — noise and review load.
- Risk of re-introducing fixed-point / stack / SoA regressions fixed on main after the fork.
- Wall-clock: multi-week expert conflict resolution; high probability of “rebasing forever.”

**Effort estimate:** 3–6+ engineer-weeks of pure conflict surgery before any new MIR work.  
**Verdict:** **Not recommended as primary.**

---

### Route B — `enir/` subtree cherry-pick + resume lettered plan (**approved**)

**What it is:**

1. On a branch from **current main**, add the frontier payload per
   `WS_C_PR1_PAYLOAD_CENSUS.md` (**63** files total; see §4.1 / §6.1):
   - **PR1:** `self-hosted/enir/**` (**14** driver-closure files) with **C4/C5
     repairs**, `bin/madaros-enir`, `MADAROS_V2_EISA_SEMANTIC_IR.md`
   - **PR2:** remaining **49** files = 14 gate scripts + 13 Python verifiers +
     22 `tools/eisa/` oracles/fixtures, with **§4.2 BASE_REF** re-anchor on
     every landed gate script
2. Wire gates into CI behind E1–E3D (or umbrella `enir_pipeline_gate.sh` first).
3. **Compile driver under seed *and* Madaros check** (PR1 acceptance §4.3).  
   Enir **source** fixes stay inside `enir/` until both receipts green.  
   **Gate scripts are out of that boundary** (§4.2 point 4).
4. Re-run E1 then cascade E2*→E3D; mark FAIL_HONEST with receipts rather than silent skip.
5. Resume lettered plan at the first non-FULL item after E3D/E3E (general multipred/N-way SSA, alias, then ABI/MIR→codegen — still deferred by the E3D contract).

**Pros**
- `enir/` is **dependency-isolated** (`use enir::*` only) → near-zero textual conflict with main’s `ir/lower.sio` wars.
- Main lacks the directory entirely → add is clean.
- Preserves executable gates + Python oracles.
- Aligns with shadow-lane discipline (once BASE_REF is re-anchored for main).

**Cons / cost (updated after C1 census)**
- Must re-validate every gate under **current** seed/Madaros.
- **PR1:** Madaros parse rewrites (`mir_join.sio`) + **semantic** `loop_closed` fix + dual receipts. File count stays **14 enir** (already budgeted); hard work is repair, not bulk.
- **PR2 grows vs preflight:** not “~23 `tools/eisa` + E1”, but **49**
  non-enir files (14 scripts + 13 verifiers + 22 fixtures). BASE_REF surgery
  multiplies across **14** gate scripts, not one. Possible oracle re-validation
  vs main’s drifted EISA (C3 / present-but-changed watchlist).
- Shared-surface coord with WS-F.
- Multi-week for true production MIR→codegen remains after E3D (by design).

**Effort estimate (re-checked against 63-file census):**
- **PR1 (days 1–4 — holds):** 14 enir subtree add + C4 rewrites + C5 semantic fix + seed ELF + Madaros check receipts + wrapper/docs. **Does not absorb the 49-file gate/payload surprise.**
- **PR2 (days 5–10 — +2 days vs prior “5–8”):** bulk-add the 49 gate/payload files; BASE_REF re-anchor on all 14 gate scripts; E1 green as acceptance (E2+/E3 scripts may land present-but-not-yet-green). Mechanical bulk is cheap; BASE_REF ×14 + C3 drift is the real cost.
- **PR3–4 (days 11–16+):** E2A–E3D cascade under main toolchain — **file-add work shrinks** if PR2 already landed the full 49; cascade is mostly green/FAIL_HONEST.
- Wave 2+: resume lettered expansion (post-E3D scope).

**PR-stack shape verdict:** **PR1/PR2 split still holds** — do **not** dump
gates into PR1 (would muddle dual-receipt acceptance) and do **not** hold PR1
for the 49-file payload. Prefer PR2 as the **bulk gate+payload** landing so
PR3–4 are cascade-only. Optional micro-split (PR2a = E1+BASE_REF pattern,
PR2b = remaining payload) only if review bandwidth forces it; default is one
PR2 bulk add.

**Verdict:** **Approved default.** Isolation claim holds; C1 multiplies PR2
review surface (~3× preflight tools/eisa estimate), not the route choice.

---

### Route C — Fresh re-derivation from the four `MIR_*` docs (+ EISA architecture)

**What it is:** ignore frontier `enir/` source; redesign MIR against current main
using the four `MIR_*` research docs, `MADAROS_V2_EISA_SEMANTIC_IR.md`, and
optional reading of frontier gates as oracles only.

**Pros**
- Avoids carrying frontier compiler history; can target MLI handoff cleanly.

**Cons / cost**
- Throws away **7310 LOC** + ~15 sequenced gates with independent Python checkers.
- The four `MIR_*` docs alone do **not** specify E2 source grammar, qd128 ENIR ops, or E3D join schema in executable detail.
- Calendar: multi-month to E3D-equivalent evidence.

**Verdict:** **Fallback only** if Route B fails a bounded enir-compile budget after C4/C5 (e.g. >2 weeks of enir-only fixes with no dual green receipts).

---

## 5. Conflict census implications by route

| Surface | Route A | Route B | Route C |
|---|---|---|---|
| `self-hosted/enir/**` | must merge + re-sync | **add** (no main counterpart) + C4/C5 edits | rewrite |
| E1–E3D gates/scripts | massive path noise + conflicts | **add** + **rewrite BASE_REF** (PR2) | rewrite |
| `tools/eisa/` fixtures | noise | **add** per C1 census | rewrite |
| `ir/lower.sio`, `module_frontend.sio`, `codegen_x86_linux.sio` | **hard conflicts** | **untouched** by enir source; **pinned by gate checks** | optional later |
| Production Madaros | high regression risk | protected once PR-range pin is correct | protected until wire-up |
| Founder ≤2 self-hosted writers rule | violates quickly | enir-only writers for PR1; PR2 is scripts/tools | design-first |

---

## 6. Recommended decision and PR stack

1. **Route B remains the WS-C port route** (founder-approved; fable-1 survived).  
2. **Do not** open Route A unless Route B fails its bounded repair budget.  
3. Keep Route C as escape hatch only.  
4. **PR1 is not hostage to WS-B** (§4.5); fleet may still prefer SOIR green before PR2 cascade for CI bandwidth.  
5. After E1–E3D green on main, treat **post-E3D general SSA / ABI / MachineIR** as a new lettered tranche — do not silently expand E3D claims (also relevant to WS-D MLI S2 feed).

### 6.1 Suggested PR stack (Route B, amended — C1 folded)

**Census authority:** `docs/architecture/WS_C_PR1_PAYLOAD_CENSUS.md`.
Per-gate transitive lists there decide which PR must already contain which
paths for a given gate to run. Union add-list = **63** files
(14 enir + 14 scripts + 13 verifiers + 22 `tools/eisa`). Exclude C2-only
`eisa_enir_c2_rump.eisa` and `$TMP_DIR` fixtures.

| PR | Content (file budget from census) | Acceptance (must) |
|---|---|---|
| **PR1** | **14** `self-hosted/enir/**` (driver closure) + `bin/madaros-enir` + `MADAROS_V2_EISA_SEMANTIC_IR.md`; **rewrite `mir_join.sio` C4 sites**; **fix `loop_closed` C5**. **No** gate scripts / verifiers / `tools/eisa` bulk (keeps dual-receipt surface clean) | (a) seed driver ELF; (b) **Madaros `souc check` green** (or FAIL_HONEST list); (c) no production IR/codegen edits |
| **PR2** | **49** remaining add-list files: **14** `scripts/dev/madaros_v2_e*` gates + **13** Python verifiers + **22** `tools/eisa/` oracles/fixtures; **BASE_REF re-anchor per §4.2 on every landed gate script** | **E1 green** on clean CI (E1 transitive = 16 = 14 enir from PR1 + E1 `.sh` + E1 `.py`); PR-range frozen-surface check fails closed if IR touched; remaining E2/E3 scripts may be present-but-not-green |
| **PR3** | E2A–E2H cascade (may split); **no new payload** if PR2 bulk-landed the 49 | Cascade under re-anchored discipline; oracle policy vs main drift (C3 / present-but-changed watchlist) documented |
| **PR4** | E3A–E3D (+ E3E) cascade | FULL gates green or FAIL_HONEST with receipts |
| **PR5** | Optional umbrella `scripts/ci/enir_pipeline_gate.sh` + madaros_full_gate hooks | CI wiring only |

**Effort vs preflight (~23 files):**

| Item | Pre-census estimate | Post-census (this amendment) |
|---|---|---|
| PR1 | days 1–4 | **days 1–4 holds** (14 enir + C4/C5; not the 3× surprise) |
| PR2 | days 5–8 | **days 5–10** (49-file bulk + BASE_REF ×14 scripts + E1 green) |
| PR3–4 | days 9–14+ | **days 11–16+** if PR2 bulk-lands; cascade-heavy, add-light |
| Stack shape | PR1 enir / PR2 E1+payload / PR3–4 cascade | **Shape holds.** Do not move the 49 into PR1. Prefer one PR2 bulk over splitting PR1. |

No production `use enir::` from `compiler/main.sio` until a later explicit integration tranche (post-WS-D MLI design).

---

## 7. Reproduction commands

```bash
# Worktree (study)
git fetch origin canon/madaros-v2-sota main
git worktree add /workspace/.wt/mir-study origin/canon/madaros-v2-sota
cd /workspace/.wt/mir-study
git rev-parse HEAD   # expect 97b5259497…

# Divergence
git merge-base origin/main HEAD
git rev-list --count origin/main..HEAD   # frontier-only
git rev-list --count HEAD..origin/main   # main-only

# Build ENIR driver (seed)
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 \
  self-hosted/enir/driver.sio /tmp/madaros-enir
/tmp/madaros-enir emit | head

# Madaros parse (C4) — expect fail on unpatched mir_join
bin/souc check self-hosted/enir/mir_join.sio

# Both-sides conflict list
MB=$(git merge-base origin/main HEAD)
comm -12 <(git diff --name-only $MB HEAD | sort) \
         <(git diff --name-only $MB origin/main | sort)

# tools/eisa delta only (incomplete vs full gate stack — use census)
comm -23 <(git ls-tree -r --name-only origin/canon/madaros-v2-sota tools/eisa | sort) \
         <(git ls-tree -r --name-only origin/main tools/eisa | sort)
# Full E1/E2/E3 add-list: docs/architecture/WS_C_PR1_PAYLOAD_CENSUS.md (63 files)
```

---

## 8. Out of scope / non-claims

- Did not run the full E1–E3D wall-clock gate matrix on this pod.  
- Did not modify `self-hosted/` on main in the study or this amendment.  
- Did not decide MLI operand model (WS-D) — see preflight D1 for Route-B MIR vs MLI S2 gap.  
- Did not claim production MIR→x86.  
- Did not re-measure divergence after every main tip motion.

---

## 9. Handoff

| Item | Location |
|---|---|
| Study worktree | `/workspace/.wt/mir-study` @ `97b525949` |
| Amendment worktree | `/workspace/.wt/amend-mir` @ branch `amend/mir-port-plan-20260816` |
| Preflight review | `docs/architecture/WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md` |
| C1 payload census | `docs/architecture/WS_C_PR1_PAYLOAD_CENSUS.md` (**landed**; **63** = 14+13+14+22) |
| Coord claim | `grok-cli1` / `amend-mir-port-plan` (release after this amendment) |
| Next port action | PR1 per §6.1 (14 enir + C4/C5 + dual receipts); PR2 bulk 49 + BASE_REF ×14 + E1 green |

*End of WS-C port plan (route study + preflight amendments + C1 census fold).*
