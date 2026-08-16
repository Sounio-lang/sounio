<!-- docs:meta
topic_id: repo.docs.architecture.mir-port-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.mir-port-plan
-->

# MIR Port Plan — WS-C route study

**Status:** route recommendation for founder decision (wave 1, read-only study).  
**Author:** grok-cli1 (`ws-c-mir-study`), 2026-08-16.  
**Plan context:** `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` §WS-C.  
**Frontier worktree:** `/workspace/.wt/mir-study` → `origin/canon/madaros-v2-sota` @ `97b5259497`.  
**Constraint:** this study did **not** edit `self-hosted/` on main; only this doc is the write product.

---

## 1. Executive summary

A real ENIR→MIR pipeline already exists on the frontier tip and **still builds** at `97b525949`:

| Check | Result |
|---|---|
| Tip SHA | `97b525949765980406a4fefa7f533e9db89721e1` (`feat(c2): emit first divergence receipt (#837)`) |
| `self-hosted/enir/` | 14 modules, **7310** LOC |
| Seed compile `enir/driver.sio` | ELF produced (`/tmp/mir-study-enir/madaros-enir`, ~1.4 MiB) under `souc-lean-single` + build lock |
| `madaros-enir emit` | Exit 0; canonical ENIR artifact ~2053 bytes |
| Production wiring | Explicitly **not** imported by production codegen (`enir/mod.sio` banner) |

**Measured divergence** (do not use the swapped wording in an early draft of the focus plan):

```text
merge-base(origin/main, frontier) = a930c8ac72
commits only on frontier (main..frontier):  189
commits only on main     (frontier..main): 2086
```

| Route | Effort (order) | Risk | Recommendation |
|---|---:|---|---|
| **B · `enir/` subtree cherry-pick + resume lettered plan** | 1–2 weeks to green gates on main; then multi-week letter resume | **Lowest** | **Recommended default** |
| **C · Fresh re-derivation from MIR_* + EISA docs** | multi-month | Medium–high scope | Only if A/B prove hostile |
| **A · Wholesale rebase of frontier onto main** | multi-week conflict storm | **Highest** | Reject as primary |

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
| `mir_join.sio` | Multi-pred join MIR (E3D) |
| `driver.sio` | CLI: emit / verify / roundtrip / lower / run / MIR cmds |
| `mod.sio` | Public boundary; **not** production codegen |

**Import topology (critical for port cost):** every `use` inside `enir/` is **`enir::*` only**. There is no `use ir::`, `use native::`, or `use compiler::` edge. ENIR is a **shadow lane** by design: gates assert zero-diff against production lowering/codegen/ABI surfaces.

### 2.2 Lettered plan state (implementation, not the four MIR_* research PDFs)

Normative narrative: `docs/architecture/MADAROS_V2_EISA_SEMANTIC_IR.md` (frontier-only on main tree; **absent** from `origin/main`).

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

### 2.4 Build evidence (this session)

```text
worktree: /workspace/.wt/mir-study @ 97b525949
seed:     bin/souc-lean-single-x86_64
cmd:      scripts/dev/souc-build-lock.sh <seed> self-hosted/enir/driver.sio /tmp/mir-study-enir/madaros-enir
result:   ELF written; driver emit OK
note:     seed log showed E200 `loop_closed` at line 529 (non-fatal under this seed path); ELF still emitted
```

Full E1–E3D gate matrix was **not** re-run end-to-end in wave 1 (Python-heavy + multi-hour). Build+emit is the hard existence proof requested for “still builds at 97b525949”.

---

## 3. Divergence census (measured 2026-08-16)

### 3.1 Commit counts

| Direction | Count | Meaning |
|---|---:|---|
| `origin/main..frontier` | **189** | Frontier-only commits (main lacks) |
| `frontier..origin/main` | **2086** | Main-only commits (frontier lacks) |
| Merge-base | `a930c8ac72` | `docs(governance): register tuple-let-desugar audit doc` |

**Correction to focus-plan wording:** an early draft swapped these numbers. **Main is the long tip; frontier is a short divergent lobe (~189 commits) with the ENIR lane.**

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

**None of these are required imports of `enir/`.** They matter only for routes that replay frontier *compiler* history or re-open production IR.

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

## 4. Route costings

### Route A — Wholesale rebase of frontier onto current main

**What it is:** take `canon/madaros-v2-sota` (189 commits) and rebase onto `origin/main` (or merge main into frontier and resolve), landing everything the frontier carried.

**Pros**
- Preserves full historical lettered receipts and S1–S5 / C2 scaffolding scripts if desired.
- One-shot “bring the sota branch home.”

**Cons / cost**
- Must resolve **115 both-sides files**, including the worst IR/compiler/native hotspots above (orders of magnitude more main churn than frontier).
- ~2086 main commits of semantic drift in the same files frontier also touched lightly.
- Frontier also carries **276** paths main never had (research tarballs, GPU experiments, paper bundles) — noise and review load.
- Risk of re-introducing fixed-point / stack / SoA regressions fixed on main after the fork.
- Wall-clock: multi-week expert conflict resolution; high probability of “rebasing forever.”

**Effort estimate:** 3–6+ engineer-weeks of pure conflict surgery before any new MIR work.  
**Verdict:** **Not recommended as primary.** Use only if founder requires full frontier history, not just ENIR/MIR capability.

---

### Route B — `enir/` subtree cherry-pick + resume lettered plan (**recommended**)

**What it is:**

1. On a branch from **current main**, add the frontier-only payload almost verbatim:
   - `self-hosted/enir/**` (14 files)
   - `scripts/dev/madaros_v2_e1`…`e3e_*.{sh,py}` (+ related verify helpers)
   - `bin/madaros-enir` wrapper
   - `docs/architecture/MADAROS_V2_EISA_SEMANTIC_IR.md` (and optionally refresh the four `MIR_*` docs)
2. Wire gates into `scripts/ci/madaros_full_gate.sh` behind the same E1–E3D sequence (or a new `enir_gate.sh` umbrella first).
3. **Compile driver under current Madaros/seed** — expect Sounio-syntax/effect residuals from 1 month of main language drift; fix *inside enir/* only until driver ELF green.
4. Re-run E1 then cascade E2*→E3D; mark FAIL_HONEST with receipts rather than silent skip.
5. Resume lettered plan at the first non-FULL item after E3D/E3E (general multipred/N-way SSA, alias, then ABI/MIR→codegen — still deferred by the E3D contract).

**Pros**
- `enir/` is **dependency-isolated** (`use enir::*` only) → near-zero textual conflict with main’s `ir/lower.sio` wars.
- Main lacks the directory entirely → add is clean, not a 3-way merge of shared blobs.
- Preserves executable gates + Python oracles (the real value, not just prose).
- Aligns with shadow-lane discipline already written into E1 (zero-diff production surfaces).
- Keeps WS-B SOIR gate sequencing: land SOIR first, then attach ENIR gates so serialization drift is caught.

**Cons / cost**
- Must still re-validate every gate under **current** seed/Madaros (frontier binaries are not evidence on main).
- Seed may surface effect/E200 residuals (seen once on driver build); budget a small repair pass **inside `enir/`**.
- Does not automatically bring S5 MIR-ABI / wide-sret experiment scripts unless explicitly listed.
- Multi-week for true production MIR→codegen remains after E3D (by design).

**Effort estimate:**
- Days 1–3: subtree add + driver compile on main + E1 green.
- Days 4–10: E2A–E3D cascade under main toolchain; fix enir-local breaks.
- Wave 2+: resume lettered expansion (post-E3D scope).

**Verdict:** **Default route.** Lowest conflict surface, highest reuse of proven work, clear resume point (E3D FULL → general SSA/ABI still open).

---

### Route C — Fresh re-derivation from the four `MIR_*` docs (+ EISA architecture)

**What it is:** ignore frontier `enir/` source; redesign MIR against current main using:
- the four `MIR_*` research docs (already on main),
- `MADAROS_V2_EISA_SEMANTIC_IR.md` (must still be ported as a doc),
- optional reading of frontier gates as oracles only.

**Pros**
- Avoids carrying any frontier compiler history.
- Can target MLI handoff (WS-D) with a cleaner module layout from day one.
- No 115-file rebase.

**Cons / cost**
- Throws away **7310 LOC** of working ENIR/MIR + ~15 sequenced gates with independent Python checkers.
- The four `MIR_*` docs alone do **not** specify E2 source grammar, qd128 ENIR ops, or E3D join schema in executable detail.
- Re-implementation risk of silent semantic drift vs Metron/EISA oracles.
- Calendar: multi-month to reach E3D-equivalent evidence.

**Effort estimate:** 2–4+ months to parity with frontier E3D.  
**Verdict:** **Fallback only** if Route B’s enir sources cannot be made to typecheck under main after a bounded repair budget (e.g. >2 weeks of enir-only fixes with no E1 green).

---

## 5. Conflict census implications by route

| Surface | Route A | Route B | Route C |
|---|---|---|---|
| `self-hosted/enir/**` | must merge + re-sync | **add** (no main counterpart) | rewrite |
| E1–E3D gates/scripts | massive path noise + conflicts | **add** frontier-only scripts | rewrite |
| `ir/lower.sio`, `module_frontend.sio`, `codegen_x86_linux.sio` | **hard conflicts** (both-sides) | **untouched** if shadow discipline holds | optional later |
| Production Madaros | high regression risk | protected by E1 zero-diff contract | protected until explicit wire-up |
| Founder ≤2 self-hosted writers rule | violates quickly (long rebase) | can stay enir-only writers | design-first, delayed code |

---

## 6. Recommended decision

1. **Choose Route B** as the WS-C port route.  
2. **Do not** open a wholesale frontier rebase (Route A) unless Route B fails its bounded enir-compile budget.  
3. Keep Route C as the documented escape hatch, not the default.  
4. Sequencing (from focus plan, unchanged): **P0-F + WS-A baseline**, **WS-B SOIR gate**, then Route B port PR stack.  
5. After E1–E3D green on main, treat **post-E3D general SSA / ABI / MachineIR** as a new lettered tranche — do not silently expand E3D claims.

### Suggested PR stack (Route B)

| PR | Content |
|---|---|
| PR1 | `self-hosted/enir/**` + `bin/madaros-enir` + docs (`MADAROS_V2_EISA_SEMANTIC_IR.md`) only; driver `check`/`compile` under seed |
| PR2 | E1 gate + Python verifier green on main |
| PR3 | E2A–E2H gates (may split) |
| PR4 | E3A–E3D (+ E3E) gates |
| PR5 | Optional: umbrella `scripts/ci/enir_pipeline_gate.sh` + madaros_full_gate hooks |

No production `use enir::` from `compiler/main.sio` until a later explicit integration tranche (post-WS-D MLI design).

---

## 7. Reproduction commands

```bash
# Worktree (already created for this study)
git fetch origin canon/madaros-v2-sota main
git worktree add /workspace/.wt/mir-study origin/canon/madaros-v2-sota
cd /workspace/.wt/mir-study
git rev-parse HEAD   # expect 97b5259497…

# Divergence
git merge-base origin/main HEAD
git rev-list --count origin/main..HEAD   # frontier-only
git rev-list --count HEAD..origin/main   # main-only

# Build ENIR driver
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 \
  self-hosted/enir/driver.sio /tmp/madaros-enir
/tmp/madaros-enir emit | head

# Both-sides conflict list
MB=$(git merge-base origin/main HEAD)
comm -12 <(git diff --name-only $MB HEAD | sort) \
         <(git diff --name-only $MB origin/main | sort)
```

---

## 8. Out of scope / non-claims

- Did not run the full E1–E3D wall-clock gate matrix on this pod.  
- Did not modify `self-hosted/` on main.  
- Did not decide MLI operand model (WS-D).  
- Did not claim production MIR→x86.  
- Did not re-measure divergence after future main motion past `a5548481d1`.

---

## 9. Handoff

| Item | Location |
|---|---|
| Study worktree | `/workspace/.wt/mir-study` @ `97b525949` |
| Driver ELF (session) | `/tmp/mir-study-enir/madaros-enir` |
| Coord claim | `grok-cli1` / `ws-c-mir-study` → `docs/architecture/MIR_PORT_PLAN.md` |
| Next founder action | Accept/reject Route B default; then schedule port PR1 after WS-B SOIR |

*End of WS-C wave-1 study.*
