<!-- docs:meta
topic_id: repo.docs.internal.coordination.madaros-focus-plan-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.madaros-focus-plan-2026-08-16
-->

# Madaros Focus Plan — Fleet Redirection

Status: **active operational plan**, wave 1 dispatched.
Authored: 2026-08-16T00:25:20Z. Drafted by Fable 5 (`claude-fable-5`) at founder
request, from ground truth gathered by two research passes over `/workspace/sounio`
the same session. Orchestrated by `claude-1` (tmux session `fleet`, coordination
bus lane `fleet-orchestrator`).
Supersedes: `.claude/OPERATIONAL_CANONICAL_INDEX.md`'s Track B skeleton (orphaned —
its referenced `.claude/prompts/*.md` files no longer exist; retire or rewrite that
doc once this plan is established).

Founder objectives (verbatim, 2026-08-16): "Madaros E2E operacional, SOIR, MIR,
MLI, HLIR EISA, f128 e f256 implantados e verificados." Confirmed definitions:
**MLI** = Machine-Level IR, a genuinely new layer (does not exist yet; sits
between MIR and codegen). **EISA** = the `tools/eisa/` + `stdlib/eisa/`
instruction-set/bridge conformance suite (not the already-shipped EISA
thinlink/CodeBuffer linker fix, PR #724).

Governance this plan operates under (unchanged, binding):
`.claude/ATTENTION_CHARTER.md` (5 = 1 + 2, ≤1 active P0),
`CLAUDE.md` §4 (**≤2 agents editing `self-hosted/` at once** on this pod),
`docs/internal/coordination/COMPILER_LANE_CONTRACT.md`,
`bin/sounio-coord` / MCP `sounio-coord` for claims and messaging.

---

## 1. Workstream decomposition

**WS-A · Madaros E2E operational** — verify + close residuals.
Done = a fresh, dated `make madaros-full-gate` at 10/10 under the *current*
binary (not the Feb-2026 numbers in `docs/MADAROS_STATUS.md` /
`artifacts/omega/*.v1.json`, both ~6 months stale), extern "C" FFI silent-noop
closed (P0-F), witness-matrix residuals dispositioned (enum-discriminant
collision; `as f32` truncation — f32 still carried as f64 internally, no real
narrowing, declared residual), and `docs/MADAROS_STATUS.md` +
`artifacts/omega/*.v1.json` regenerated from that run.

**WS-B · SOIR** — verify existing, add one gate. Done = a numeric
`scripts/ci/soir_roundtrip_gate.sh` (serialize→deserialize→structural compare
over a representative corpus of `self-hosted/ir/*`) green under default
Madaros, plus a coverage census of what already exercised SOIR before the gate
existed. Must land **before** the MIR port lands, so serialization drift
during the port is caught mechanically.

**WS-C · MIR port** — multi-week build/integrate. Real ENIR/MIR pipeline
(`self-hosted/enir/{ir,mir,mir_cfg,mir_join,canonical,hash,interpreter,parser,
qd,shadow_fixture,source_lower,verify,driver,mod}.sio`, 4 architecture docs
`docs/architecture/MIR_*`) exists only on frontier branch
`canon/madaros-v2-sota`, stalled at `97b525949` (2026-07-12). **Correction
(grok-cli1, `MIR_PORT_PLAN.md`, 2026-08-16): the divergence direction in the
original draft of this plan was backwards.** Measured:
`origin/main..frontier` = **189** commits (frontier-only), `frontier..origin/main`
= **2086** commits (main-only) — main is the long tip, the frontier is a short
189-commit divergent lobe carrying the ENIR lane. 115 files touched on both
sides since merge-base (41 under `self-hosted/`); the ENIR tree itself
(14 files, 7310 LOC) has **zero** main-side counterpart and imports only
`enir::*`, so it is not on the conflict surface.

**Route decision: Route B — `enir/` subtree cherry-pick + resume the lettered
plan — APPROVED by founder 2026-08-16**, per the costed recommendation in
`docs/architecture/MIR_PORT_PLAN.md` (Route A wholesale rebase rejected —
3–6+ engineer-weeks of conflict surgery on the worst IR/compiler/native
hotspots; Route C fresh re-derivation kept only as a fallback if Route B's
enir sources fail to typecheck under main within a bounded repair budget).
Suggested PR stack: PR1 add `self-hosted/enir/**` + `bin/madaros-enir` + docs
only (driver check/compile under seed) → PR2 E1 gate green on main → PR3
E2A–E2H → PR4 E3A–E3D(+E3E) → PR5 optional umbrella gate. No production
`use enir::` from `compiler/main.sio` until a later explicit integration
tranche, post-WS-D MLI design.

**WS-D · MLI design** — greenfield, multi-week.
Phase 1 done = an approved `docs/architecture/MLI_DESIGN.md`: layer contract
(input = MIR, WS-C route now B), instruction/operand model, register/stack
discipline, verification story (interpreter or shadow-execution parity,
mirroring `enir/interpreter.sio` / `verify.sio`), staged implementation
ladder. Phase 2 (wave 3+) = a first vertical slice, one function lowered
MIR→MLI→x86 bit-identical to the existing direct path.

**Design option: Option C — dual-track (kinds first-class in the type
system; R0 scalar-only emit first, R1 epistemic/CD emit staged) — APPROVED
by founder 2026-08-16**, per `docs/architecture/MLI_DESIGN.md` §2.4 (Option A
scalar-only rejected as sole architecture — would forfeit the plan's one
"beyond any existing language" bet; Option B full-first-class-from-day-one
rejected — blocks Phase-2 bit-identity for months). Implementation ladder
S0 (this approval) → S1 `mli` module/kinds/builder/V-struct verify → S2
`mir_to_mli` scalar R0 → S3 `legalize_x86`, Phase-2 bit-identical vertical
slice → S4 f32/f128/f256 slots (WS-G coordination) → S5–S7 Knowledge/CD
kinds staged. **S1 implementation dispatch (touches `self-hosted/`) HELD
pending P0-F close**, same 1-active-P0 discipline the founder set for WS-C
— not re-asked separately, applying the precedent.

> **Design principle, binding for the design doc** (founder + orchestrator,
> 2026-08-16): Sounio already has two features no mainstream compiler backend
> treats as native machine-level types — `Knowledge<T>` / GUM-propagated
> uncertainty, and Cayley-Dickson hypercomplex algebras (already lowered to
> GPU tensor cores in `self-hosted/gpu/`, "Validated research" status, PR
> #1207). Today both are erased or reduced to library/struct calls by the
> time code reaches codegen. **MLI is the one layer where this doesn't have
> to happen.** The design doc must evaluate — not silently default away from
> — carrying `Knowledge<T>` provenance and CD-algebra dimension as first-class
> MLI operand *kinds* (alongside `f32`/`f64`/`f128`/`f256`), so register
> allocation and instruction selection can be uncertainty- and
> algebra-aware rather than erasing that information before codegen. This is
> the concrete "beyond any existing language" opportunity in this plan; it is
> a research-grade design decision, not a default, and the doc must state the
> tradeoff explicitly (scope/schedule cost vs. novelty) rather than assume it.

**WS-E · HLIR re-verify** — verify existing, days. Real source
(`self-hosted/hlir/`, `hlir_to_gpu.sio`, `hlir2wasm_driver.sio`,
`test_epistemic_hlir_gpu.sio`); its "pass" status is from the stale Feb-2026
snapshot. Done = fresh pass/fail evidence under current Madaros, replacing
that snapshot; failures filed as scoped dispatches, not patched ad hoc.

**WS-F · EISA Madaros port** — bounded build, 1-2 weeks.
`scripts/ci/eisa_bridge_conformance_gate.sh` hardcodes
`SOUNIO_SOUC_ENGINE=lean_single`, never touches Madaros — this is the real
gap. `scripts/ci/eisa_h_zd_reference_gate.sh` is already Madaros-aware
(tolerates a documented `BLOCKED` rc=12) — use it as the structural template.
Done = the bridge-conformance gate running its full matrix under default
Madaros, golden `artifacts/eisa/*.eisax.elf` reproduced byte-identical or
divergence-receipted. Check for `extern "C"` dependence first — may be gated
on P0-F.

**WS-G · f128/f256** — greenfield from V0-A, multi-week, the largest single
item. `docs/EXACT_CORE.md:55-57`: "no f256 arithmetic is implemented" by
design; `self-hosted/parser/types.sio:41` rejects f128/f256 literals on
purpose, staged "V0-A". Existing: `ty_f128()`/`ty_f256()` table entries,
probe/scaffold files `self-hosted/compiler/f128_f256_{numeric_wire,
format_descriptor,numeric_payload}_probe.sio` (wire encoding only). Staged
done: **V0-B** literals accepted end-to-end through check; **V0-C** wire
format/limb pools live (extend the probe files into real modules); **V0-D**
softfloat arithmetic (add/sub/mul/div/cmp, compiler-owned limb routines in
Sounio); **V0-E** stdlib surface + printing + GUM interaction. Each stage
gated, with external-oracle test vectors (MPFR-derived, generation receipts
checked in; verification runs in Sounio).

---

## 2. Sequencing

```
P0-F (extern C) ──► WS-A fresh gate ──► trustworthy baseline for everything
WS-B (SOIR gate) ──► must land BEFORE WS-C port lands
WS-C route decision ──► WS-C port ──► WS-D MLI implementation (input contract)
WS-D design doc ── parallel now (needs only frontier docs + native/ reading)
WS-E, WS-F ── independent; WS-F possibly gated on P0-F
WS-G V0-B/C ── independent, wave 2
WS-G V0-D ── recommend: land on current pipeline now, accept small re-lower
              cost after MLI, rather than waiting
```

---

## 3. Wave plan (≤2 self-hosted/ writers at a time)

Ceiling counts only lanes editing `self-hosted/`. Gate runs, script/spec/test
authoring, `tools/`/`stdlib/`/`docs/` work, and read-only frontier study do
not count — but full builds still serialize through
`scripts/dev/souc-build-lock.sh`; cap concurrent heavy gate runs at 2.

### Wave 1 (now, ~5-7 days) — dispatched 2026-08-16

Hot writers (unchanged, already in flight): **glm-cli1** (P0-F to completion),
**claude-3** (land PR #1737, then the f32-narrowing residual).

Cold lanes (dispatched this wave):

| Lane | Task |
|---|---|
| codex-1 | WS-A: fresh `make madaros-full-gate` under the build lock; regenerate `artifacts/omega/*.v1.json`; diff vs the stale Feb-2026 claims |
| codex-2 | WS-B: SOIR coverage census + author `scripts/ci/soir_roundtrip_gate.sh` |
| grok-cli1 | WS-C: frontier study in an isolated worktree off `origin/canon/madaros-v2-sota`; author `docs/architecture/MIR_PORT_PLAN.md` (3 routes costed, conflict census) |
| grok-cli2 | WS-D: draft `docs/architecture/MLI_DESIGN.md`, honoring the binding design principle in §1 |
| grok-cli3 | WS-G: spec doc lifting the V0-A boundary into the V0-B..E ladder |
| grok-cli4 | WS-F: Madaros-engine EISA gate variant (template off `eisa_h_zd_reference_gate.sh`); run + classify failures; check `extern "C"` dependence |
| grok-cli5 | WS-E: inventory + run existing HLIR-exercising tests/gates; dated status note |
| glm-cli2 | WS-A support: author `tests/run-pass` / `tests/compile-fail` witnesses for the f32-narrowing and enum-discriminant fixes, queued for claude-3 |
| minimax-cli1 | Fleet hygiene: inspect `cursor-1`/`cursor-3` current tasks and report; draft retirement note for `.claude/OPERATIONAL_CANONICAL_INDEX.md` |
| minimax-cli2 | WS-A: write the explicit "E2E operational" acceptance checklist (which gates constitute E2E), cross-check each `MADAROS_STATUS.md` claim |
| minimax-cli3 | WS-G: generate + receipt the MPFR binary128/256 test-vector corpus |
| cursor-2 | Land this doc in the docs registry (`bash scripts/dev/docs_registry_sync.sh` or equivalent — sync **after**, not before, per known gate behavior) |

Held, not redirected (founder decision, 2026-08-16): **claude-2**,
**kimi-cli1**, **kimi-cli2** finish PR #1580 (ZD-fiber theorems) first — Garden
work close to done, do not interrupt. **codex-3** finishes its current SLURM
run first, then reallocate. Both re-evaluated at start of wave 2.

### Wave 2 (10-14 days, gated on P0-F closing + WS-C route decision)

Next P0 nomination (founder decision, 2026-08-16): **WS-C, the MIR port
route**, once WS-A's baseline is trustworthy and P0-F closes — highest
runaway-scope risk in the whole plan, decide the route early.
Hot writers: (1) WS-C port lane per the chosen route; (2) WS-G V0-B/V0-C lane
(parser + wire modules — candidate: glm-cli1 rolling off P0-F). Cold: WS-F
source fixes if needed; WS-B/WS-E gates land in CI; WS-D doc review + MLI
vertical-slice test fixtures; freed lanes rotate to reviewing the port plan
and running new gates nightly. claude-2/kimi-cli1/kimi-cli2/codex-3
reallocated here if their current work has closed.

### Wave 3 (3-4 weeks+, continuing)

Hot writers: (1) WS-D MLI vertical slice (only once MIR is stable on main);
(2) WS-G V0-D softfloat routines. Cold: conformance fixtures, divergence
receipts, EISA golden reproduction, omega/status regeneration cadence. WS-C
and WS-G both extend past wave 3 — treat these as checkpoints, not
completion dates.

---

## 4. Founder decisions taken (2026-08-16, interactive)

- (a) claude-2 / kimi-cli1 / kimi-cli2 (ZD-fiber, PR #1580): **let finish**,
  do not interrupt.
- (b) codex-3 (SLURM cs6_v7b research): **let current job finish**, reallocate
  after.
- (c) MIR port route: **still open** — decide from grok-cli1's wave-1
  conflict census (`MIR_PORT_PLAN.md`), not before.
- (d) Next P0 after F: **WS-C, MIR port route decision.**
- (e) WS-G V0-D timing: land on current pipeline now (Fable 5 recommendation,
  not separately re-litigated).

---

## 5. Risk register

1. **MIR port becomes a multi-month rabbit hole.** 189 main-side commits can
   silently invalidate frontier assumptions. Mitigation: no port work lands
   before the wave-1 conflict census; hard checkpoint at day 10 of wave 2 — if
   the chosen route hasn't produced a compiling `enir/` on main by then,
   escalate to route re-decision, not more effort.
2. **f128/f256 destabilizes the parser/checker.** V0-A rejection is a
   deliberate boundary; lifting it touches literal parsing and type inference
   shared by every test. Mitigation: smallest possible V0-B diff, full
   run-pass + witness-matrix regression gate before merge, detectable refusal
   on new-type paths (never silent zeros — cf. the token-ceiling lesson).
3. **Stale-status false confidence.** `MADAROS_STATUS.md`/omega numbers are 6
   months old. Mitigation: wave 1's first deliverable is the fresh full-gate
   run; forbid citing pre-refresh numbers anywhere downstream.
4. **Silent ceiling violation.** An eager cold lane edits `self-hosted/`
   without claiming. Mitigation: coord claims mandatory before edits;
   minimax-cli1's hygiene lane audits `git status` across worktrees daily; a
   third self-hosted/ editor releases immediately.
5. **CPU-saturation pod eviction.** Has killed this pod twice already. All
   full compiles through `souc-build-lock.sh`; cap concurrent heavy gate runs
   at 2; cold lanes schedule gate runs via coord messages, not ad hoc.
6. **Instrument trust.** Prebuilt `bin/souc` staleness has cost a full false
   investigation before (#1689). Every new gate ships with a positive control
   that must fire; codegen claims are made against a binary built this
   session, never the checked-in ELF.

Retire or rewrite `.claude/OPERATIONAL_CANONICAL_INDEX.md` once this plan is
established (minimax-cli1's wave-1 task).
