<!-- docs:meta
topic_id: repo.docs.audit.dce-silent-truncation-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dce-silent-truncation-census-2026-08-19
-->

# DCE silent truncation — census, not a patch

**Status:** measured. No DCE implementation is changed this turn.
**Claims-forbidden:** “Madaros is fixed-point-verified.” It is not.
**Do not touch:** `check/specializer.sio` spec_dce. It already refuses
loudly. That behaviour is correct.

This note answers a dispatch that began as “does DCE truncate in
silence?” and was amended to three DCEs plus a spec_dce-margin clock.
The clock is first, because it is a date, not a hypothesis.

**Verdict, one screen.** spec_dce occupancy is **7075 / 8192**
(from-source, #1947). Margin **1117**. #1935 ate 46. **24** merges
of that size remain; the 25th saturates and spec_dce refuses to
filter `main.sio` (loudly, which is correct). Ordinary merges since
#1935 ate **3** marks on the shipped ELF (7044 → 7047). `ir/dead_code.sio`
is **orphan** — the naked return at 128 is not live. `ir/dce.sio`
refuses at 8192 and is self-test only. Production IR DCE is
`opt_cleanup`. Positive control fired: chain 602→602 exit 99, prune
303→3 exit 7. No DCE code is changed.

Worktree: `/workspace/.wt/dce-trunc-20260819`
Branch: `lane/grok-cli1/dce-truncation-census-20260819`
Source SHA under test: `1dc0df549d` (`origin/main` at branch creation).

---

## Semantic declaration

```text
Semantic-Lane-ID: dce-truncation-census-20260819
Owner: grok-cli1
Concept-IDs: none
Intent-Preserved: a capacity ceiling that drops live code without a
  diagnostic is a lie; measuring whether such a ceiling is on the
  production path is not a licence to raise, lower, or rewire it.
Transformation: none. Observation of existing passes. No type, effect,
  IR field, or claim meaning is altered.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the production compile path of Madaros does not run
  ir/dead_code.sio; spec_dce mark occupancy vs SPEC_DCE_MAX=8192 is a
  dated remaining-budget, not a live silent truncate.
Claims-Forbidden: Madaros is fixed-point-verified; the 128-cap silent
  return is live in production; ir/dce.sio:821 silent refuse is a
  truncation (it is a refuse); spec_dce should be patched this turn.
Assumptions: “production path” means the ordinary multi-module compile
  of user code through module_frontend (the path main.sio itself takes).
  Self-tests inside main.sio are not that path. The bootstrap seed
  concat (lean_single) is a different binary.
Write-Set: docs/audit/DCE_SILENT_TRUNCATION_CENSUS_2026-08-19.md
Read-Set: self-hosted/check/specializer.sio, self-hosted/ir/dce.sio,
  self-hosted/ir/dead_code.sio, self-hosted/ir/opt_cleanup.sio,
  self-hosted/ir/optimize.sio, self-hosted/compiler/main.sio,
  self-hosted/compiler/module_frontend.sio,
  self-hosted/compiler/module_loader.sio,
  self-hosted/compiler/lean_single.sio, scripts/bootstrap/bootstrap_concat.sh,
  bootstrap/selfhost-kernel.manifest, PRs #1935 #1938 #1947,
  docs/audit/MADAROS_BOX_AUTODEREF_MAIN_REPRODUCTION_2026-08-17.md,
  docs/audit/WS_C1_MERGE_DCE_COST_2026-08-19.md (sibling worktree)
Positive-Witness: SOUNIO_MM_SPEC_TRACE=1 prints a dce-marks line on a
  program below SPEC_DCE_MAX (dce_reach chain/prune fixtures).
Negative-Witness: a live call-chain from the compile driver to
  dead_code.sio:155, or a printed output that changes vs lean_single
  because an instruction past a silent cap was dropped.
Acceptance-Gate: the refutation criteria below; if the positive control
  does not fire, every subsequent number is void.
Integration-Target: origin/main (docs only)
Authoritative-Only-If: the call-chain is from `use` + callee, not from
  a name grep; mark counts name the ELF and the source SHA that produced
  them; mixed instruments are not subtracted from each other.
```

---

## 0. Refutation criteria — written before the runs

The claim under test is:

> A live DCE path silently truncates (drops instructions, marks, or
> functions without refuse) on the production compile of real programs,
> at DCE2_MAX_INSTRS=128 per block or DCE_MAX_INSTRS=8192 per function.

Refute that claim if any of the following holds. These are the stop
conditions. They were written before the Slurm jobs and before the
source census.

**R1 — orphan path.** Path P is not reachable from the production
driver: no `use` of the module, and no callee in the compile/check/run
call-chain of user code. Truncation on P is then not live. Finding that
`ir/dead_code.sio` is orphan is a complete answer to question 1, not a
prelude.

**R2 — refuse, not truncate.** Path P is reachable, but saturating the
cap refuses the analysis (returns without filtering / without sweeping)
rather than analysing a prefix. Truncation is then not live on P, even
if the refuse itself is silent. This is the existing decision at
`ir/dce.sio:821` (empty `DceStats`, comment entered in `042c29be53`).

**R3 — cap not exercised.** Path P is reachable and would truncate, but
the largest basic block and the largest `func.instr_count` produced by
compiling `main.sio` are strictly below the cap. The defect is then
real and of lower severity; the report must still name the measured
maxima and say the cap is theoretical on today's corpus.

**R4 — positive control.** A program *below* the cap, whose live value
is defined early and used after a dead definition, MUST keep the live
value. For spec_dce that is `tests/multimodule/dce_reach/`:

- arm `chain`: 602 live functions survive (`marks=602`, exit 99)
- arm `prune`: 303 item_fns → 3 surviving (`marks=3`, exit 7)

The instrument of the mark clock is the trace line itself. If
`SOUNIO_MM_SPEC_TRACE=1` does not print `lower_array: dce marks=` (or
`spec_dce: item_fns … marks=`) on a program that takes the ordinary
multi-module path, the instrument is broken and every mark number in
this note is void.

**What would confirm, and is not being hunted:** a live call-chain to
`dead_code.sio:155` plus a real basic block of ≥128 instructions plus a
witness whose printed output changes against lean_single as oracle.

**Mark-clock refutation (the first number).** “The margin is closing
toward a date” is refuted if comparable from-source mark counts are
flat or falling across recent merges. One observation of +46 is a
rate of one class of merge, not a trend. Mixing the shipped-ELF series
(7044) with the from-source series (7029 / 7075) is forbidden: those
are different compilers.

---

## FIRST NUMBER — spec_dce margin vs SPEC_DCE_MAX=8192

`SPEC_DCE_MAX = 8192`. When marks hit that ceiling,
`spec_dce_mark_across_programs` prints

```text
spec_dce_mm: REFUSING to filter — mark set incomplete
```

(`specializer.sio:1793`) and returns -1. The filters then hand their
input back unchanged. Loud, correct, and it turns specializer DCE *off*
for the entire `main.sio` closure. That is a date.

### Comparable from-source series (one instrument)

| when | compiler | source | dce marks | declared fns | margin vs 8192 | origin |
|---|---|---|---:|---:|---:|---|
| 2026-08-17/18 | from-source raise build (`ac82ead7` / `madaros-raise2`) | then-main | **7029** | 10705 | 1163 | autoderef note; #1947 table |
| 2026-08-19 | from-source pre-#1935 (`madaros-raise2`) | pre-#1935 | **7029** | 10705 | 1163 | #1947 |
| 2026-08-19 | from-source with #1935 (`madaros-main-1935`) | `6f23dfe1da` | **7075** | 10940 | **1117** | #1947 |

#1935 ate **46** marks in one merge (7029 → 7075). Occupancy at 7075 is
86.4% of 8192.

### Not the same series (shipped ELF, pre-#1935 pass)

| when | compiler | source | dce marks | origin |
|---|---|---|---:|---|
| 2026-08-19 | shipped `bin/madaros-linux-x86_64` `3d1f143e7a` (predates #1935 and the 16384 raise) | `6f23dfe1da` | **7044** | #1938 / WS-C1 |

7044 is the *old* pass looking at *post-#1935* trees. It is not 7075
and must not be subtracted from it. #1938 said so: that turn did not
rebuild Madaros.

### Rate, and how many #1935-sized merges remain

From-source marks sat at **7029** across the autoderef measurement and
the pre-#1935 build (days, +0). Then one retention fix (#1935) added
**+46**. That is the only comparable step in the series.

After #1935, first-parent on `origin/main` through `1dc0df549d` is nine
merges. **One** of them touches `self-hosted/` (#1939,
`module_frontend.sio` + `lower.sio`). The rest are tests, docs,
governance, TypeKind archaeology.

Remeasured this turn, **same shipped ELF as #1938**
(`3d1f143e7a`, SHA-256 `437bdd8f…`), source now `1dc0df549d`,
host `cpuops-t560-proxmox`, 2026-08-19T09:29:31Z:

```text
imported_compile: loaded 120 modules
imported_compile: typecheck ok
lower_array: dce marks=7047
IR slot census: globals 1892 + functions 7224 = 9116 (max 8191, over by 925)
IR lowering failed during merge
MAIN_REFUSE=0
```

The **8191** in that census is not a source defect.
`self-hosted/ir/ir.sio:35` has `IR_MAX_FUNCS=16384`; the source
limit is `IR_MAX_FUNCS - 1 = 16383`. 8191 is `8192 - 1`, so it
came from the shipped ELF `bin/madaros-linux-x86_64`
(`3d1f143e7a`, 2026-08-17), which predates the raise. The real
defect is that the binary in git does not compile the tree in git.
That gate is glm-cli1's; it is not this census.

Shipped-ELF series: 7044 (`6f23dfe1da`) → **7047** (`1dc0df549d`).
**+3 marks across nine first-parent merges.** Ordinary-merge rate on
this instrument is ~0.3 marks/merge. That is not a date.

The date is the other instrument. From-source occupancy is 7075
(86.4% of 8192). The only comparable from-source step is #1935's +46.

```text
remaining margin at 7075 :  8192 − 7075 = 1117
#1935-sized steps left   :  1117 / 46  = 24.28
```

**24** merges of size 46 leave 13 marks. **The 25th saturates.** On
saturation spec_dce refuses loudly and stops filtering `main.sio`.

Give the 24 first. The +3/9 is the check that the margin is *not*
closing on ordinary merges; it is closing when someone lands another
retention-alignment of #1935's size.

This turn did **not** rebuild Madaros. 7047 is the pre-#1935 pass
looking at current trees (same caveat #1938 stated). When the shipped
ELF is refreshed to include #1935, occupancy will jump toward the
from-source 7075, not crawl from 7047. Budget against the number that
will actually hit the ceiling: **24**.

### What this number is not

- Not a licence to raise `SPEC_DCE_MAX` this turn.
- Not a claim that the 25th such merge is imminent. #1935 was a
  one-time alignment of mark policy with retention policy (scan every
  impl/trait body because the filter never deletes `ItemImpl`). The
  next +46 needs another hole of that shape, or a large growth of
  reachable names.
- Not mixed with 7044.

---

## Question 1 — which DCE is on the real pipeline

Resolved by `use` and by call-chain to the driver, not by grepping the
string `dce`. Four implementations exist. The dispatch named three.

### (1) `check/specializer.sio` — spec_dce — LIVE, loud refuse

Wired onto the ordinary multi-module path since `146f5b039f`
(2026-08-05). Call-chain:

```text
module_frontend_lower_programs_array_direct_box
  → spec_dce_mark_across_programs(programs, loaded)     :5495
  → spec_dce_filter_with_global_marks(...)              :5519
```

`main.sio` reaches that function through
`use module_frontend::{load_multimodule_ir, …, module_frontend_compile_imported_to_file}`.
On saturate: print `spec_dce_mm: REFUSING to filter — mark set incomplete`
and return -1. **Do not touch.**

### (2) `ir/dce.sio` — imported, self-test only, silent refuse

`main.sio:66`:

```text
use ir::dce::{dce_check_nop, dce_empty_instrs, dce_make_test_func, dce_optimize_function}
```

The only call of `dce_optimize_function` in `main.sio` is
`compiler_main_test_dce_removes_dead` at line 6417 (Sprint 51 self-test,
three instructions). Smoke drivers (`lowering_trace_smoke.sio`,
`wide_native_compile_driver.sio`, …) import the same four names for
the same self-tests.

The compile path does **not** call it. After lowering, with `-O`:

```text
main.sio:5485     opt_cleanup_module
main.sio:5487     lopt_optimize_module
main.sio:5493     opt_cleanup_module
module_frontend   opt_cleanup_module_inplace          :6502, :6588
module_loader     opt_cleanup_module                  :2978, :2982
```

The refuse at `dce_run_impl:821` (`if instr_count > DCE_MAX_INSTRS {
return dce_stats_new() }`) is the decision the founder named. The
comment at 817–820 (commit `042c29be53`, 2026-08-11) is exact: a
truncated liveness analysis deletes live code. PR #1947 invariant 2
is accusing that already-landed decision, not a live truncate on the
compile path. **R2** for this file.

`dce_populate` still stops at `DCE_MAX_INSTRS` (line 792). That loop is
behind the refuse, so it is not a second truncate.

### (3) `ir/dead_code.sio` — ORPHAN. Complete finding for question 1.

Evidence, in the order the dispatch asked for:

1. **No `use ir::dead_code` anywhere in the tree.** `rg 'use ir::dead_code'`
   is empty. `ir/mod.sio` does not export it.
2. **No callee outside the file.** `test_dce2_main` is defined at line
   1301 and called from nowhere. Every `dce_add_instr` / `dce_run` hit
   in this file is its own test. `lean_single.sio:5090` is a *comment*
   (“Live casualties found by shape census: ir/dead_code.sio:260”), not
   a call.
3. **No IrFunction bridge.** The module operates on a private `DceCfg`
   of integer opcodes (`DCE2_ADD=2`, …). Nothing in the compiler
   converts an `IrFunction` into that CFG. Even a future `use` would
   not see production IR without a new converter.
4. **Not in the lean_single concat.** `scripts/bootstrap/bootstrap_concat.sh`
   lists `ir/dce.sio`, `ir/optimize.sio`, `ir/opt_cleanup.sio`. It does
   not list `ir/dead_code.sio`. Concatenating both would collide:
   both files define `fn dce_run` with different signatures.
5. **Listed in `bootstrap/selfhost-kernel.manifest:97`.** That manifest
   is the *bootstrap-seed kernel* (`scripts/bootstrap/build_bootstrap_seed.sh`,
   selfhost-cycle gates). It is not the Madaros user compile pipeline.
   Presence on a seed-file list is not a call-chain.

**R1.** The naked `return` at line 155
(`if bref.count >= DCE2_MAX_INSTRS { return }`) is unreachable from
production. The 128-cap silent drop is not live. Question 3 (witness
vs lean_single) is therefore not opened. Hunting a witness after an
orphan finding would be confirmation-seeking.

### (4) Found, not in the original three: `ir/opt_cleanup.sio` — LIVE IR DCE

This is the production IR dead-code pass. `ocp_mfi_dce_once`
(`opt_cleanup.sio:9353`) and `ocp_dce_once` (`:7902`) run from
`opt_cleanup_module(_inplace)` on the compile path above.

It does **not** cap instructions at 128 or 8192. Liveness is a
`[bool; 256]` over virtual registers. Registers `>= 256` are neither
marked nor swept (the `dst < 256` guard). That is conservative
ignorance of high vregs, not a prefix truncate of the instruction
stream. #1682 already taught this pass to see `IrIndexSet`'s third
operand and call-args; that is a different defect class (missed use,
now patched).

`ir/optimize.sio` `dce_pass` (`dst < 8192`, no refuse) is in the
lean_single concat and is **not** imported by the modular driver.
`optimize_function` is only called from `optimize.sio` itself
(self-tests). A fourth-and-a-half, also not on the modular compile
path.

### Q1 verdict

| pass | ceiling | on saturate | on production compile of user code |
|---|---|---|---|
| spec_dce | 8192 marks | loud refuse, no filter | **yes** (ordinary multi-module) |
| ir/dce.sio | 8192 instrs | silent refuse (empty stats) | **no** (self-test only) |
| ir/dead_code.sio | 128 instrs/block | naked return (drop from CFG) | **no** (orphan) |
| opt_cleanup DCE | 256 vregs | ignore vreg ≥ 256 (keep) | **yes** (IR `-O`) |

Silent truncation at 128 is refuted by **R1**. Silent truncation at
8192 on `ir/dce.sio` is refuted by **R2** plus “not on the compile
path.” The live ceiling that can actually turn DCE off for `main.sio`
is spec_dce at 8192 marks, and it talks.

---

## Question 2 — are 128 and 8192 reachable on real IR?

For `dead_code.sio`, the 128-cap is not reachable because the pass is
not on the path (**R1**). Severity of a theoretical cap in an orphan
file: low. Still named.

For `ir/dce.sio`, the 8192-cap is reachable in principle:
`IR_MAX_INSTRS = 16384` is twice that, and a historical function
(`run_compiler_main_self_tests`) needed 33829 instructions before it
was split (`ddcded1284`). That pass is not on the compile path. If it
were, `dce_run_impl` would refuse the function rather than truncate.

Source-level proxy (not IR, not basic-block) and any Slurm IR census
are in §Measurements. A source function of more than 128 body lines is
enough to say “if someone wired dead_code 1:1 onto IrFunction, 128
would not be theoretical.” It is not evidence that the orphan path
runs.

---

## Question 3 — witness vs lean_single

Not opened. The condition was: live silent-cut path *and* reachable
cap. Q1 closed the 128-path as orphan. A witness whose output changes
because of line 155 cannot exist on a pass that is never called.

The positive control that *must* fire (R4) is the dce_reach pair, not
a 128-cap program.

---

## Measurements

### 0. Founder context, verified, not trusted

| claim | verified at `1dc0df549d` |
|---|---|
| `dce.sio:821` refuses `instr_count > 8192` with empty stats | **yes.** Comment 817–820 matches `042c29be53`. |
| that comment is why truncation is not live on this path | **yes**, and the path is self-test only (stricter). |
| `dead_code.sio:155` naked return at 128 | **yes.** |
| spec_dce refuses loudly at `specializer.sio:1793` | **yes.** |
| spec_dce live since `146f5b039f` | **yes**, call at `module_frontend.sio:5495`. |
| #1938 marks=7044 | **yes**, WS-C1 audit, shipped ELF vs `6f23dfe1da`. |
| #1947 marks=7075 | **yes**, PR body, from-source `madaros-main-1935`. |
| #1935 ate 46 | **yes**, 7029 → 7075 on matched from-source builds. |
| margin 1117, 86% of 8192 | **yes**, 8192−7075=1117; 7075/8192=86.4%. |

The research branch `research/zd-fiber-antisymmetry-lemma-20260731`
(`d9b99b95b2`) still has the *pre-refuse* `dce.sio` (populate-only).
That is why this census is on a worktree of `origin/main`, not on the
control checkout.

### 1. Positive control (R4) — Slurm, FIRED

Host `cpuops-t560-proxmox`. Raw ELF
`bin/madaros-linux-x86_64` (SHA-256 `437bdd8f…`).
`SOUNIO_MM_SPEC_TRACE=1`. Stage
`/orangefs/training/dce-trunc-20260819T0927Z`.

| arm | trace | build rc | run rc | required |
|---|---|---:|---:|---|
| `dce_chain_main.sio` | `spec_dce: item_fns 602 -> 602 marks=602` | 0 | **99** | keep every live fn |
| `dce_prune_main.sio` | `spec_dce: item_fns 303 -> 3 marks=3` | 0 | **7** | drop 300 of 303 dead |

The trace is present. The pass ran. Live code survives. Dead code is
still deleted. **The instrument is not broken.** Every mark number
below is live.

(First attempt used `test -x` on the ELF before `chmod +x` and
reported `CHAIN_NO_ELF`. Rebuild + chmod reproduced the gate's
exit codes. The trace was already correct on the first attempt.)

### 2. Mark remasurement at `1dc0df549d` — Slurm

Instrument A (comparable to 7044): **7047**. No refuse.
`120` modules. Slot census `1892 + 7224 = 9116` against the
**shipped ELF's** 8191 (`8192 - 1`), not against source
`IR_MAX_FUNCS - 1 = 16383`. Same wall #1938 named; three more
function slots and three more marks. The source ceiling already
fits; the blob in `bin/` does not.

Instrument B (comparable to 7075): not rerun. From-source rebuild
is a different dispatch. 7075 remains the last from-source point.
The remaining-budget of 24 is against that point, labelled as such.

### 3. Source-level function-size proxy (local, not IR)

Not basic-block size and not `instr_count`. A Python walk of `^fn `
bodies. One source line is typically several IR instructions; a body
of ≥128 lines is therefore enough to say “if dead_code were wired 1:1
onto IrFunction, 128 would not be theoretical.” It is not evidence
that the orphan path runs.

`self-hosted/compiler/main.sio` at `1dc0df549d`: **1451** functions,
largest body **831** lines (`compiler_main_self_tests_part_01` at
L22048 — the post-`ddcded1284` split). Ten of the eleven largest are
self-test slices. `compile` is 276 lines. Every one of these is
**below 8192** source lines and **above 128**.

Whole `self-hosted/`: **28718** `fn` bodies.

| threshold | count |
|---|---:|
| body_lines ≥ 128 | **374** |
| body_lines ≥ 8192 | **1** |

The one function past 8192 source lines is
`lower_program_has_captured_closure_value_misuse_ref`
(`self-hosted/ir/lower.sio:5916`, **11571** lines). That is a source
body, not a basic block, and not a reason to call `dead_code.sio`
live. It *is* a reason to keep `ir/dce.sio`'s refuse-at-8192: if
that function lowered to more than 8192 instructions and
`dce_optimize_function` were on the compile path, the refuse would
be the thing that stopped a truncated liveness sweep. It is not on
the compile path.

Shipped ELF identity, same blob #1938 measured:

```text
bin/madaros-linux-x86_64
SHA-256 437bdd8f96a205906d53ca50a2a29ccf5f03a71c2e98e020b54d01351a0bff44
commit  3d1f143e7a  2026-08-17T07:09:56Z
```

---

## What this turn is not

- Not a patch to spec_dce, dce.sio, dead_code.sio, or opt_cleanup.
- Not a raise of `SPEC_DCE_MAX`.
- Not a wiring of `dead_code.sio` onto IrFunction.
- Not a claim that rung gen2 is green.
- Not a claim that Madaros compiles itself.
- Not a claim that opt_cleanup's 256-vreg table has been shown safe
  by a vreg census (it has not; it is a different question).

## What a later turn may do

1. From-source remasure of marks at HEAD, same instrument as 7075.
2. If marks approach ~7800, treat saturation as a near-term date and
   decide *before* the refuse: raise `SPEC_DCE_MAX` in lockstep with
   the mark arrays, or accept DCE-off for self-compile.
3. A vreg census of `opt_cleanup`'s live 256-slot table, if anyone
   wants to know whether high vregs exist in production IR.
4. Delete or quarantine `ir/dead_code.sio` only after the
   selfhost-kernel.manifest consumers are named. Orphan ≠ unused on
   every seed path.

## Registry

Synced after the file existed, by
`node scripts/docs/sync_governance_metadata.mjs`. The registry is
derived from the filesystem; that is why the first push was red.

## AI disclosure

Measurement and prose by grok-cli1. No DCE code generated. Founder
numbers from #1938 / #1947 / the autoderef note were re-read and then
re-measured on Slurm where the same instrument still exists (R4
fixtures; shipped-ELF marks=7047). 7075 was not from-source-rebuilt
this turn.
