<!-- docs:meta
topic_id: repo.docs.audit.e035-mod-blast-radius-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e035-mod-blast-radius-2026-08-19
-->

# E035 blast radius if `with Mod` became `Mut`

**Filed:** 2026-08-19 · **Lane:** grok-cli5 · **Verdict:** LARGE BUT CLOSED

**SHA:** `bd361fd092` (`origin/main`). Engine named on every number.

This is a measurement. `self-hosted/` is not edited. `Mod` is not added
to `effect_name_to_id`. The substitution was applied only in a disposable
tree on a Slurm node and was never committed.

## Verdict

**LARGE BUT CLOSED.** The caller closure terminates at depth 2. It does
not enter the compiler. It does not add any stdlib file beyond the 59
that already write `with Mod`. The new annotations live in tests.

If the founder later decides to rewrite `Mod` as `Mut`, the edit is:

- 2,813 function signatures (`Mod` → `Mut`)
- **843 further functions / 552 files** that must gain a `Mut`
  declaration (759 of them at wave 1, 84 at wave 2)

That is a large mechanical pass, not an open-ended refactor.

## 1. How many `with Mod` sites?

Three instruments, same SHA, **not averaged**:

| Instrument | Sites | Files | What it counts |
|---|---:|---:|---|
| `git grep -nE '\bwith[[:space:]]+Mod\b'` (#2009) | **2806** | 363 | Adjacent `with Mod`, including comments |
| Parser-faithful `with`-clause tokens (#2004 gate) | **2813** | 365 | Every `Mod` in a `with` list; no comments/strings |
| Function-owned signatures (this lane) | **2813** | 365 | The function's own clause, not a parameter fn-type |

**Why they differ, site by site:**

- Grep has **1** hit the extractor refuses:
  `self-hosted/check/effects.sio:24` — the hold comment
  (`` `with Mod` still return -1 ``).
- The extractor has **8** hits grep misses: `Mod` is in the clause but
  not adjacent to `with` (`-> i64 with Div, Mod` in
  `stdlib/systems/ball_fixed.sio:873` and seven siblings).
- Arithmetic: 2806 − 1 + 8 = **2813**.

**The 2,793 from #2004 / #2009 is not a third count.** The closed-list
gate printed twenty `Mod` sites and then `omitted=2793` (2813 − 20
stdout cap). #2009 read that line as a census. It is a log cap.

Coexistence (parser-faithful files): 3 files write both `Mod` and
`Mut`; **362 write `Mod` and never `Mut`**. Directory buckets of the
365 files: `tests/run-pass` 304, `stdlib/systems` 41,
`stdlib/theorem` 13, `stdlib/safety` 5, `tests/effects` 2.

## 2. Transitive caller closure (static)

A file-local then unique-name call graph over versioned `.sio`
(archive / bootstrap / `*.sio.old` excluded).

| | |
|---|---:|
| Affected functions (`with Mod` on the fn itself) | 2813 |
| Affected files | 365 |
| Closure functions | 3656 |
| Closure files | 695 |
| Maximum chain depth | **2** |
| Wave-1 callers that would lack `Mut` after the rewrite | 759 |
| Wave-1 callers already covered (they also have `Mod`) | 1239 |
| Wave-1 callers that already declare `Mut` only | **0** |
| Closure members that need a new `Mut` | **843** |
| Closure members that already declare `Mut` only | 0 |
| Closure members that get `Mut` from the rewrite | 2813 |
| Reaches `self-hosted/` | **no** |
| Extra stdlib files in the closure | **0** (59 = the 59 that already write Mod) |
| Test files in the closure | 636 |

The 843 sit in **552 files**, all under `tests/` (551 `tests/run-pass`,
1 `tests/effects`). They are `main` / helper functions that call a
Mod-function and declare neither `Mod` nor `Mut`. Typical shape:
`tests/run-pass/*_imported.sio` `fn main() -> i64 { … }`.

## 3. Empirical (Slurm, both engines)

Disposable tree `/tmp/e035-in` on **cpuops-t560-proxmox** (partition
`cpu-ops`, 32 cores). Tarball stdin. 2,813 `Mod` tokens rewritten to
`Mut` in that tree only. The git worktree refused `--apply-scratch`
because `.git` exists.

Corpus: the 695-file closure. `souc check` (Madaros v0.80.0) and
`SOUNIO_SOUC_ENGINE=lean_single` compile. Baseline then after.

| Engine | Condition | Files | E035 | Files with E035 | check/compile rc=0 |
|---|---|---:|---:|---:|---:|
| **Madaros** | baseline | 695 | **0** | 0 | 695 |
| **Madaros** | after Mod→Mut | 695 | **13653** | **552** | 143 |
| lean_single | baseline | 695 | 0 | 0 | 13 |
| lean_single | after Mod→Mut | 695 | 0 | 0 | 221 |

**Madaros delta: +13,653 E035 on 552 files.** Zero E035 in the
baseline, so the difference is the whole after-count. File-level match
with the static “need new Mut” set: **552 = 552**.

The 143 files that still `check: OK` after the rewrite are the
self-contained Mod files: every function in them already had `Mod`,
so after the rewrite they all declare `Mut` and do not call anything
that now requires more.

13,653 diagnostics vs 759/843 functions: E035 is per call site, not
per function. A handful of `*_tiny.sio` files emit >100 lines
(e.g. `lorenz_i256_cover_child0_local_flowpipe_preflight_tiny.sio`
+111). Same files, denser reports. Not a disagreement about *where*.

**lean_single emits no E035** on either side. The seed does not carry
this diagnostic. Its rc=0 counts (13 → 221) are compile-to-ELF of a
mixed library/test corpus and are not an effect signal. Do not read
them as a blast radius.

### Positive control

`tests/effects/archaeology/mod_refuse.sio`

```
fn marked() -> i64 with Mod { 7 }
fn drops_effect() -> i64 { marked() }
```

| | Madaros |
|---|---|
| before | `check: OK` · rc=0 · E035=0 |
| after  | rc=1 · **E035=1** |

```
error[E035] in archaeology/mod_refuse::drops_effect at 0..147:
effect not declared in function signature (missing: Mut)
-- required by `marked`
```

That line does not exist in the baseline log. It is the change, not
corpus noise.

A second imported witness, same pattern:

```
error[E035] in run-pass/kernel_replay_evidence_router_imported::main
(missing: Mut) -- required by `kernel_replay_evidence_router_check`
error[E035] in run-pass/kernel_replay_evidence_router_imported::main
(missing: Mut) -- required by `kernel_replay_evidence_router_audit_fingerprint`
```

Baseline of that file: `check: OK`.

### Negative control

Same 695-file Madaros check **without** the substitution: **E035 = 0**.
The after-count is the delta.

Self-contained library `stdlib/safety/kernel_replay_evidence_router.sio`:
`check: OK` before and after, E035=0. Intra-file callers already had
`Mod`; the rewrite does not invent a hole there.

## What this does not decide

- Whether every `Mod` *means* `Mut`. #2009 read two bodies; this lane
  did not re-read 2,811 more.
- Whether to do the rewrite. That is the founder's. This only says
  the E035 wave is large, closed, and confined to tests.

## Reproduce

```bash
# static (no compiler)
python3 scripts/dev/e035_mod_blast_radius.py --root . --json-out /tmp/e035_mod_static.json

# substitution is refused inside a git checkout
python3 scripts/dev/e035_mod_blast_radius.py --apply-scratch .   # exits 1

# empirical: disposable copy + Slurm, never the worktree
# see /tmp/e035_mod_slurm.sh from the measurement turn
```

Instrument: `scripts/dev/e035_mod_blast_radius.py`.
Table: `docs/audit/E035_MOD_BLAST_RADIUS_2026-08-19.tsv`.
