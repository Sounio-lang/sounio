<!-- docs:meta
topic_id: repo.docs.audit.canonical-compiler-gate-structural-cost-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.canonical-compiler-gate-structural-cost-2026-08-18
-->

# Structural cost of `canonical_compiler_gate` — measurement, not a rebuild

**Date:** 2026-08-18  
**Lane:** grok-cli4 / epistemic-fabrication drain aftermath  
**Trigger:** PR #1750 — Contracts red on md5 divergence after `lean_single.sio` edit  
**Constraint:** no seed touch, no live rebuild; numbers from open PRs, CI logs, and historical commits only  
**Status:** measured

### Folklore cost (the number that matters)

Re-measured 2026-08-18, open=51. Touch `self-hosted/compiler/lean_single.sio`: **5**.  
Live Canonical = Contracts step present on latest completed CI (gate since 2026-08-04).

| class | n | who |
|---|---:|---|
| **A. Blocked ONLY by canonical md5** | **1** | **#1750** — `fail_steps` = only Canonical; no other CI red; +172/−32 codegen |
| **B. Blocked by md5 + something else** | **0** | — |
| **C. Touches lean_single + live Canonical PASS** | **1** | **#1729** — Canonical success 2026-08-13; +4/−4 **comment-only**; does not ship seed |
| **X. Touches lean_single; no live Canonical step** | **3** | #1034 pre-gate+ships seed; #1527 pre-gate; #1758 Docs registry (never reaches gate) |

**`folklore_cost_open_prs_blocked_only_by_canonical_md5 = 1`**

Category C is not a hole and not a rebuild author: comments preserve the fixed
point. The in-era rebuild author among open PRs is #1034 (ships ELF), pre-gate.

Recipe closed: `docs/ops/LEAN_SINGLE_SEED_REFRESH.md` (M1 hard stop) +
`bash scripts/dev/refresh_lean_seed.sh --print`. Rebuild still founder-gated.

---

## The rule (what the gate actually checks)

`scripts/ci/canonical_compiler_gate.sh` is wired hard into Contracts
(`.github/workflows/ci.yml`, step "Canonical lean_single fixed point", no
`continue-on-error`). It:

1. Takes the committed lean_single ELF — default
   `bin/souc-lean-single-x86_64` (override `SOUNIO_CANONICAL_SOUC`).
2. Compiles `self-hosted/compiler/lean_single.sio` once with that ELF.
3. **FAILS** if `md5(committed ELF) ≠ md5(self-compile output)`.

The gate is correct. A shipped seed that is not the fixed point of its own
source is a lying instrument (see #1606 / gen1-vs-fixed-point, #1678 handoff).

What is not correct is what the rule implies for everyday PR flow.

---

## 1. Open-PR census (2026-08-18)

| quantity | n | how measured |
|---|---:|---|
| Open PRs total | **50** | `gh pr list --state open` |
| Touch `lean_single.sio` and/or seed/wrapper bins | **6** | PR file lists |
| Touch `lean_single.sio` specifically | **5** | subset |
| Of those 5, ship a seed ELF that matches their own source | **0** at current heads* | blob md5 vs self-compile expectation |
| Contracts blocked **only** by canonical md5 (sole hard-fail step) | **1** (#1750) | step-level job API |
| Merge-ready if only md5 were fixed | **0** | #1750 is also `CONFLICTING`/`DIRTY` |

\* #1034 ships *an* ELF with the source, and its last Contracts run is green —
but the branch is an era behind `main` (996-line lean_single diffstat vs main)
and `mergeable=CONFLICTING`. It is not a live "resync satisfied current main"
example; it is a frozen green snapshot.

### The six lean/seed-adjacent open PRs

| PR | opened | touches | seed at HEAD | Contracts / CI | md5-only block? |
|---|---|---|---|---|---|
| **#1750** | 2026-08-16 | `lean_single.sio` only | main's `c28659d8…` (source diverged → self-compile `c4fe3c51…`) | **sole Contracts hard-fail = Canonical**; all other jobs green | **YES for the gate**; also merge `DIRTY` |
| #1758 | 2026-08-17 | `lean_single.sio` + `bin/souc-linux-x86_64` | **no** `souc-lean-single`; wrong binary refreshed | Docs registry + Lint + Suite + macOS self-host | **NO** — multi-fail; wrong ELF path |
| #1729 | 2026-08-13 | `lean_single.sio` | stale `3a7a17a0…` | Madaros f64 Lowering fails; no live Contracts canonical signal | **NO** |
| #1527 | 2026-07-27 | `lean_single.sio` | stale `68367ecc…` | last full CI **2026-07-28**, **before** canonical was wired (2026-08-04); now stale/`DIRTY` | **NO** (gate did not exist on last green run) |
| #1034 | 2026-07-16 | `lean_single.sio` + `souc-lean-single` | ships ELF `a920eac0…`; Contracts green on that head | green checks; `CONFLICTING` vs main | **NO** — resync done once; era conflict blocks merge |
| #1605 | 2026-08-02 | `bin/souc` wrapper only | n/a | green; `CONFLICTING` | **N/A** (no lean_single.sio) |

### #1750 in detail (the exemplar)

Latest Contracts job `95770888702` (run `32155091510`, 2026-08-18):

- **Only step with `conclusion=failure`:** "Canonical lean_single fixed point"
  (step 40; ~2.4 s wall: 15:40:03 → 15:40:05).
- E219 / epistemic-correspondence lines print internal `FAIL:` diagnostics but
  the **steps themselves are `success`** (soft/control harnesses).
- Every other CI job on that run is green: Full Test Suite, Madaros Witness,
  Native Self-Host (Linux + macOS), Source-Bootstrap, Lint, Website, Lean Proofs.
- CI Decision fails solely because Contracts failed.
- GitHub merge state: **`CONFLICTING` / `DIRTY`** — so even a correct seed
  resync would still need a main rebase before merge.

Logged divergence:

```
bin/souc md5     = c28659d8538534ff0c0c1166c7e87b2a   # committed lean-single (main's)
self-compile md5 = c4fe3c517b2aea750259e8010af7cebb   # this PR's lean_single.sio
```

**Second-order trap on #1750:** the branch *did* once land a correct resync —
commit `4581f72345` (`build(seed): refresh … after the CUDA ABI fix`, md5
`296c8e3b…`). Subsequent merges of `origin/main` replaced the seed blob with
main's `c28659d8…` while keeping the PR's `lean_single.sio` edits. Resync is
not durable across main merges unless repeated. That is structural cost, not
author laziness alone.

---

## 2. Is there a sanctioned resync path?

### What exists (fragments)

| surface | what it does | ships/commits the seed? | wired into CI? |
|---|---|---|---|
| Gate fail message in `canonical_compiler_gate.sh` | prints a two-pass recipe | tells you to `cp … bin/souc` | yes (as stderr on FAIL) |
| `make build` | gen1→gen2→gen3 from `bin/souc-linux-x86_64`; checks gen2==gen3 | **no** — leaves `gen3.elf` at repo root | not as seed install |
| `scripts/ci/verify_lean_seed.sh` | verifies committed seed is fixed point (+ optional DDC); **on FAIL prints a derive recipe** using `bin/souc-linux-x86_64` → gen1 → gen2 → `cp gen2 bin/souc-lean-single-x86_64` | verify only | **not** referenced from `.github/workflows/` or Makefile |
| ADR-006 (`docs/decisions/adr-006-fixed-point-trust-anchor.md`) | policy: change source → validate FP → commit `.sio` + ELF together | policy for **boot4** artifact path; pre-dates dual naming | n/a |
| Scattered audit recipes | `scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 lean_single.sio /tmp/…` | ad hoc | no |
| `scripts/dev/seed-1678/README.md` | explains *why* ELF must ship with source | handoff note | no |
| Handoffs (`docs/handoff/compiler_generic_struct_return_diagnosis.md`) | "requires Foundry refresh of bin/souc-lean-single-x86_64" | names Foundry; no recipe | no |
| Commit `973b022b1a` message | documents Slurm off-pod resync, g0→g1→g2→g3 depth | one successful instance | no |

### What does **not** exist

- No operator guide titled anything like "when you edit `lean_single.sio`, run X, commit Y".
- No Makefile target `make refresh-seed` / `make install-seed` that copies gen3 into
  `bin/souc-lean-single-x86_64`.
- No CI job that *produces* the seed artifact for a PR (only a job that *refuses*
  a mismatched one).
- No durable link between `make build`'s gen3 and the path the gate checks.

### Naming drift (active footgun)

The gate **checks** `bin/souc-lean-single-x86_64` but its FAIL text still says:

```text
To resync:
  bin/souc $SRC /tmp/s1 && /tmp/s1 $SRC /tmp/s2 && cp /tmp/s2 bin/souc
  (verify /tmp/s2 self-reproduces, then commit bin/souc)
```

After the Madaros wrapper split (noted in the gate header, 2026-06-14),
`bin/souc` is a **10 KB shell wrapper**, not the lean_single ELF. Following the
printed recipe literally either fails or corrupts the wrong path.

#1758 illustrates the complementary error: it refreshed `bin/souc-linux-x86_64`
(a third, older bootstrap binary) and does not even carry
`bin/souc-lean-single-x86_64` at HEAD. Wrong name, multi-fail CI, still open.

`verify_lean_seed.sh` has the **correct** path names in its derive recipe. That
script is the closest thing to a sanctioned procedure — and it is discoverable
only if you already know to open it. It is not linked from CLAUDE.md §4,
`docs/guide/installation.md`, or the gate's own FAIL text.

### Verdict on Q2

**A mechanical path exists in pieces** (two- or three-generation self-compile,
optionally under `souc-build-lock.sh`, preferably off-pod/Slurm, commit
`bin/souc-lean-single-x86_64`). **A sanctioned, single documented operator path
does not.** The gate's own help text is stale and points at the wrong binary.
That gap *is* the discovery.

---

## 3. Real cost of a seed rebuild (from receipts — not run here)

### What is cheap

| operation | wall clock | source |
|---|---|---|
| Gate's single self-compile check on GHA | **~2.4 s** | #1750 Contracts step 40 timestamps |
| Native Self-Host (Linux x86_64) job | **~53 s** | same run |
| Source-Bootstrap Self-Host job | **~103 s** | same run |

The *check* is cheap. The *repair* is not the check.

### What historical resyncs actually did

| receipt | what was done | environment | wall clock stated? |
|---|---|---|---|
| `973b022b1a` (in #1768, 2026-08-17) | g0→g1→g2→g3; g1≠g2, g2==g3=`c28659d8…`; commit lean-single ELF | **Slurm off-pod** (`--partition=all`, `gpuorangefs-5860-proxmox`); explicit "zero pod CPU" | **no numeric minutes** — only "one generation more than a plain two-pass" |
| `4581f72345` (on #1750 branch) | old→s1→s2, s1==s2=`296c8e3b…` | not stated | no |
| `a30726e1c9` / #1606 follow-up | caught shipping **gen1** instead of fixed point; replaced with gen2 | local | no seed-only time; corpus parity was "**about two hours** of machine time" for 1688 programs — that is **not** the seed rebuild cost |
| `e63c569f8b` (#1678 handoff) | source patched, seed omitted first; gate was **not running** (Contracts died earlier until #1684) | — | shows the failure mode when the gate is dark |
| ADR-006 / `make build` | gen1→gen2→gen3 theory | pod-hostile under concurrency (CLAUDE.md §4) | no current measured wall time in docs |

### Pod vs Slurm

- CLAUDE.md concurrency discipline: full self-compile / `make build` under
  multi-agent load has **evicted the workspace pod** (2026-05-29, load ~153).
- `scripts/dev/souc-build-lock.sh` is the in-pod serialisation tool; it does not
  move work off-pod.
- The one resync that documents placement (`973b022b1a`) **went to Slurm on
  purpose**.
- Madaros heavy builds already have a Slurm path (`#1505`); lean_single seed
  refresh has **no parallel documented recipe** — only that one commit message.

### Implied cost (bounded without a live run)

- Lower bound: **2 successful self-compiles** of a ~2.5 MB lean_single (gate
  recipe). At GHA's ~2.4 s/compile that would be seconds — but GHA runners are
  not the pod, and cold/`ulimit -s`/lock contention dominate real sessions.
- Realistic lower bound when codegen changed: **3–4 generations** (973b:
  g1≠g2, settle at g2==g3) plus lock wait, plus `verify_lean_seed` / canonical
  re-check, plus commit of a binary blob.
- Upper bound observed near seed work: **~2 h** only when someone also re-runs
  full engine-parity corpus (#1606) — that is validation blast-radius, not the
  compile chain itself.
- **No receipt in-repo states "seed-only wall clock = N minutes."** Absence of
  that number is itself part of the structural cost: authors cannot plan the
  work from documentation.

---

## 4. Stall vs bypass — which is happening?

### Stall (dominant on today's open set)

| evidence | reading |
|---|---|
| 5 open PRs touch `lean_single.sio`; **0** current heads are both md5-clean *and* mergeable against main | queue does not clear |
| #1750: every substantive job green; **only** canonical blocks Contracts; open ≥2 days; seed resync once landed then **lost on main merge** | pure stall + non-durable repair |
| #1527, #1729: lean_single edits, no matching seed, era/`DIRTY`, other failures or no fresh Contracts | abandoned / unmaintainable under the rule |
| #1034: did the resync in its era; still unmergeable | even successful resync does not finish the story without continuous rebase |

### Bypass / near-miss patterns (historical, not "green by cheating the gate today")

| pattern | example | effect |
|---|---|---|
| Gate offline / short-circuited by earlier Contracts death | #1678 first handoff; fixed by #1684 | source-only patches could land unnoticed |
| Commit gen1, not fixed point | #1606 needed `a30726e1c9` correction after adversarial review | looks green until someone checks gen2==gen3 |
| Refresh the **wrong** binary | #1758 → `bin/souc-linux-x86_64`; gate text still says `bin/souc` | author believes they resynced; gate still red or irrelevant |
| Separate "seed refresh" commit by founder after the fact | `973b022b1a`, `4581f72345`, `8ef762a99d` | works when someone who knows the folklore is available; does not scale to every lane |
| Avoid touching lean_single | fleet preference / handoff Path A | real defects stay on Madaros-only or stay undiagnosed on the seed |

**Nobody is currently bypassing a live red canonical step with a green Contracts
on mainline PRs** — the gate, when it runs, holds. What happens instead is:

1. **PRs stall** (#1750 is the clean specimen), or  
2. **authors never attempt the resync** and the PR rots under conflicts, or  
3. **knowledgeable operators** land seed blobs in follow-up commits using
   unpublished Slurm folklore.

That is exactly the predicted failure mode: *a gate satisfiable only by heavy,
undocumented work becomes a gate nobody routine-satisfies.*

---

## 5. Answers in one place

1. **How many open PRs touch `lean_single.sio`?** **5** (of 50 open).  
   **Blocked exactly by md5 and nothing else (Contracts sole hard-fail)?** **1**
   — #1750. That one is still not merge-ready: `CONFLICTING`/`DIRTY`, and its
   earlier correct seed refresh was overwritten by merging main.

2. **Sanctioned regen path?** **Fragmented, not sanctioned as a single operator
   procedure.** Closest real recipe lives inside `scripts/ci/verify_lean_seed.sh`
   (not CI-wired). Gate FAIL text names the **wrong** output path (`bin/souc`).
   `make build` does not install the seed. **Discovery: the documentation gap
   is real.**

3. **Rebuild cost?** Check ≈ **2.4 s** on GHA. Repair = multi-generation
   self-compile; one documented instance used **Slurm off-pod** and needed
   **g0…g3** when codegen moved. **No numeric seed-only wall-clock receipt**
   exists in logs/PRs; corpus validation around seed swaps has hit **~2 h**.
   Pod is the wrong default under concurrency; Slurm is what successful recent
   resync actually used.

---

## 6. Follow-up (done in-session after this measurement)

Written without running a rebuild:

- Gate FAIL text now points at `scripts/dev/refresh_lean_seed.sh` and
  `docs/ops/LEAN_SINGLE_SEED_REFRESH.md` (no longer tells authors to
  `cp … bin/souc`).
- Executable recipe reconstitutes 8ef762 / 4581 / 973b; default mode is
  `--print`; `--execute` requires `SOUNIO_SEED_REFRESH_EXECUTE=1` and uses
  **`srun`** via `slurm_srun_minimal.sh` (never sbatch).
- Still open for a later founder decision: actually run the resync for #1750
  (and re-resync after its main merge).

---

## 7. Commands used (reproducible)

```bash
gh pr list --state open --limit 200 --json number
# per-PR files + checks
gh pr view <n> --json files,statusCheckRollup,mergeable,mergeStateStatus
gh pr checks 1750
gh api repos/Sounio-lang/sounio/actions/jobs/95770888702   # step conclusions
gh run view 32155091510 --job 95770888702 --log-failed
git log --oneline -- bin/souc-lean-single-x86_64
git show 973b022b1a   # Slurm resync receipt
git show 4581f72345   # #1750 seed refresh later clobbered
# blob md5s at PR heads vs origin/main
git cat-file -p <rev>:bin/souc-lean-single-x86_64 | md5sum
```

No `make build`, no seed compile, no modification of
`bin/souc-lean-single-x86_64` or `lean_single.sio` in this session.

---

*End of measurement.*
