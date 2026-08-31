<!-- docs:meta
topic_id: repo.docs.audit.omega-stale-artifact-audit-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.omega-stale-artifact-audit-2026-08-16
-->

# Omega stale status artifacts — audit and disposition (2026-08-16)

**Date:** 2026-08-16  
**Agent / lane:** grok-cli5 / `omega-artifact-audit-v2`  
**Scope:** investigation + documentation only. **No hand-edit** of the JSON
artifacts. **No `self-hosted/` edits.**  
**Prior failure:** a previous claim on this path reported the audit written;  
`ls` proved the file did not exist. This note is the real deliverable.

**Rebase provenance (2026-08-27).** Republished on `origin/main@055825a3f9`. Re-checked
structurally at the new base: `scripts/omega/omega_parallel_cutover_status.py` and the
`artifacts/omega/*.v1.json` set are still present and still un-regenerated. The dry-run
`status=fail` result below is **not** re-executed here — it is a dated 2026-08-16
measurement. Disposition (RETIRE as operational source of truth) is unchanged.

### Measurement mode (plain)

**INSPECTED + producer dry-run only — NOT a full re-derivation of the artefacts.**

| Done | Not done |
|------|----------|
| Read both JSON files and `generated_at_utc` | Re-run `scripts/archive/omega_sprint1_gate.sh` on Slurm |
| Read `omega_parallel_cutover_status.py` end-to-end | Emit new committed `*.v1.json` with a real gate log |
| Confirm gate log missing; archive-only producer | Hand-edit JSON to look current (**refused**) |
| Dry-run producer to `/tmp` only → **status=fail** | Treat dry-run fail as a fresh green snapshot |

Inspection alone cannot prove an on-disk green is still true; the dry-run shows an
honest re-emit **without** inventing a gate log would be **fail**. Disposition is
still **RETIRE as operational source of truth**, not “regenerate until green.”

---

## 0. Recommendation (plain)

### **RETIRE** these two files as operational sources of truth.

Do **not** regenerate them to show `status: pass` under current HEAD.  
Do **not** hand-edit the JSON timestamps or fields to look current.

| Option | Verdict |
|--------|---------|
| Hand-edit JSON to look current | **Refuse** (same class of retrofit fable-1 refused) |
| Regenerate honestly as *current* operational snapshot | **No** — the measured surface no longer maps to how the compiler is verified |
| Keep as historical Feb-2026 receipts, stop citing as “now” | **Yes** |
| Point agents at live gates instead | **Yes** (index / contract update — separate change) |

**One sentence:** they are six-month-old **omega sprint-1 cutover marker digests**,
not a live Madaros or dissertation truth clock; citing them as green today is the
same disease as the MADAROS_STATUS 10/10 vs 4/4 mismatch and the June 6/6 vs
job-9908 1/3/2 dissertation gate re-measure.

---

## 1. The two artifacts (on disk)

```text
# measured 2026-08-16 on worktree /workspace/.wt/grok-cli5
artifacts/omega/selfhost_compiler_progress.v1.json
  size=1191  generated_at_utc=2026-02-26T11:48:43.338890Z  status=pass  order_status=pass

artifacts/omega/parallel_cutover_status.v1.json
  size=403   generated_at_utc=2026-02-26T11:48:43.338911Z  status=pass
  track_a_status=pass  track_b_status=pass  track_b_order_status=pass
  provenance_status=pass  no_rust_surface_status=pass
  no_rust_runtime_absence_status=pass  crates_mainline_removed_status=pass
  blocking_failures=[]
```

**Age:** 170 days from `generated_at_utc` to 2026-08-16.  
**Git history:** last content commits are 2026-02-26 only
(`9178b480ae`, `ece571eb42`, …). Worktree mtime 2026-08-14 is checkout/sync,
not a fresh measurement.

---

## 2. What they actually measured

### 2.1 Producer

Single emitter:

```text
scripts/omega/omega_parallel_cutover_status.py
```

Writes **both** JSON files in one run from:

| Input | Role |
|-------|------|
| `--gate-log` default `artifacts/omega_sprint1_gate.log` | String-search for Track A/B **marker tokens** |
| `artifacts/omega/souc_release_provenance.v1.json` | Track A provenance `status == pass` |
| `artifacts/omega/no_rust_execution_surface.v1.json` | Track A no-rust report + runtime absence flags |
| presence/absence of top-level `crates/` | Track A “crates mainline removed” |

**It does not compile Madaros. It does not run the test suite. It does not run
`make madaros-full-gate`.** It greps a log for substrings and copies sibling JSON
`status` fields.

### 2.2 Track B (`selfhost_compiler_progress.v1.json`)

Ordered **plan markers** from the Feb-2026 self-host / GPU IR cutover sequence:

| stage | marker (must appear in gate log) |
|-------|----------------------------------|
| data_structures | `DATA_STRUCTURES_PASS` |
| gpu_ir_expansion | `GPU_IR_EXPANSION_PASS` |
| hlir_gpu_cross_coverage | `HLIR_GPU_CROSS_COVERAGE_PASS` |
| hlir_lowering | `HLIR_LOWERING_PASS` |
| metal_msl_codegen | `METAL_MSL_CODEGEN_PASS` |
| ptx_regalloc_expansion | `PTX_REGALLOC_EXPANSION_PASS` |
| gpu_opcode_smoke | `GPU_OPCODE_SMOKE_PASS` |

`order_status` only checks that a subset of those markers appear in plan order
in the log. **Presence of the string is the entire oracle.**

### 2.3 Track A + composite (`parallel_cutover_status.v1.json`)

Track A requires log markers `REPO_HARD_NO_RUST_PASS`,
`SOUC_RELEASE_PROVENANCE_PASS`, plus provenance JSON pass, no-rust surface pass,
`require_runtime_absence == true` with zero runtime violations, and no `crates/`
directory.

Composite `status` is pass iff `blocking_failures` is empty.

### 2.4 Historical caller

```text
scripts/archive/omega_sprint1_gate.sh   # ~line 1834
  python3 scripts/omega/omega_parallel_cutover_status.py --gate-log "$GATE_LOG"
```

The only in-tree **producer** invocation found is under **`scripts/archive/`**.  
**No** match in `Makefile`, `scripts/ci/` (except consumers of the *outputs*), or
`.github/` for `omega_sprint1_gate` / live production of these two files.

---

## 3. Do the gates still exist and still run?

| Piece | Exists? | Still a live verification path? |
|-------|---------|----------------------------------|
| `scripts/omega/omega_parallel_cutover_status.py` | Yes | Runnable as a script; **not** wired as current compiler truth |
| `scripts/archive/omega_sprint1_gate.sh` | Yes (archive) | **Not** CI/Makefile; archive-only |
| `artifacts/omega_sprint1_gate.log` | **Missing** on this worktree | Required input for honest Track B markers |
| `souc_release_provenance.v1.json` | Yes | Stale itself (`generated_at_utc` 2026-03-06; path `/home/demetrios/work/sounio/...`; version `0.100.3`) |
| `no_rust_execution_surface.v1.json` | Yes | Stale (2026-03-07); `require_runtime_absence: false` while listing cargo/rustc on a VM path |
| Consumers: `plan_big_status_board.sh`, `plan_big_gate.sh`, `omega_hardware_telemetry_regression.py` | Yes | Still **read** the stale JSONs and can treat `pass` as critical |
| **Current** compiler proof | `make madaros-full-gate` → `scripts/ci/madaros_full_gate.sh`, `madaros_operational_contract_gate.sh`, epistemic/dissertation gates | **Different instrument family** |

### 3.1 Dry-run of the producer **today** (no write to committed paths)

```bash
python3 scripts/omega/omega_parallel_cutover_status.py \
  --out-cutover /tmp/parallel_cutover_status.dryrun.json \
  --out-progress /tmp/selfhost_compiler_progress.dryrun.json
# EXIT=2
# status=fail track_a=fail track_b=fail order=pass
```

Measured blocking failures (2026-08-16T13:14:05Z dry-run):

- `missing gate log: artifacts/omega_sprint1_gate.log`
- all seven Track B markers fail (`gate_log_offset: -1`)
- Track A: missing log markers + `NO_RUST_RUNTIME_ABSENCE_PASS`
  (`require_runtime_absence` is false in the on-disk no-rust report)

So an **honest** re-emit **right now**, without inventing a gate log, would replace
the committed green snapshot with **fail**. That is evidence the committed green
is **not** a live measurement — it is a frozen Feb-26 digest.

Committed JSON paths were **not** overwritten by this dry-run.

---

## 4. How agents are still misled

### 4.1 `.claude/OPERATIONAL_CANONICAL_INDEX.md` (active contract text)

```text
## Current Operational Snapshot
- Track A (no-rust cutover): pass
- Track B (self-host sequence): pass
- Track B order: pass
- Composite cutover status: pass

Source of truth:
- artifacts/omega/selfhost_compiler_progress.v1.json
- artifacts/omega/parallel_cutover_status.v1.json
```

Any agent that obeys this index treats a **170-day-old** marker digest as fact
about **now**.

### 4.2 Other still-live citations

- `.claude/PLAN_CANONICAL_EXECUTION.md` — canonical precedence list includes both files  
- `.claude/PROMPT_EXECUTION_CONTRACT.md`  
- `docs/implementation/CODEX_CLAUDE_PARALLEL_CONTRACT.md`  
- `docs/internal/implementation/CODEX_CLAUDE_PARALLEL_CONTRACT.md`  
- `artifacts/omega/claude_operational_contract_status.v1.json` (2026-05-12) lists both in `canonical_precedence`  
- `scripts/ci/plan_big_gate.sh` requires `.summary.parallel_cutover_status == "pass"` from a board that **reads** these files (`plan_big_status_board.v1.json` generated 2026-05-07 still shows parallel_cutover pass)

### 4.3 Same disease class (external confirmations — not re-derived here)

| Incident | Pattern |
|----------|---------|
| minimax-cli2 / `MADAROS_STATUS.md` | Headline 10/10 PASS vs on-disk 4/4 receipt, worktree hundreds of commits behind |
| Slurm job 9908 dissertation six-gate A/B on `main@6f2c4e2461` | June claim 6/6 green → measured **1 pass / 3 fail / 2 unmeasured**; identical on checked-in and from-source Madaros |
| WS-E HLIR re-verify | Feb omega-era HLIR “pass” markers vs current Madaros HLIR/GPU oracle breakage |

Stale green is **decision-grade false confidence**, not a doc nit — especially
~37 days before dissertation defense.

---

## 5. What honest regeneration would take (and why it is the wrong product)

To **regenerate** the two JSONs with the existing schema:

1. Restore or re-run something that emits `artifacts/omega_sprint1_gate.log` with the
   seven Track B strings and Track A strings (historically
   `scripts/archive/omega_sprint1_gate.sh`).
2. Refresh provenance + no-rust surface reports under **current** paths (not
   `/home/demetrios/work/sounio/...`).
3. Run `omega_parallel_cutover_status.py` and commit only if the log is real.

**Compute note (if anyone attempts the archived sprint gate):** do not load the
workspace pod. Use Slurm:

```bash
env SLURM_CONF=/tmp/slurm-direct.conf \
  srun --partition=cpu-ops --chdir=/tmp --time=120 bash -lc '
    git clone --depth 1 <remote-or-bundle> sounio-omega-audit && cd sounio-omega-audit
    # compute nodes cannot see /workspace
    bash scripts/archive/omega_sprint1_gate.sh   # if still runnable
  '
```

**Why this still should not become “Current Operational Snapshot” even if green:**

- Marker grepping ≠ Madaros E2E (`make madaros-full-gate`).  
- Marker grepping ≠ dissertation suite (job 9908: 24/53 FAIL under both engines).  
- Track B “HLIR_LOWERING_PASS” in a log is **not** the same claim as
  `docs/audit/HLIR_REVERIFY_2026-08-16.md` under Madaros v0.80.0.  
- The archive gate is a **2026-02 product era** (omega sprint-1 / no-rust cutover
  theater). Current verification authority is Madaros + named CI gates +
  attention charter.

Regenerating green would **re-arm** the false-confidence weapon. Regenerating
fail without retiring the index citations would still leave agents reading
OPERATIONAL_CANONICAL_INDEX’s hard-coded “pass” bullets.

---

## 6. What *does* map to how the compiler is verified now

Prefer these as operational truth (non-exhaustive; executable over narrative):

| Surface | Role |
|---------|------|
| `make madaros-full-gate` / `scripts/ci/madaros_full_gate.sh` | Stage1 Madaros E2E |
| `scripts/ci/madaros_operational_contract_gate.sh` | Wrapper/contract drift |
| `bin/souc --version` + fresh gate logs with `generated_at` / commit SHA | Identity |
| Named science gates (epistemic trust, dissertation suite, etc.) with **dated** receipts | Domain claims |
| `bin/sounio-coord brief` + worktree SHA | Multi-agent presence — not semantic pass |

`docs/MADAROS_STATUS.md` must be treated as **suspect until a fresh receipt**
matches its headline (per minimax-cli2 finding) — same principle as these omega
files.

---

## 7. Disposition checklist (recommended follow-on; not done in this audit)

1. **Retire** the two JSON files from “source of truth” language in:
   - `.claude/OPERATIONAL_CANONICAL_INDEX.md`
   - `.claude/PLAN_CANONICAL_EXECUTION.md`
   - `.claude/PROMPT_EXECUTION_CONTRACT.md`
   - parallel-contract docs  
   Replace with: “historical omega sprint-1 receipts (2026-02-26); do not treat
   `status: pass` as current.”
2. Optionally move or annotate the JSON with a **sidecar**  
   `*.STALE.md` / schema field `lifecycle: historical` — only if a later change
   is allowed to touch artifacts; **do not** flip `status` without a real run.
3. Stop `plan_big_gate.sh` from treating stale parallel_cutover `pass` as critical
   **or** make the board fail closed when `generated_at_utc` is older than N days
   (fix the instrument; do not forge the reading).
4. Keep `omega_parallel_cutover_status.py` only as a historical emitter or delete
   after consumers are unhooked — out of scope for this note.

---

## 8. What was measured vs not measured (honesty box)

**Measured:**

- Full contents and `generated_at_utc` of both JSON files  
- Producer script logic end-to-end  
- Absence of `artifacts/omega_sprint1_gate.log`  
- Archive-only producer call site; no Makefile/CI producer  
- Dry-run emit to `/tmp` → fail with listed blockers (committed files untouched)  
- Consumer and index citations listed above  
- Age = 170 days; last git content commits 2026-02-26  

**Not measured:**

- Full re-execution of `scripts/archive/omega_sprint1_gate.sh` on Slurm  
- Whether every Track B marker step would still print under Madaros v0.80.0  
- Fresh `make madaros-full-gate` on this worktree (out of scope; separate WS-A)  

---

## 9. Bottom line

| Question | Answer |
|----------|--------|
| What did they measure? | Feb-2026 omega sprint-1 **log marker + provenance digest** for no-rust cutover (Track A) and GPU/HLIR plan sequence (Track B). |
| Do producing gates still run as current truth? | **No.** Emitter exists; sprint-1 gate is **archived**; required gate log **missing**. |
| Regenerate honestly as current snapshot? | **No** — wrong instrument for today’s compiler/dissertation verification. |
| Disposition? | **RETIRE** as operational source of truth; keep as historical; fix instruments/indexes that still cite them as green **now**. |

*Audit complete. JSON files were not modified.*
