<!-- docs:meta
topic_id: repo.docs.ops.lean-single-seed-refresh
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.lean-single-seed-refresh
-->

# lean_single seed refresh — the folklore, written

**Status:** recipe only. This document does **not** authorise a rebuild.  
**Driver:** `bash scripts/dev/refresh_lean_seed.sh`  
**Gate that enforces the rule:** `scripts/ci/canonical_compiler_gate.sh`  
**Slurm path:** `bash scripts/dev/slurm_srun_minimal.sh <cmd>` only — **never sbatch**.

---

## 0. Quantified cost — open PR census (re-measured 2026-08-18, open=51)

Criterion: PR file list contains `self-hosted/compiler/lean_single.sio`.  
Live Canonical signal = Contracts step `Canonical lean_single fixed point` present
on the latest completed CI run for that branch (gate wired `ed2cd88bf8`, 2026-08-04).

| class | n | PRs | evidence |
|---|---:|---|---|
| **A. Blocked ONLY by canonical md5** | **1** | **#1750** | Run `32155091510` (2026-08-18). Contracts `fail_steps` = only `Canonical lean_single fixed point`. No other CI job red. Codegen diff `+172/−32`. Seed blob = main's `c28659d8…` (not refreshed). |
| **B. Blocked by md5 + something else** | **0** | — | No open PR has Canonical fail *and* another CI job fail on the same head. |
| **C. Touches lean_single.sio and PASSES Canonical** | **1** | **#1729** | Run `31743279455` (2026-08-13). Canonical step = **success**. Diff `+4/−4` **comment-only**. Does **not** ship a new seed. (Overall CI may still be red on Madaros f64 — orthogonal.) |
| **X. Touches lean_single; no live Canonical step** | **3** | #1034, #1527, #1758 | See below. |

**The number: `folklore_cost_open_prs_blocked_only_by_canonical_md5 = 1`.**

### Category C — hole or rebuild author?

**Neither, for #1729.** Learning:

1. **Not a gate hole.** The gate compares md5(committed ELF) to md5(ELF compiling
   current source). Comment-only edits do not change the object code the seed
   emits, so the fixed point holds without a blob change. The gate protects
   **codegen identity**, not "file touched in git".
2. **Not a silent rebuild author.** #1729 does not include
   `bin/souc-lean-single-x86_64` in the PR files. Nobody on that PR demonstrated
   the resync procedure; they simply did not need it.
3. **Who already knew how:** #1034 (class X) **ships** `bin/souc-lean-single-x86_64`
   (`a920eac0…`) with its lean_single edit — that is an in-era rebuild, but its
   last Contracts run is **2026-07-17**, before the gate existed, so it is not
   live class C. Authors of merged seed commits `8ef762` / `4581` / `973b` are
   the procedure sources reconstituted into this recipe.

### Class X detail

| PR | why no live Canonical |
|---|---|
| #1034 | Contracts green Jul 17 — **pre-gate**; ships seed (did rebuild then) |
| #1527 | Contracts green Jul 28 — **pre-gate**; no seed in PR |
| #1758 | Contracts fails at **Docs registry** (2026-08-17); never reaches Canonical; multi-fail; wrong ELF (`souc-linux`); no `souc-lean-single` at head |

Full narrative: `docs/audit/CANONICAL_COMPILER_GATE_STRUCTURAL_COST_2026-08-18.md`.

---

## 1. What the three commits actually did

Reconstituted from commit messages and blobs. No live rebuild in the session
that wrote this page.

### 1.1 `8ef762a99d` + `a30726e1c9` — PR #1606 (2026-08-02/03)

| | |
|---|---|
| **Why** | Committed seed lagged two `lean_single.sio` fixes; parity gate blamed Madaros for seed bugs |
| **Start ELF** | `bin/souc-linux-x86_64` (third bootstrap binary) |
| **Chain** | linux → gen1 → gen2 → gen3 |
| **Where** | not stated (pre-Slurm-recipe era) |
| **What they almost shipped** | **gen1** (`305f4cfa…`) — wrong |
| **What is correct** | **gen2** where gen2 == gen3 (`f330d612…` then) |
| **End verification** | `sha256(gen2) == committed`; gen2 built twice agrees; optional full corpus (~2 h) is blast-radius, not the seed proof |
| **Lesson** | Generation 1 of a foreign bootstrap is **never** the fixed point of current source. Ship the first `genN` with `genN == genN+1`. |

Exact command recorded in `a30726e1c9`:

```bash
./bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio /tmp/gen1.elf
chmod +x /tmp/gen1.elf
/tmp/gen1.elf self-hosted/compiler/lean_single.sio /tmp/gen2.elf
sha256sum /tmp/gen2.elf   # must equal the committed artifact
```

### 1.2 `4581f72345` — on PR #1750 branch (2026-08-16)

| | |
|---|---|
| **Why** | CUDA ABI + PTX param matching edited `lean_single.sio` without ELF |
| **Start ELF** | previous committed lean_single seed ("bin/souc (old)" in the message = seed, not the wrapper) |
| **Chain** | old → s1 → s2 with **s1 == s2** (`296c8e3b2b0581ad01844df27a59e1f5`) |
| **Where** | not stated |
| **End verification** | `canonical_compiler_gate.sh` PASS; `make build` gen2==gen3, **same md5** |
| **Lesson** | When the new source's codegen is already produced by g1, two passes settle. Always still check s1==s2 before commit. |

### 1.3 `973b022b1a` — inside #1768 (2026-08-17)

| | |
|---|---|
| **Why** | FFI `system()` touched `append_extern_c_stubs` in `lean_single.sio`; gate red on #1768 (`25fb229c…` vs `489cda9b…`) |
| **Start ELF** | committed seed g0 |
| **Chain** | g0 → g1 → g2 → g3; **g1 ≠ g2**, **g2 == g3** (`c28659d8538534ff0c0c1166c7e87b2a`) |
| **Where** | **Slurm off-pod**, `--partition=all`, node `gpuorangefs-5860-proxmox`, zero pod CPU |
| **End verification** | g2 compiling `lean_single.sio` reproduces byte-identical g3 — exactly the property `canonical_compiler_gate.sh` checks |
| **Lesson** | Codegen that is part of the compiler's own bootstrap surface needs **one extra** self-application. `g1 ≠ g2` is expected, not a non-determinism bug. Prefer off-pod. |

### 1.4 Unified rule extracted from all three

```
start = committed lean_single ELF   (preferred)
     OR bin/souc-linux-x86_64       (Makefile path; never ship gen1)

for i = 1, 2, 3, …:
    gen[i] = gen[i-1] compiling lean_single.sio

ship gen[k]  where  k >= 1 and md5(gen[k]) == md5(gen[k+1])
             and gen[k] compiling lean_single.sio == gen[k]   (self-repro)

install path:  bin/souc-lean-single-x86_64     ONLY
never:         bin/souc   (Madaros wrapper)
never:         bin/souc-linux-x86_64 as the "refreshed seed"
```

---

## 2. Executable recipe (follow in order)

### 2.0 Print the recipe from the tree

```bash
bash scripts/dev/refresh_lean_seed.sh --print
bash scripts/dev/refresh_lean_seed.sh --cost    # folklore cost number only
```

### 2.1 Confirm you actually need a resync (cheap)

```bash
bash scripts/ci/canonical_compiler_gate.sh
# FAIL with two different md5s → continue
# PASS → stop; do not refresh
```

Or:

```bash
bash scripts/dev/refresh_lean_seed.sh --check
```

### 2.2 Placement — founder decides; cluster is up

| Path | Status | How |
|---|---|---|
| **`srun` via `scripts/dev/slurm_srun_minimal.sh`** | **Supported** | default for `--execute --via-slurm` |
| **`sbatch`** | **Forbidden** | `user_env_retrieval_failed_requeued_held` for `openvscode-server`; held jobs are corpses, not load |
| Pod + `souc-build-lock.sh` | Last resort | eviction history under concurrent self-compiles |

Positive control (measured 2026-08-17, still the contract):

- helper: `bash scripts/dev/slurm_srun_minimal.sh '…'`
- partition: `cpu-ops` (or `all`)
- example host: `cpuops-t560-proxmox`, 32 cores, rc=0
- `/workspace` **invisible** on compute; `/orangefs` **visible**
- details: `docs/ops/SLURM_LAUNCH_REPAIR_2026-08-17.md`

**Do not write recipes that say `sbatch`.**

### 2.3 Stage inputs onto OrangeFS (login/pod — no compile)

```bash
bash scripts/dev/refresh_lean_seed.sh --stage
# prints STAGE_DIR=/orangefs/training/sounio/seed-refresh/<UTC>/
```

Staged minimum:

- `self-hosted/compiler/lean_single.sio`
- `bin/souc-lean-single-x86_64` (g0)
- optionally `bin/souc-linux-x86_64`
- the two verify scripts

### 2.4 Derive + install (founder only — consumes cluster)

```bash
# DOES THE REBUILD. Do not run from an agent lane without founder go-ahead.
SOUNIO_SEED_REFRESH_EXECUTE=1 \
  bash scripts/dev/refresh_lean_seed.sh --execute --via-slurm
```

What the driver runs on the node (same loop as §1.4):

1. `ulimit -s 1048576`
2. g0 = staged seed
3. compile until `md5(g_{i-1}) == md5(g_i)` with `i >= 2`
4. extra self-repro: settled ELF compiling SRC == itself
5. copy settled ELF back to workspace `bin/souc-lean-single-x86_64`

Pod fallback (discouraged):

```bash
SOUNIO_SEED_REFRESH_EXECUTE=1 \
  bash scripts/dev/refresh_lean_seed.sh --execute --local-locked
```

### 2.5 HARD STOP — fixed point before anything else

```
╔══════════════════════════════════════════════════════════════════╗
║  DO NOT install or commit a seed without M1.                     ║
║                                                                  ║
║  M1 (mandatory, not optional):                                   ║
║      exists k ≥ 1 such that  md5(g_k) == md5(g_{k+1})             ║
║      If start ELF ≠ the seed you will ship, k ≥ 2                ║
║      (never ship generation 1 of a foreign bootstrap — #1606).   ║
║                                                                  ║
║  A rebuild that only produces "a different ELF" is NOT success.  ║
║  Different-but-unsettled is worse than no refresh: it looks      ║
║  green and is wrong.                                             ║
║                                                                  ║
║  The driver refuses to install without writing out/SETTLED.md5.  ║
║  If you derive by hand and skip M1, you are off-recipe. STOP.    ║
╚══════════════════════════════════════════════════════════════════╝
```

#### M1 during derive (before `cp`)

```bash
# after each generation i >= 2:
if md5(g_{i-1}) == md5(g_i); then
  echo "M1 PASS settle=g$((i-1))==g${i} md5=..."
  # only then may you consider install
else
  continue compiling   # or FAIL if MAX_GENS exceeded
fi
```

#### M2 + M3 after install (still mandatory)

```bash
# M2 — self-repro of the installed blob
bash scripts/ci/canonical_compiler_gate.sh
# expect: committed md5 == self-compile md5 == <H>   (same <H> as M1)

# M3 — determinism
bash scripts/ci/verify_lean_seed.sh
# expect: FIXED POINT ok + DETERMINISTIC ok
```

| id | when | criterion | if skipped |
|---|---|---|---|
| **M1** | **before install** | `md5(g_k)==md5(g_{k+1})` | may ship gen1 / non-fixed point — **#1606 failure mode** |
| **M2** | after install | installed ELF self-reproduces | gate red or silent wrong blob path |
| **M3** | after install | two compiles agree | non-deterministic emitter unnoticed |

Commit message **must** contain the M1 line `settle: gK==g{K+1} md5=<H>`.  
No M1 line → do not merge.

#### Optional only after M1–M3 are green

```bash
SOUNIO_SEED_DDC=1 bash scripts/ci/verify_lean_seed.sh   # independent start → same FP
scripts/dev/souc-build-lock.sh make build                 # Makefile chain; md5 must == <H>
```

**Never sufficient alone:** "md5 changed", "binary is newer", "make build ran",
"file size grew", "CI green on an unrelated job", "procedure exited 0".

### 2.5b SeedReceipt — procedure vs proof

`--execute` always writes a **SeedReceipt** (not optional):

```
artifacts/seed-refresh/SeedReceipt-<UTC>.json
artifacts/seed-refresh/SeedReceipt-<UTC>.txt
artifacts/seed-refresh/SeedReceipt.latest.json   # stable pointer
```

Schema: `docs/ops/SEED_RECEIPT.schema.json`  
Emitter: `scripts/dev/write_seed_receipt.py`  
Validate later:

```bash
bash scripts/dev/refresh_lean_seed.sh \
  --verify-receipt artifacts/seed-refresh/SeedReceipt.latest.json
```

| field | purpose |
|---|---|
| `source.sha256` | exact `lean_single.sio` bytes that were compiled |
| `input_seed.sha256` | g0 / starting ELF (before the chain) |
| `generations[]` | g0…gN with md5 + sha256 each |
| **`fixed_point`** | **FIELD**, not a step log — see below |
| `output_seed.sha256` | installed / to-commit ELF |
| `environment` | placement, hostname, `slurm_partition`, `slurm_job_id`, nodelist |
| `checks` | canonical_compiler_gate / verify_lean_seed pass\|fail |
| `limits.provenance_note` | what the receipt deliberately does **not** claim |

**Fixed point as a field (must be confirmable by eye):**

```text
--- md5 side-by-side (must be identical) ---
gk_md5:       <H>
gk_plus1_md5: <H>
md5_equal: true
--- sha256 side-by-side (must be identical) ---
gk_sha256:       <S>
gk_plus1_sha256: <S>
sha256_equal: true
verified: true
```

If a reader cannot confirm `gk_md5 == gk_plus1_md5` without re-running the
chain, the receipt proves nothing. `--verify-receipt` fails closed when the
two lines differ or `verified` is not true.

**What a green `canonical_compiler_gate` self-repro leg is not:** it checks
`md5(committed ELF) == md5(that ELF compiling current source)` — **stability**.
It does **not** alone prove the ELF was *derived from* that source.

**Provenance leg (now wired):** `scripts/ci/seed_receipt_provenance_gate.sh`
runs after self-repro inside `canonical_compiler_gate.sh`.

| tree / change set | gate result |
|---|---|
| receipt present, matches source+seed+FP | **PASS** |
| receipt present, `source.sha256` ≠ live `lean_single.sio` | **FAIL** (mutant control proves this path every run) |
| **no receipt**, PR does **not** touch seed surface | **PASS** (main and unrelated PRs stay green) |
| **no receipt**, PR touches `lean_single.sio` / seed ELF / receipt path | **FAIL** — must land a receipt with the change |
| `SOUNIO_SEED_RECEIPT_REQUIRED=1` and no receipt | **FAIL** (optional later global flip) |

**Policy choice:** require receipt **only when the change set touches the seed
surface**, not a fake bootstrap receipt and not permanent warn-only. Reason:
limits force to the case the recipe exists for; never paints main red for
absence alone; still hard-checks any receipt that *is* committed. Tradeoff:
the committed ELF on main has no permanent paper trail until someone lands the
first receipt — self-repro still runs; provenance is enforced at the moment of
seed-surface change (#1750 class).

Committed path (tracked, not under gitignored `artifacts/`):

```
bin/souc-lean-single-x86_64.SeedReceipt.json
```

After `--execute`, the driver copies the latest receipt there — commit it with
the ELF on any seed-surface PR.

Founder optional first-receipt on main (no rebuild in the gate PR itself):
**~5–15 min** idle srun, same recipe as §2.4.

### 2.6 Commit

```bash
git add bin/souc-lean-single-x86_64
# optional but recommended: commit the receipt next to the blob or under artifacts/
git add artifacts/seed-refresh/SeedReceipt-<UTC>.json
# if SRC changed in the same PR, add it in the same commit or the immediately
# preceding one — never land SRC without the matching ELF on the merge tip.
git commit -m "$(cat <<'EOF'
build(seed): refresh bin/souc-lean-single-x86_64 to lean_single fixed point

settle: gK==g{K+1} md5=<H>
receipt: artifacts/seed-refresh/SeedReceipt-<UTC>.json
canonical: committed==self-compile==<H>
placement: srun/cpu-ops
verified: canonical_compiler_gate + verify_lean_seed
EOF
)"
```

### 2.7 After every merge of `main` into a lean_single-touching branch

Re-run §2.1. If red, re-run §2.3–2.6.

**#1750 already paid this tax:** commit `4581f72345` installed a correct seed
(`296c8e3b…`); later merges of `main` replaced the blob with main's
`c28659d8…` while keeping the PR's `lean_single.sio`. Resync is not durable
across main merges unless repeated.

---

## 3. Wrong paths (measured footguns)

| Action | Why it fails |
|---|---|
| `cp settled bin/souc` | `bin/souc` is the Madaros wrapper (~10 KB). Gate checks `bin/souc-lean-single-x86_64`. |
| `cp settled bin/souc-linux-x86_64` | Third binary; #1758 did this. Gate never looks there. |
| Commit gen1 from `souc-linux` | #1606 first attempt; adversarial review caught it in `a30726e1c9`. |
| Trust "md5 changed" | Different ≠ fixed point. |
| `sbatch …` | Held forever for this submitter. Use `slurm_srun_minimal.sh`. |
| Assume `/workspace` on compute | Invisible. Stage on `/orangefs`. |
| Skip resync after merging main | #1750 seed clobber. |

The gate used to print the `bin/souc` recipe on FAIL. That text is corrected to
point at this document and `refresh_lean_seed.sh`.

---

## 4. Wall-clock estimate (from receipts — no new rebuild)

None of 4581 / 973b / a30726 state "seed-only took N minutes". Bounds from what
they *did* record, plus one live CI measurement:

| observation | value | source |
|---|---|---|
| 1× self-compile of lean_single on GHA | **~2.4 s** | #1750 Contracts Canonical step 15:40:03→15:40:05 |
| Generations needed (codegen-neutral settle) | **2** (g0→g1→g2, g1==g2) | 4581 |
| Generations needed (bootstrap-surface codegen) | **3–4** (g0…g3, g1≠g2, g2==g3) | 973b |
| Full engine-parity corpus around a seed swap | **~2 h** | #1606 — **not** seed derive cost |
| Placement of 973b | Slurm off-pod, partition `all` | commit body |
| Positive srun control | `cpuops-t560-proxmox`, 32 cores, rc=0 | founder / SLURM repair doc |

**Estimated seed-only wall clock on idle srun (planning number):**

| phase | estimate |
|---|---|
| stage to `/orangefs` | &lt; 1 min (copy ~2.5 MB + source) |
| derive 2–4 self-compiles | **~10 s – 2 min** if GHA-like; pad to **5–15 min** on first cold node / disk |
| install + M1–M3 verify | **&lt; 1 min** |
| **Total seed-only** | **~5–15 min** typical; **≤ 45 min** hard budget (`--time=00:45:00`) |
| + optional DDC | roughly ×2 derive |
| + optional #1606-style corpus | **+ ~2 h** — separate decision |

Honesty: the **5–15 min** band is extrapolated from ~2.4 s/compile × 3–4 gens
plus staging/srun overhead, not a stopwatch in those three commits. The commits
prove the *shape* of the work; they do not publish a timer. If a run exceeds
45 minutes something other than the derive loop is wrong (queue, lock, wrong
binary path).

---

## 5. Authority

- **Writing this recipe:** any lane (documentation).
- **Running `--execute`:** founder (or explicit founder go-ahead). Env var
  `SOUNIO_SEED_REFRESH_EXECUTE=1` is the deliberate friction.
- **Changing the gate criterion:** separate dispatch; not this page.

---

## 6. File map

| path | role |
|---|---|
| `scripts/dev/refresh_lean_seed.sh` | print / check / stage / execute / verify-receipt driver |
| `scripts/dev/write_seed_receipt.py` | emit + validate SeedReceipt (JSON + human .txt) |
| `docs/ops/SEED_RECEIPT.schema.json` | SeedReceipt schema v1 |
| `scripts/dev/slurm_srun_minimal.sh` | supported Slurm launch (`srun` only) |
| `scripts/dev/souc-build-lock.sh` | pod serialisation if local-locked |
| `scripts/ci/canonical_compiler_gate.sh` | CI FAIL/PASS on self-repro md5 match (not provenance) |
| `scripts/ci/verify_lean_seed.sh` | FP + determinism (+ optional DDC) |
| `bin/souc-lean-single-x86_64` | **the** committed seed ELF |
| `self-hosted/compiler/lean_single.sio` | seed source |
| `artifacts/seed-refresh/` | default SeedReceipt output directory |
| `docs/audit/CANONICAL_COMPILER_GATE_STRUCTURAL_COST_2026-08-18.md` | why the recipe exists |

---

*End of recipe. Default command remains `--print`. No rebuild was run to author this.*
