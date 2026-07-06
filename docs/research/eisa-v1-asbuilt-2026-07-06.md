<!-- docs:meta
topic_id: repo.docs.research.eisa-v1-asbuilt-2026-07-06
authority: historical/research
audience: researchers
last_validated: 2026-07-06
validated_by: as-built audit (git + file inspection)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.eisa-v1-asbuilt-2026-07-06
-->

<!-- docs:status-note:start -->
> Docs status: `historical/research`
> As-built chronicle of the EISA v0/v1 stack as shipped on branch
> `gpu/epistemic-tensor-core-next` in worktree `/workspace/sounio-eisa`.
> Records what landed in git, not what the v1/v2 plans originally promised.
<!-- docs:status-note:end -->

# EISA v0/v1 — as-built chronicle (2026-07-06)

## 1. Summary

As of 2026-07-06 the EISA stack in this worktree is a self-contained epistemic
executable pipeline: the dual-plane `.eisax` container (`stdlib/eisa/format.sio`,
assembled from `.eisa` text via `stdlib/eisa/asm.sio`), the **Metron VM** primary
executor (`stdlib/eisa/evm.sio`; internal module name `eisa::evm`), a Sounio
→ `.eisax` **Metron** surface compiler (`stdlib/eisa/backend.sio`), reference
semantics in `stdlib/eisa/core.sio` (dd64 err lane, v0/v1) and
`stdlib/eisa/core_v2.sio` (qd128 err lane, v2/W3), and an x86-64 AOT
**conformance bridge** (`stdlib/eisa/bridge_x86.sio`) checked byte-for-byte
against the VM via `scripts/ci/eisa_bridge_conformance_gate.sh`. Every register
carries three lanes — `val` (IEEE f64), `err` (dd64 in v0/v1 images; qd128 in
v2 images), and `u` (GUM σ) — plus poison/frail machinery from v1; execution
evidence is emitted as versioned receipts (`v=1` v0 EVM, `v=2` v1 Metron VM,
`v=3` v2 Metron VM) citing the container's `prog_hash`.

Lineage docs: `eisa-stack-architecture-2026-07-05.md` (v0 design + E-phases),
`eisa-v0-spec-2026-07-05.md`, `eisa-v1-plan-2026-07-05.md` (plan; contrast
with §3 below), `eisa-v2-arch-2026-07-05.md` (what v2 supersedes; §6).

---

## 2. Phase table

Commit hashes verified with `git show --stat <hash>` against the files listed.
Witness suite names and counts taken from each test file's header comments and
final `println("ALL PASS: …")` line.

### 2.1 Pre-stack foundation (not E-phases; required by E2+)

| Label | Commit(s) | Files landed | Witness suite | Count | Caveat |
|---|---|---|---|---:|---|
| v0 spec | `876e28b9e` | `docs/research/eisa-v0-spec-2026-07-05.md` | — | 0 | Textual `.eisa` grammar versioned, not frozen. |
| Reference core | `7eff69256` | `stdlib/eisa/core.sio`, `tests/stdlib/eisa/test_eisa_core.sio` | `eisa core` **W1…W5** | 5 | `err` is formulaic dd64; cross-terms dropped in `emul` (honest in header). I1–I5 normative in spec; partial witness coverage until parser landed. |
| ISA seed | `bc52ff251` | `stdlib/eisa/isa.sio`, `tests/stdlib/eisa/test_eisa_isa.sio` | `eisa isa` **P1…P5** | 5 | In-memory straight-line interpreter; seed for E2 differential harness, not the shipping executor. Commit message says "v1a" — naming predates the V1a format phase. |

### 2.2 E0…E5 (v0 executable stack)

| Phase | Commit(s) | Files landed | Witness suite | Count | Caveat |
|---|---|---|---|---:|---|
| **E0** | `7673b36f0` | `docs/research/eisa-stack-architecture-2026-07-05.md`, offload log row | — | 0 | Direction doc + format freeze; offload-reviewed per log. |
| **E1** | `53177645e` | `stdlib/eisa/format.sio`, `stdlib/eisa/asm.sio`, `docs/audit/LEAN_SINGLE_NAN_SEMANTICS_2026-07-05.md`, tweaks to `stdlib/eisa/isa.sio`, `tests/stdlib/eisa/test_eisax_format.sio` | `eisax` **F1…F7** | 7 | Container is flat parallel arrays (checker workaround). v0 rejects opcodes 9–15. |
| **E2** | `72c06ae8b` | `stdlib/eisa/evm.sio`, `tests/stdlib/eisa/test_eisa_evm.sio` | `eisa evm` **V1…V5** | 5 | Depends on E1 + `bc52ff251` isa seed. Receipt **v=1** with `prog=` field. Public surface refuses unvalidated images. |
| **E3** | `9c8be5054` | `stdlib/eisa/backend.sio`, `tests/stdlib/eisa/test_eisa_backend.sio`, architecture doc as-built note | `eisa backend` **B1…B6** | 6 | Standalone stdlib driver, not wired into the real compiler driver. Move via `eadd(x,Z)` canonicalises `-0.0` → `+0.0` (arch §3.1b). |
| **E4** | `0edaea539` | `stdlib/eisa/bridge_x86.sio`, `scripts/ci/eisa_bridge_conformance_gate.sh`, `tools/eisa/eisa_evm_run.sio`, `tools/eisa/eisa_bridge_emit.sio`, `tests/stdlib/eisa/test_eisa_bridge.sio` | `eisa bridge` **X1…X5** | 5 | Gate initially **4** differential programs + tamper + anti-vacuity. Bridge u-lane poison test uses sign bit (differs from EVM for `u == -0.0`; latent, v0 corpus never hits it). |
| **E5** | `09e3da9f1`, `accf146d3` | `examples/eisa_cancellation_kernel.sio`, `tests/stdlib/eisa/test_eisa_e5_kernel.sio`; gate + emit/run drivers extended for `e5-cancellation` | `eisa e5` **K1…K5** | 5 | Kernel is reduced-quartic cancellation at `x = 1+1e-6`, not full Rump 1988 (arch §6). E5 example commit landed **before** E4 bridge commit; gate closure is `accf146d3`. |

**E5 gate corpus at ship (`accf146d3`):** 5 differential programs
(`golden-mul`, `golden-add`, `golden-sqrt`, `golden-poison`, `e5-cancellation`)
plus tamper-sensitivity and anti-vacuity lanes.

### 2.3 V1a…V1e (Metron control flow + budgets)

| Phase | Commit(s) | Files landed | Witness suite | Count | Caveat |
|---|---|---|---|---:|---|
| **V1a** | `ad61af54c` | `stdlib/eisa/format.sio` (version word 1, fuel word, ops 9–13, budget lift), `tests/stdlib/eisa/test_eisax_v1_format.sio` | `eisax v1` **F1…F8** | 8 | v0/v1 cross-reject witnesses. Fuel word at header word 6; hashed. |
| **V1b** | `6f5d9dc1a` | `stdlib/eisa/evm.sio`, `tests/stdlib/eisa/test_eisa_evm_v1.sio` | `eisa evm v1` **U1…U8** | 8 | Receipt **v=2** (`frail=`, `stop=`). I6 sticky `br_poisoned`. u-lane `-0.0` normalised on write. |
| **V1c** | `7af4d08c7` | `stdlib/eisa/backend.sio`, `tests/stdlib/eisa/test_eisa_backend_v1.sio` | `eisa backend v1` **C1…C8** | 8 | Metron surface: `while`/`if`/`set`, block scoping, fuel synthesis. **Str cap 256 bytes** → status **-18** (see §3). |
| **V1d** | `8cf3954ab` | `stdlib/eisa/bridge_x86.sio`, `tests/stdlib/eisa/test_eisa_bridge_v1.sio`, gate + emit/run drivers | `eisa bridge v1` **Y1…Y7** at ship | 7 | Gate adds 6 v1 programs (10 differential total at V1d). High-register templates incomplete — only low window fully covered until bridge-highreg closure. |
| **V1e** | `528531bbe` | `stdlib/eisa/bridge_x86.sio` (dynamic fuel-stop), showcase test, gate + drivers | `eisa v1e` **S1…S3** | 3 | **Rump showcase withdrawn** (see §3). Gate adds 3 `v1e-*` programs (13 differential total at V1e). |

**Metron naming (not a ship phase):** `092ab73c4` records operator decision in
`docs/research/eisa-v1-plan-2026-07-05.md` §1 and `.claude/decisions.md` §14;
internal identifiers unchanged.

**V1 plan doc:** `3b243ae2c` (plan), `850494905` (frail exact-operand exclusion).

### 2.4 W1…W3 (v2 substrate landed in this worktree)

| Phase | Commit(s) | Files landed | Witness suite | Count | Caveat |
|---|---|---|---|---:|---|
| **W1** | `ad277b26e` | `stdlib/math/qd128.sio`, `tests/stdlib/math/test_qd128_core.sio`, `tests/stdlib/math/test_qd128_rump.sio` | `qd128 core` **Q1…Q8**; `qd128 Rump` **R1…R4** | 8 + 4 | Priest renormalisation sufficiency ≤51 overlap bits assumed (HLB); not proved. |
| **W2** | `15dd4ce3c` | `docs/research/eisa-v2-arch-2026-07-05.md` | — | 0 | Architecture draft; v2 closure contract defined. |
| **W3** | `20aa83eca` | `stdlib/eisa/core_v2.sio`, `stdlib/eisa/format.sio` (v2 dispatch), `stdlib/eisa/evm.sio` (v2 paths), `tests/stdlib/eisa/test_eisa_evm_v2.sio`, `tests/stdlib/eisa/test_eisax_v1_format.sio` (+regression pins) | `eisa evm v2` **W-A…W-H** | 8 | Receipt **v=3** (`roundoff1..3=`). **No bridge v2** in this commit — differential gate still v0/v1 images only. |

### 2.5 Bridge high-register closure (post-V1e, pre-v2 bridge)

| Label | Commit(s) | Files landed | Witness suite | Count | Caveat |
|---|---|---|---|---:|---|
| **bridge-highreg** | `5cf18e8af` (merged `622e199ae`) | `stdlib/eisa/bridge_x86.sio` (e16..e63 templates), extended `tests/stdlib/eisa/test_eisa_bridge_v1.sio`, gate + emit/run drivers | `eisa bridge v1` **Y1…Y9** (Y9 = arith-high + fuel-high + branch-high sub-lanes) | 9 | Gate adds 3 programs (`v1-arith-high`, `v1-fuel-high`, `v1-branch-high`) → **16** differential programs. v1 images capped by 65536-byte emission buffer (bridge header §v1 caveats). Fuel-stop `last_written` snapshot discipline documented in `bridge_x86.sio`. |

---

## 3. Deviations from plan

### 3.1 V1e: S1–S3 without Rump (rolled back)

**Planned** (`eisa-v1-plan-2026-07-05.md` §9 V1e row): full Rump 1988 under v1
budgets with dd64 honest-boundary framing, byte-identical EVM/bridge err lanes.

**Shipped:** three hand-lowered showcase images only — fixed-point loop (**S1**),
frail-cancellation (**S2**), `emov` `-0.0` bit-faithfulness (**S3**). The drafted
Rump bridge lane segfaulted on unfinished high-register templates and was rolled
back rather than shipped fragile.

**Recorded in:** `.claude/decisions.md` §15;
`tests/stdlib/eisa/test_eisa_v1e_showcase.sio` lines 134–140 (withdrawal note);
`docs/research/eisa-v2-arch-2026-07-05.md` §6 (`v2-rump-qd` supersedes the
planned V1e Rump lane on qd128).

### 3.2 Metron surface Str 256-byte cap (status -18)

**Planned:** v1 surface growth without a hard source-size ceiling (plan §10
defers "growable / chunked source input").

**Shipped:** `stdlib/eisa/backend.sio` documents status **-18** when parsing
ends structurally unclosed **and** `src.len >= 256` (fixed `Str` buffer in
`stdlib/str/lib.sio` truncates silently). V1c shipped the tripwire; larger
kernels must be hand-lowered via `EisaxBuild` (as V1e showcase does).

**Recorded in:** `eisa-v1-plan-2026-07-05.md` §10; `backend.sio` status convention
comment block; `tests/stdlib/eisa/test_eisa_backend_v1.sio` header (C3 leak witness
fits under cap; >64-reg leak witness unrepresentable).

### 3.3 W-H anchor-limit restructuring (v2 EVM witness)

**Planned** (`eisa-v2-arch-2026-07-05.md` §6 `v2-rump-qd`): Rump at version 2
with val+err reconstruction pinning `-54767/66192`.

**Shipped (W3):** witness **W-H** hand-lowers Rump on the v2 EVM. The final
register's val debris is `-2^70`, so a **single-register** val+err reconstruction
cannot reach truth's 4th component (~2^-163) within qd128 span — arithmetically
impossible (measured span 70+163+53 > 212 bits). The witness was restructured:
exact value is receipt-recoverable **bit-identically from two gated source
registers** (s2 with true = −2, t4 with true = round_qd(77617/66192)); both
forms asserted in `test_eisa_evm_v2.sio`.

**Recorded in:** `tests/stdlib/eisa/test_eisa_evm_v2.sio` lines 14–24 (W-H honesty
note); v2-arch §6 still describes the single-register framing — treat the test
header as the as-built correction until v2 docs are updated elsewhere.

### 3.4 Other honest gaps (not rollbacks)

| Item | Plan | As-built | Where recorded |
|---|---|---|---|
| E5 kernel | PK dose step or Rump | Reduced-quartic cancellation | `eisa-stack-architecture-2026-07-05.md` §6 |
| E3 backend in driver | eventual compiler integration | standalone stdlib driver | `eisa-v1-plan-2026-07-05.md` §10; `backend.sio` |
| v2 bridge + gate v2 lanes | W4 in v2-arch §7 | **Not landed** in commits through `5cf18e8af` | `eisa-v2-arch-2026-07-05.md` §7 |
| Subnormal-injective receipts | deferred v2 | still deferred | v1-plan §4; v2-arch §8 |

---

## 4. Verification commands

All EISA witnesses in this worktree declare `validated_lane: lean_single`.
Run from repository root `/workspace/sounio-eisa` with stdlib on the path.

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC=./bin/souc
```

### 4.1 Per-suite (mirrors test headers)

```bash
# v0 reference + container + VM + backend + bridge + E5
$SOUC run tests/stdlib/eisa/test_eisa_core.sio
$SOUC run tests/stdlib/eisa/test_eisa_isa.sio
$SOUC run tests/stdlib/eisa/test_eisax_format.sio
$SOUC run tests/stdlib/eisa/test_eisa_evm.sio
$SOUC run tests/stdlib/eisa/test_eisa_backend.sio
$SOUC run tests/stdlib/eisa/test_eisa_bridge.sio
$SOUC run tests/stdlib/eisa/test_eisa_e5_kernel.sio

# v1 format + Metron VM + Metron surface + bridge v1 + showcase
$SOUC run tests/stdlib/eisa/test_eisax_v1_format.sio
$SOUC run tests/stdlib/eisa/test_eisa_evm_v1.sio
$SOUC run tests/stdlib/eisa/test_eisa_backend_v1.sio
$SOUC run tests/stdlib/eisa/test_eisa_bridge_v1.sio
$SOUC run tests/stdlib/eisa/test_eisa_v1e_showcase.sio

# v2 substrate (W1 math + W3 Metron VM v2)
$SOUC run tests/stdlib/math/test_qd128_core.sio
$SOUC run tests/stdlib/math/test_qd128_rump.sio
$SOUC run tests/stdlib/eisa/test_eisa_evm_v2.sio
```

Each command expects exit code 0 and a final `ALL PASS: …` line (receipt lines
may precede it on VM tests).

### 4.2 Bridge conformance gate (byte-identical EVM vs AOT x86-64)

The gate itself sets `SOUNIO_SOUC_ENGINE=lean_single` internally:

```bash
bash scripts/ci/eisa_bridge_conformance_gate.sh
```

Expect `PASS` for each differential program, `PASS tamper-sensitivity`,
`PASS anti-vacuity`, and final `PASS eisa_bridge_conformance`.

### 4.3 Slurm battery (workspace-sparing full run)

From `slurm-jobs/eisa/submit-eisa-battery.sh` (topology measured 2026-07-06):

```bash
bash slurm-jobs/eisa/submit-eisa-battery.sh
# later:
bash slurm-jobs/eisa/submit-eisa-battery.sh <run-id>
```

The batch job exports `SOUNIO_STDLIB_PATH`, `SOUNIO_SOUC_ENGINE=lean_single`,
runs all `tests/stdlib/eisa/*.sio` plus `test_qd128_core.sio`,
`test_qd128_rump.sio`, `test_dd64_cancellation.sio`, `test_dd64_eft_exact.sio`,
`test_dd64_algebra.sio`, then the conformance gate above.

**Execution note:** this chronicle was assembled by git/file inspection. A
battery run on the same tree (run id `eisa-battery-20260706T030444`, node
`gpuorangefs-5860-proxmox`) confirmed 18/18 test suites PASS and 17/18 gate
lanes PASS — the sole FAIL is the anti-vacuity lane, an environmental
`strings`-missing artefact on the compute node, not a semantic failure
(see `docs/audit/CI_GATE_PORTABILITY_2026-07-06.md`).

---

## 5. Current lane and witness counts (HEAD)

### 5.1 Conformance gate (`scripts/ci/eisa_bridge_conformance_gate.sh`)

| Lane class | Count | Names / behaviour |
|---|---:|---|
| Differential corpus programs | **16** | `golden-mul`, `golden-add`, `golden-sqrt`, `golden-poison`, `e5-cancellation`, `v1-loop`, `v1-if-both`, `v1-i6`, `v1-fuel`, `v1-highreg`, `v1e-fixedpoint`, `v1e-frail`, `v1e-emov-negzero`, `v1-arith-high`, `v1-fuel-high`, `v1-branch-high` |
| Tamper-sensitivity | **1** | `golden-mul-tampered.eisax.elf` must differ from untouched EVM stdout |
| Anti-vacuity | **1** | over all 16 programs: receipt label prefix present in ELF; mantissa digit runs absent |
| **Total gate lanes** | **18** | 16 + 1 + 1 |

Receipt prefix expectation in gate: `v=1 prog=` for v0 programs; `v=2 prog=` for
`v1-*` and `v1e-*` programs (anti-vacuity case statement).

### 5.2 Test suites (`tests/stdlib/eisa/`)

| File | Witness IDs | Count |
|---|---|---:|
| `test_eisa_core.sio` | W1…W5 | 5 |
| `test_eisa_isa.sio` | P1…P5 | 5 |
| `test_eisax_format.sio` | F1…F7 | 7 |
| `test_eisax_v1_format.sio` | F1…F8 | 8 |
| `test_eisa_evm.sio` | V1…V5 | 5 |
| `test_eisa_evm_v1.sio` | U1…U8 | 8 |
| `test_eisa_evm_v2.sio` | W-A…W-H | 8 |
| `test_eisa_backend.sio` | B1…B6 | 6 |
| `test_eisa_backend_v1.sio` | C1…C8 | 8 |
| `test_eisa_bridge.sio` | X1…X5 | 5 |
| `test_eisa_bridge_v1.sio` | Y1…Y9 | 9 |
| `test_eisa_e5_kernel.sio` | K1…K5 | 5 |
| `test_eisa_v1e_showcase.sio` | S1…S3 | 3 |
| **EISA subtotal** | | **82** |

### 5.3 Math suites in the Slurm battery (W1 substrate + dd64 boundary)

| File | Witness IDs | Count |
|---|---|---:|
| `tests/stdlib/math/test_qd128_core.sio` | Q1…Q8 | 8 |
| `tests/stdlib/math/test_qd128_rump.sio` | R1…R4 | 4 |
| `tests/stdlib/math/test_dd64_cancellation.sio` | (descriptive; no letter inventory) | unverified |
| `tests/stdlib/math/test_dd64_eft_exact.sio` | — | unverified |
| `tests/stdlib/math/test_dd64_algebra.sio` | — | unverified |

### 5.4 v2 cross-reference — what v2 supersedes (not yet fully shipped)

From `eisa-v2-arch-2026-07-05.md`:

- **Err lane depth:** dd64 formulaic propagation → qd128 closure contract
  (`core_v2.sio`, W3 landed).
- **Format/receipt:** version word 2, receipt v3 (W3 landed).
- **Rump showcase:** moved from planned V1e to v2 (`W-H` on EVM; bridge v2 + gate
  v2 lanes are **W4**, not in git through bridge-highreg).
- **Mandatory dd64-failure lane** alongside qd128 success (decisions §15; planned
  as `v2-rump-dd` in v2-arch §6 — **not in conformance gate** at HEAD).

---

## 6. Module map (header caveats, as of HEAD)

| Module | Role | Honest caveat (from file header) |
|---|---|---|
| `core.sio` | v0/v1 dd64 reference oracle | Second-order err terms in mul/div/sqrt; frozen as v0 oracle when v2 exists. |
| `core_v2.sio` | v2 qd128 reference oracle | Priest renorm assumption; failure observable at harness level, not silently proven. |
| `format.sio` | `.eisax` encode/decode/validate/hash | v2 = same layout as v1; version word alone selects semantics. |
| `isa.sio` | In-memory differential seed | Dual-semantics NaN finiteness detector for lean_single vs IEEE. |
| `evm.sio` | Metron VM | v2 widens err storage into reg_ehi…reg_e3 as Qd128 components; v0/v1 paths byte-untouched. |
| `backend.sio` | Metron surface compiler | Status -18 Str cap; not in real compiler driver. |
| `asm.sio` | `.eisa` text → container | Shares Str limitations via `str::lib`. |
| `bridge_x86.sio` | AOT conformance bridge | v0 u-lane sign-bit poison test; v1 highreg state layout; 65536-byte emit cap. |

---

*Branch: `gpu/epistemic-tensor-core-next`. Worktree: `/workspace/sounio-eisa`.
Last commit inspected for gate counts: `622e199ae` (merge of bridge-highreg).*
