<!-- docs:meta
topic_id: repo.docs.audit.enir-mir-disconnect-cost-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.enir-mir-disconnect-cost-2026-08-19
-->

# ENIR / MIR / MLI disconnect cost — measurement only

**Date:** 2026-08-19  
**sha:** `515d93a8e3`  
**Host (souc check):** Slurm `cpu-ops` / `cpuops-t560-proxmox`  
**Host (git / full-tree imports):** login worktree (full `.git` + full `self-hosted/`)  
**Not done this round:** any rewiring, any `build_modular_madaros`, any production path change.

Receipts: `docs/audit/enir_mir_disconnect/`  
Per-file check TSV: `docs/audit/ENIR_MIR_DISCONNECT_COST_2026-08-19.tsv`  
Script: `scripts/dev/enir_mir_disconnect_measure.sh`

---

## Semantic lane declaration

```text
Semantic-Lane-ID: enir-mir-cost-20260819
Owner: grok-cli4
Concept-IDs: none
Intent-Preserved: live pipeline remains parser→check→ir→native; this lane only costs the dark layers
Transformation: none — measurement
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - enir/ and mli/ both typecheck file-by-file under Madaros (31/31 rc=0)
  - both have zero external importers (use X:: outside X/)
  - enir MIR and mli are concurrent designs for different IR families, not stages of one pipe
  - enir was shadow-executable via driver.sio + E* gates, never a production importer
  - EMIR_MAX_INSTRS=128 is one-block fixture scale; MLI_MAX_INSTRS=32 is per-block body
  - shortest existing link to ir/ is mli/ir_to_mli.sio; enir has none
Claims-Forbidden:
  - "enir is dead source" (it checks clean)
  - "mli is production machine IR" (machine_ir.sio holds that slot)
  - aggregate check green as proof of liveness
  - rewiring cost or schedule (not measured; founder decides)
Assumptions: import instrument = `use <dir>::` with external = path outside dir/
Write-Set: docs/audit/ENIR_MIR_DISCONNECT_COST_*, docs/audit/enir_mir_disconnect/**, scripts/dev/enir_mir_disconnect_measure.sh
Read-Set: self-hosted/{enir,mli,ir,native,compiler}, scripts/dev/madaros_v2_e*, .github/workflows/ci.yml
Positive-Witness: parser/ast.sio souc check rc=0 on Slurm; parser external≈93 full tree
Negative-Witness: enir external=0, mli external=0
Acceptance-Gate: re-run measure on Slurm; per-file TSV + receipt
Integration-Target: none this round
Authoritative-Only-If: receipt host=cpuops-* and pos_ctrl_check_OK
```

---

## Refutation criteria (written before the run)

Full text: `docs/audit/enir_mir_disconnect/REFUTATION_CRITERIA.txt`

| ID | If we see this… | Then… |
|---|---|---|
| R1 | enir majority fails `souc check`, or caps cannot hold real functions **and** no standalone driver | not recoverable without rewrite |
| R2 | mli has no `use ir::` bridge, or production already owns the slot via machine_ir with mli external=0 | mli is not the live machine IR |
| R3 | consume→produce sets are disjoint and both claim “after ir, before native” | concurrent designs (founder choice), not stages |
| R4 | no gate ever built driver.sio | never executable; else shadow-only |
| R5 | import instrument fails positive control (parser ext≪80 or ir ext≪40) | discard import zeros |
| R6 | only an aggregate check result is offered | ruler fail |

---

## Instrument validation (R5)

**Import syntax:** `use <dir>::` only. External = file path not under `self-hosted/<dir>/`.

| dir | external (full tree) | internal | founder band |
|---|---:|---:|---|
| parser | **94** | 7 | ~93 ✓ |
| check | **49** | 6 | ~49 ✓ |
| ir | **50** | 21 | ~48 ✓ |
| native | **21** | 36 | ~19 ✓ |
| wasm | 17 | 8 | ~15 |
| hlir | **2** | 1 | 2 ✓ |
| gpu | 2 | 21 | ~1 |
| **enir** | **0** | 10 | 0 ✓ |
| **mli** | **0** | 16 | 0 ✓ |
| llvm | 0 | 0 | 0 |
| vm | 0 | 0 | 0 |
| effects | 0 | 0 | 0 |

Positive control **PASS**. Bad instrument (`use|import|from|mod`) hits hundreds of files including the word “model” — not used.

**souc check positive control (Slurm):** `self-hosted/parser/ast.sio` rc=0 in 0.226s.

---

## Bytes (dark layers)

| dir | files | bytes | KB |
|---|---:|---:|---:|
| enir/ | 14 | 410 469 | 400.8 |
| mli/ | 17 | 266 864 | 260.6 |
| hlir/ | 5 | 188 947 | 184.5 |
| effects/ | 4 | 183 498 | 179.2 |
| **sum dark-ish** | | | **~1025 KB** of designed surface with ≤2 external importers each |

(Founder’s 1345 KB band includes further projected layers; this table is measured `*.sio` only.)

---

## Q1 — `souc check` per file (no aggregate as the answer)

Engine: Madaros via `bin/souc` → `bin/madaros-linux-x86_64` on Slurm.  
Full table: `docs/audit/ENIR_MIR_DISCONNECT_COST_2026-08-19.tsv`

### enir/ (14/14 rc=0)

| path | rc | bytes | s |
|---|---:|---:|---:|
| enir/canonical.sio | 0 | 7625 | 0.167 |
| enir/driver.sio | 0 | 19258 | 0.537 |
| enir/hash.sio | 0 | 516 | 0.150 |
| enir/interpreter.sio | 0 | 36252 | 0.260 |
| enir/ir.sio | 0 | 14985 | 0.158 |
| enir/mir.sio | 0 | 46814 | 0.273 |
| enir/mir_cfg.sio | 0 | 63643 | 0.328 |
| enir/mir_join.sio | 0 | 68807 | 0.342 |
| enir/mod.sio | 0 | 309 | 0.506 |
| enir/parser.sio | 0 | 22278 | 0.183 |
| enir/qd.sio | 0 | 13348 | 0.175 |
| enir/shadow_fixture.sio | 0 | 8727 | 0.162 |
| enir/source_lower.sio | 0 | 69003 | 0.269 |
| enir/verify.sio | 0 | 38904 | 0.195 |

### mli/ (17/17 rc=0)

| path | rc | bytes | s |
|---|---:|---:|---:|
| mli/aggregate_store_diag.sio | 0 | 2643 | 0.156 |
| mli/builder.sio | 0 | 7891 | 0.165 |
| mli/dump.sio | 0 | 9436 | 0.177 |
| mli/expand_cd.sio | 0 | 12478 | 0.175 |
| mli/expand_k.sio | 0 | 20791 | 0.190 |
| mli/inst_landing_diag.sio | 0 | 4173 | 0.428 |
| mli/interp.sio | 0 | 19734 | 0.179 |
| mli/ir.sio | 0 | 23195 | 0.160 |
| mli/ir_to_mli.sio | 0 | 24955 | 0.376 |
| mli/legalize_x86.sio | 0 | 25959 | 0.186 |
| mli/s2a_gate_runner.sio | 0 | 26986 | 0.445 |
| mli/s2b_gate_runner.sio | 0 | 15464 | 0.428 |
| mli/s3_emit_runner.sio | 0 | 2475 | 0.442 |
| mli/s3_gate_runner.sio | 0 | 15315 | 0.506 |
| mli/s3b_gate_runner.sio | 0 | 10654 | 0.259 |
| mli/self_test_runner.sio | 0 | 16929 | 0.231 |
| mli/verify.sio | 0 | 27786 | 0.206 |

**Reading:** “compiles” (typechecks) ≠ “lives on the pipeline”. Both packages are **source-healthy and import-dark**.

R1(a) does **not** fire: source is not rotten.

---

## Q2 — Two concurrent designs, or two stages of one pipe?

Decided by **consume → produce**, not by the shared word “MIR”.

| | **enir/** (incl. mir.sio / mir_cfg / mir_join) | **mli/** | **native/machine_ir.sio** (live) |
|---|---|---|---|
| **Consumes** | only `enir::*` (self). **No `use ir::`.** source_lower from EISA/qd text, not production IR | **`use ir::ir::*`** in `ir_to_mli.sio`; internal mli::* | production `ir` + parser; used by codegen / main |
| **Produces** | `EnirModule`, `EnirMirModule` (semantic/translation-validated MIR for EISA/qd128), interpreter receipts, canonical ENIR text | `MliFunction`, legalize_x86 forms, gate runners | `MachineFunction` → x86-64 ELF path |
| **Stated role** | “Native compiler-owned ENIR driver for E1 **shadow**… separate from production codegen” (`driver.sio:1–2`) | “Machine-Level IR… Option C” side door after production IR (`ir.sio` banner; MLI_DESIGN) | production native-v2 substrate |
| **External importers** | **0** | **0** | many (`compiler/main`, `codegen*`, `module_loader`, …) |

**Verdict (R3):**

1. **enir MIR ≠ mli ≠ machine_ir** as layers of one oleoduct.  
2. **enir** is a **parallel shadow pipeline** (EISA/epistemic-numeric IR → its own MIR → interpret/verify). It does not sit between `ir/` and `native/`.  
3. **mli** is a **competing design for the post-`ir` machine-ish slot** (explicit `ir_to_mli`, `legalize_x86`), but the **live** occupant of that slot is **`native/machine_ir.sio`**.  
4. Therefore enir vs mli are **not** “two stages of the same pipe”. enir vs machine_ir are different IR families. **mli vs machine_ir are concurrent candidates for the same job** (after production IR, before/as native). Choosing between mli and machine_ir is a **founder decision**; this measurement only shows both cannot be “the” layer without a merge or a kill.

---

## Q3 — Was enir ever actually executable?

| evidence | fact |
|---|---|
| Last (and landing) touch of `self-hosted/enir/` on main | `8999e0fdff` 2026-08-16 — **WS-C PR1: ENIR/MIR shadow lane (#1753)** |
| External `use enir::` ever on mainline callers | **none today**; package is self-import only |
| Driver | `enir/driver.sio` implements CLI verbs: `emit`, `verify`, `roundtrip`, `lower`, `run`, `*_mir` |
| Who builds the driver | **≥12 gate scripts** `scripts/dev/madaros_v2_e{1,2*,3*}_*.sh` all  
  `souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER"` then invoke `"$DRIVER" emit|verify|…` |
| CI | `.github/workflows/ci.yml` runs the E1–E3 enir gate set **when** `self-hosted/enir` or `tools/eisa` diffs |

**Verdict (R4):**

- **Yes, shadow-executable:** driver was designed to be compiled as a standalone program and driven by E* gates (emit fixture, roundtrip, verify, lower, run MIR). That is “executable in fact” for the shadow lane.  
- **No, production-executable:** zero importers from `compiler/main.sio` / `ir/` / `native/`. Comment on `mod.sio`: “Production codegen does not import this lane.”  
- Cost implication: recovery is not “fix bitrot”; it is **wiring a live edge** (and deciding whether EISA-ENIR belongs on the default path at all).

---

## Q4 — `EMIR_MAX_INSTRS = 128` — toy ceiling or per-block?

| constant | value | scope (from source) |
|---|---:|---|
| `IR_MAX_INSTRS` | **16384** | production `ir::` function arena; real tests need ~14389 |
| `native MIR_MAX_INSTRS` | **4096** | live machine_ir; `MIR_MAX_BLOCKS = 1` historically single-block materialise |
| `HLIR_MAX_INSTRS` | 16384 | hlir (also nearly dark) |
| **`EMIR_MAX_INSTRS`** | **128** | `enir/mir.sio:13`; arrays `[EnirMirInstr; 128]` |
| `EMIR_MAX_VALUES` | 128 | same module |
| **`MLI_MAX_INSTRS`** | **32** | **per-block body** (`mli/ir.sio:50`); `MLI_BLOCK_STRIDE = 33` (32 body + 1 terminator); `MLI_MAX_BLOCKS = 16` → pool 528 |

**ENIR comment (`mir.sio:1–2`):**

> ABI-independent semantic MIR for translation-validated ENIR lowering.  
> **E3A admits only one-block** qd128 constants, arithmetic, observations, and halt.

Lowering guard (`mir.sio:219–221`) **rejects** `source.block_count != 1`.

**Verdict:**

- **128 is a whole-module (single-block) hard ceiling**, not “128 per block of a large CFG”.  
- Relative to production IR (16k) and even live machine_ir (4k), **128 is fixture/oracle scale** — consistent with shadow E3A qd128 strips, not `main.sio`.  
- That is **evidence the ENIR MIR path never carried production-sized functions**, not proof the *idea* is impossible (a redesign could raise the cap). Under R1(c): **recovering ENIR MIR onto production bodies is a capacity + multi-block redesign, not a one-line constant bump**, because the data model is one `EnirMirBlock` plus fixed `[;128]` arrays.  
- **MLI’s 32 is per-block** with multi-block pool — smaller per block but intentionally multi-block; still far below production IR body sizes without many blocks / further growth. `expand_cd.sio` notes Hamilton product “Fits MLI_MAX_INSTRS=32 when…”.

---

## Q5 — Shortest link to `ir/` that exists today

| from | edge to `ir/` | exists? |
|---|---|---|
| **mli/** | `self-hosted/mli/ir_to_mli.sio` → `use ir::ir::*` (IR arena columns → MliFunction) | **YES — shortest existing link** |
| **enir/** | any `use ir::` | **NO** |
| **enir/** | production lowerer calling enir | **NO** |
| **machine_ir** | lowers from production IR (live) | YES — already the production edge |

**Shortest path if the goal is “something after ir/” without inventing edges:**  
`ir/` --(exists)--> `mli/ir_to_mli.sio` --(dark)--> `mli/legalize_x86.sio` --(**missing**)--> native/codegen  

vs production today:  
`ir/` --(exists)--> `native/machine_ir.sio` --(exists)--> codegen/ELF  

**enir** has **no** edge to `ir/`. Connecting it is a **new** edge (or a new lowerer from AST/EISA into the live pipe), not a re-enable.

---

## Cost reading (no schedule)

| package | source health | pipeline liveness | nature of cost to “make live” |
|---|---|---|---|
| enir/ + its MIR | 14/14 check | external importers 0; shadow gates only | **Product decision** + new edge into/out of live pipe + capacity redesign for non-toy bodies |
| mli/ | 17/17 check | external 0; has ir_to_mli | **Compete with or replace machine_ir**; wire legalize→codegen; raise caps for real functions |
| machine_ir | (not re-audited file-by-file here) | **live** | baseline — do not pay twice |

This round does **not** estimate engineer-weeks. It states what kind of work each path is.

---

## Reproduce

```bash
# Slurm (souc check + subset imports)
bash scripts/dev/enir_mir_disconnect_measure.sh   # on a tree the job can read
# or stream payload as in this lane's srun recipe

# Full-tree imports (login)
for d in parser check ir native enir mli; do
  echo -n "$d external="; grep -RIl --include='*.sio' "use ${d}::" self-hosted \
    | grep -vc "self-hosted/${d}/"
done
```
