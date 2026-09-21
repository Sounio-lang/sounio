<!-- docs:meta
topic_id: repo.docs.audit.refinement-div-discharge-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: cursor-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.refinement-div-discharge-2026-08-19
-->

# Can refinement discharge a divisor≠0 obligation?

**Date:** 2026-08-19  
**SHA:** `92fade0be1a6700b583da235f64d2b344aa82f2f` (`origin/main`)  
**Compute:** Slurm `srun` via `scripts/dev/slurm_srun_minimal.sh` (`--partition=cpu-ops`, host `cpuops-t560-proxmox`). Both engines: default Madaros (`bin/madaros-linux-x86_64`, v0.80.0) and `bin/souc-lean-single-x86_64`.  
**Instrument:** `scripts/dev/refinement_div_discharge_probe.sh`  
**Witnesses:** `docs/audit/repro/refinement_div_discharge/`  
**This is a measurement.** `self-hosted/` was not edited. No split of `Div` is proposed.

**Sentence the founder asked for:** **DECLARA APENAS** — refinement checks handwritten predicates (and can prove some of them). It does not derive a divisor≠0 obligation from `/`, and it cannot discharge `Div` because `/` does not generate `Div`. `Div` has nowhere to go without new checker work.

## 1. What refinement checks today

`self-hosted/check/refinement.sio` is a **static predicate engine** (comment: "no Z3"). Callers in `check.sio` use it at **declared-type boundaries**, not at operators.

| site | what it does |
|---|---|
| `checker_lower_refinement_type_mut` | lowers `{ x: T \| pred }` into a `RefinementTable` slot |
| `checker_check_refinement_literal_inplace` | literal assigned to a refined type: `refine_static_check_int_literal` / float |
| `checker_check_call_arg_refinement_boundary_inplace` | call argument vs parameter predicate |
| `if_cond_extract_pred` + `refine_cond_env` | pushes an `if` comparison into a guard stack for the then-branch |

There is **no** call from `checker_check_binary_with_operand_types_inplace` into refinement. `OpDiv` only constant-folds an integer-zero denominator and reports **E056**. Indexing (`checker_check_index_inplace`) does not read a refinement on the index and does not mention `Panic`.

So: **declared** predicates, not **derived** operator obligations. Division, indexing, and unsigned subtraction are not refinement sources.

## 2. Solver or syntactic equality?

Not mere textual equality. `pred_implies` (`refinement.sio`) implements interval / comparison implication: `x > 5` implies `x > 0`; `x != 0` does not imply `x > 0`. Compound `and`/`or` are handled by a conservative structural walk. There is **no** SMT solver.

Path narrowing exists, but only for `if` (not `while`) and only when the condition is `ident <op> integer-literal` (or a boolean combination of those). `if d != 0.0` is **not** a guard: `if_cond_extract_pred` requires `ExprIntLit` on the RHS.

Unproven call arguments fall through to a runtime-checked path (comment: W040). On this SHA, `check` accepted those cells with **no W040 token** on stderr.

## 3. Decisive witnesses (both engines)

Machine table: [`REFINEMENT_DIV_DISCHARGE_2026-08-19.tsv`](REFINEMENT_DIV_DISCHARGE_2026-08-19.tsv).

### Div is propagated, not generated

| cell | question | Madaros | lean_single |
|---|---|---|---|
| A1 `a / d` on `f64`, no `with Div` | does `/` introduce Div? | **accept** | **accept** |
| A2 `a / d` on `i32`, no `with Div` | same | **accept** | **accept** |
| A3 caller of a callee that **declares** `with Div` | generic E035? | **E035** missing Div | **E035** |
| B1 `if d != 0.0 { a / d }`, no Div | dispatch float guard | **accept** | **accept** |
| B2 `if d != 0 { a / d }`, no Div | extractable int guard | **accept** | **accept** |
| C1 `denom: { d: i32 \| d != 0 }`, `num / denom`, no Div | refined NonZero | **accept** | **accept** |
| E1 `1 / 0` | constant zero | **E056** | reject (no E056 token) |

The negative control **also accepts**. Guarded and refined division accept **for the same reason** A1/A2 accept: nothing at the `/` leaf asserts Div. That is the Mut-before-#1488 shape (`tests/compile-fail/effect_mut_generated_at_excl_ref_store.sio`): effects are true only where a human wrote them, then E035 copies them up the call graph.

Mut **does** have a leaf now (`checker_check_assign_stmt_inplace`, `store_effects[0] = 1`). Div and Panic do not.

### Refinement itself is live (so the Div measurement is not "the checker is dead")

| cell | question | Madaros | lean_single |
|---|---|---|---|
| D1i `takes_pos(0)` inline `{ v: i32 \| v > 0 }` | rejects bad literal? | **E042** | **E042** |
| D2i `{ n > 5 }` passed to `{ n > 0 }` | implication? | **accept** | **accept** |
| D3i `{ n >= 0 }` passed to `{ n > 5 }` | converse? | **E042** | **E042** |
| D4 `if y != 0 { takes_nz(y) }` | path narrowing at a **call**? | **accept** | **accept** |
| D5 unguarded `takes_nz(y)` | unproven arg | **accept** (no W040 seen) | **accept** |

Named aliases (`type Pos = { v: i32 \| v > 0 }`) **diverge**: Madaros reports **E008** (`expected i32, found Pos`) on D1/D2/D3. lean_single treats the alias as a refined i32 (D2 accept, D1/D3 E042). The implication engine works on **inline** refinements on both engines. The `type NonZero = …` surface the Div-retirement story wants is **not** live on default Madaros.

### Panic is the same leaf gap

| cell | question | Madaros | lean_single |
|---|---|---|---|
| F1 `arr[i]` no `with Panic` | does index introduce Panic? | **accept** | **accept** |
| F2 refined `i: { k \| k >= 0 && k < 4 }`, no Panic | discharge? | **accept** | **accept** |
| F3 caller of a callee that **declares** Panic | E035? | **E035** missing Panic | **E035** |

A refined index does not waive Panic, because Panic was never required.

## 4. How much work if `Div` is to leave

Not a new solver. `pred_implies` already proves `d != 0` and `d > 0`. The missing pieces are **wiring**:

1. **Generate** Div at `OpDiv` / `OpRem` (copy the #1488 Mut store-site pattern into `checker_check_binary_with_operand_types_inplace`). Without this, "discharge" is undefined: there is no obligation.
2. **Discharge** that generated fact when the denominator's refinement or `refine_cond_env` proves `≠ 0` (or `> 0` or `< 0`). That is one `refine_static_check_var` / `pred_implies` consult at the binary, which does not exist today.
3. **Float guards:** extend `if_cond_extract_pred` past `ExprIntLit`, or the dispatch's `if d != 0.0` remains invisible.
4. **Named aliases on Madaros:** E008 on `type Pos = { … }` must be closed if the language surface is a named `NonZero`.
5. **Same pair for Panic** at `ExprIndex` (generate + prove `0 <= i < N`). Index does not consult refinement today.
6. Only **after** 1–2 would retiring `Div` from tens of thousands of signatures be a migration rather than a lie.

Step 1 alone without step 2 would *increase* `with Div` noise: every `/` would start requiring the effect that today is optional at the leaf.

## Verdict

| option | |
|---|---|
| DESCARREGA | no — `/` never becomes an obligation, so nothing is proved away |
| **DECLARA APENAS** | **yes** — handwritten predicates, with real implication and `if`-narrowing at call boundaries |
| NEM VERIFICA | no — D1i/D2i/D3i show the engine is not string equality |
| INDETERMINADO | no |

`Div` cannot be replaced by refinement on this SHA. The mechanism that would replace it has not been connected to `/`.
