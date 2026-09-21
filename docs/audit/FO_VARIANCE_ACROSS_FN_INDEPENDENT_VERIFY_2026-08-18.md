<!-- docs:meta
topic_id: repo.docs.audit.fo-variance-across-fn-independent-verify-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.fo-variance-across-fn-independent-verify-2026-08-18
-->

# Independent verify — first-order variance across a function boundary

**Date:** 2026-08-18
**Lane:** grok-cli5 / fo-call-boundary-20260818
**Question:** does Madaros lose first-order uncertainty when a non-zero variance crosses a function, and if so, at which site, on which engine, and since when?

**Short answer.** Yes, on this unpatched Madaros, and not only at the two thesis cells another lane named. Same-file 1-arg and 2-arg additive helpers still carry variance. Same-file 3-arg and 4-arg additive helpers print `0`. An imported 1-arg identity already prints `0`. lean_single keeps the additive matrix in both the same-file and the imported probe. The arity hole is the original `>2 params: skip` in `fo_register_pure_fn_transfer` (`990168abc2`, 2026-07-26). The import hole is a transfer table that never left the defining module; the compiler fix (`d9fe77669a`, 2026-07-28) is not an ancestor of `HEAD`. This is not a recent regression. Bisect will not find an introducing commit on `main` later than the birth of the FO transfer path.

The unconditional reading — *every* thesis uncertainty that crosses *any* function is underestimated — is false on today's Madaros. The conditional reading is not: every Madaros number that crosses a ≥3-argument helper, or any imported helper, has been able to die since this FO path existed.

**lean_single is the reference of this additive matrix. It is not a universal oracle.** `tests/run-pass/madaros_gum_fo_interproc.sio` is `PASS` on Madaros (`scale2=0.010000`) and `FAIL` on lean_single (`scale2=0.002500`). Do not invert the engines. Anyone who cites the matrix in §2–§3 must cite this limit with it.

---

## 0. What this receipt is, and is not

grok-cli2 reclassified Family A of `#1792` as compiler loss of first-order information at a function boundary, not an emitter bug. This hand built its own minimum repros and did not open `tests/run-pass/gum_fo_across_call.sio` or her adaptive / RK4 edits. The numbers below are from those probes.

Not in scope:

- Patching `self-hosted/ir/lower.sio`.
- Source-building Madaros. The prebuilt ELF here predates `#1882` and has `runtime_context_size=248` (no E230).
- Taking her dispatch files under `docs/audit/repro/fo_call_boundary_*.sio`.
- Recalculating the GUM sums in another language. Independent measurements of `u = 2, 1, 3, 0.5` have variances `4, 1, 9, 0.25`. For uncorrelated inputs, `var(a+b) = var(a)+var(b)` is JCGM 100:2008 §5.1.2. That is the expected column, not a novel formula.

Instrument: `souc compile <src> -o <elf>`, then `\x7fELF`, then execute. Bare `-o` is refused (`#1885`).

---

## 1. Binaries

| Surface | Path | sha256 | mtime (UTC) | bytes |
|---|---|---|---|---:|
| Madaros (default `bin/souc`) | `artifacts/self-hosted/madaros` | `05d95342e42b36d4ccb8b694b401df92c9918637e7a4cc6dcf4568cf424d9963` | 2026-08-17 15:32 | 99964760 |
| lean_single | `bin/souc-lean-single-x86_64` | `337d5a86f44ef9320a0485f181283df7d0662b944fe83ada3e536ca45ce48db7` | 2026-08-17 14:49 | 2555805 |

`./bin/souc --version` prints `Madaros v0.80.0`. Worktree `lane/grok-cli5/fo-call-boundary-20260818` at `665f412bd92e`, tracking `origin/main`.

---

## 2. Own same-file matrix

Source: `docs/audit/repro/fo_var_samefile.sio`.

`measure(value, uncertainty: u)` then `.value`, then `variance_of` on the peeled `f64`. Helpers are `id1`, `ret_only` (bare `return x`), `add2`, `add3`, `add4`. Expected column is JCGM independent sum of `u²`.

| Cell | Expected | lean_single | Madaros |
|---|---:|---:|---:|
| `LOCAL_a` (`u=2`) | 4.0 | 4.000000 | 4.000000 |
| `LOCAL_b` (`u=1`) | 1.0 | 1.000000 | 1.000000 |
| `LOCAL_c` (`u=3`) | 9.0 | 9.000000 | 9.000000 |
| `LOCAL_d` (`u=0.5`) | 0.25 | 0.250000 | 0.250000 |
| `ID1` | 4.0 | 4.000000 | 4.000000 |
| `RET_ONLY` | 4.0 | 4.000000 | 4.000000 |
| `ADD2` | 5.0 | 5.000000 | 5.000000 |
| `ADD3` | 14.0 | 14.000000 | **0.000000** |
| `ADD4` | 14.25 | 14.250000 | **0.000000** |

Both ELFs were `\x7fELF`. The July 28 witness `tests/run-pass/gum_cross_function.sio` (2-arg add + 2-arg scale) still prints `var(sum)=5`, `var(scaled)=16`, `PASS` on **both** engines. Family A is therefore **not** a regression of that fix.

A first probe that constructed `Knowledge { value, variance: 0.25 }` hit Madaros `E036` even with `ε=1.0`. That is a separate compile-time hole. The matrix above uses the July `measure()` shape, which compiles.

---

## 3. Own imported matrix

Sources: `docs/audit/repro/fo_var_callee.sio` + `docs/audit/repro/fo_var_import.sio`. Compile with cwd = `docs/audit/repro` so `use fo_var_callee` resolves. Same arithmetic as §2, now across a module.

| Cell | Expected | lean_single | Madaros |
|---|---:|---:|---:|
| `IMP_LOCAL_a` | 4.0 | 4.000000 | 4.000000 |
| `IMP_ID1` | 4.0 | 4.000000 | **0.000000** |
| `IMP_ADD2` | 5.0 | 5.000000 | **0.000000** |
| `IMP_ADD3` | 14.0 | 14.000000 | **0.000000** |

lean_single invocation that worked: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile fo_var_import.sio -o <elf>` from the directory that holds both files. Direct `bin/souc-lean-single-x86_64 compile …` treats `compile` as a source path (`source: compile -1 bytes`). The wrapper is the instrument.

Existing repo corroboration, not a substitute for §3:

- `tests/run-pass/madaros_gum_fo_import.sio` — Madaros `v_imp_mul=0`, `v_imp_div=0`, `v_imp_css=0`, `MADAROS_GUM_FO_IMPORT_FAIL`. Same-file peel of `a*b` on that driver is `0.250000` (correct GUM product). The import is the loss.
- `tests/run-pass/madaros_gum_fo_eight_param.sio` — Madaros `v5s=v5p=v8s=vw=0`, `FAIL`. lean_single keeps the 5-sum and 8-sum (`0.0125`, `0.02`) and still `FAIL`s the weighted 6-arg cell (`vw=0.0075` vs `0.0725`). High-arity *additive* FO is a lean_single capability; high-arity *scaled* FO is not claimed here.

---

## 4. Three sites

| Site | Same-file evidence | Imported evidence | Owner |
|---|---|---|---|
| Argument passage (1–2 params) | `ID1=4`, `ADD2=5` on both engines | Madaros `IMP_ID1=0`, `IMP_ADD2=0` | Not the ABI. Same-file 1–2 arg transfer works. |
| Return | `RET_ONLY=4` on both engines | (identity is already a return) | Not the hole. July 28 `let`/`return` unwrap still holds. |
| Arity ≥3, same file | Madaros `ADD3=ADD4=0`; lean keeps | Madaros already zero at 1-arg | `fo_register_pure_fn_transfer` collects at most two param names, then `// >2 params: skip (unsupported transfer) return lo` (`self-hosted/ir/lower.sio` ~8595–8614). Silent. |
| Module traversal | n/a | Madaros loses even 1-arg identity; lean keeps the additive cells | Transfer table is registered at `lower_fn` and “accumulates across the module”. Imported callees are a different module. The July 28 multi-mod fix is not on `main`. |

So the premise “the compiler loses first-order information at a function boundary” is true, and it splits into two owners:

1. **Arity cap** — same compilation unit, ≥3 parameters.
2. **Module wall** — any imported helper, including 1-arg identity.

It is **not** “any function”. A 2-arg `add` in the same file is the July witness, and it still passes.

---

## 5. How long?

Hope: a recent introducing commit, cheap bisect. That hope is false for both holes.

| Fact | Evidence |
|---|---|
| FO transfer is born with the ≥3 skip | `990168abc2d15843c464b7a944d51c67c8b90d2e` 2026-07-26 02:15 UTC, `fix(madaros): inter-procedural FO via pure-fn transfer table`. The same commit already has `// Collect up to 2 param names` and `// >2 params: skip (unsupported transfer)`. Ancestor of `HEAD`. |
| A ≤4-param lift exists and never landed | `4f8e7dcf237fc3de4f3849c34a2594bdff096810` 2026-07-26 02:51 UTC, `fix(madaros): FO bytecode for pure let-bodies and ≤4 params`. `git merge-base --is-ancestor 4f8e7dcf23 HEAD` → **not an ancestor**. Lives on `fix/fo-*` branches. |
| 2-arg `let`/`return` unwrap did land | `b069fd4143546ddfc7da0783801bd6ab2bf0556d` 2026-07-28 14:54 UTC. Ancestor of `HEAD`. `docs/audit/MADAROS_FO_CROSS_FN_RETURN_2026-07-28.md` is still `CLOSED FIXED` on today's binaries. |
| Imported FO fix never landed | `d9fe77669a8d9533cf2a0092c8377555876acaee` 2026-07-28 08:58 UTC, `fix(madaros): multi-mod FO for imported pure helpers`. On `fix/fo-multimod-import-20260728`. **Not** an ancestor of `HEAD`. The test `madaros_gum_fo_import.sio` was added later (`154450258d`, 2026-08-03) and still fails on this Madaros. |
| Before 2026-07-26 | Madaros had no inter-procedural FO transfer table. Every call was outside the FO path. |

There is nothing to bisect on `main` for “when did ≥3-arg FO start dying”. It never lived on `main`. The same is true of imported FO.

`stdlib/epistemic/fo.sio` (landed `93bec8acd0`, 2026-08-03) is the science surface that sits on both holes at once: `fo_css` is 5-arg, `fo_auc` is 4-arg, `fo_infusion_rate` is 3-arg, and every caller that `use epistemic::fo::…` is an imported call. Under this Madaros those helpers cannot carry FO. The file's own header says the engine is “compiler-injected” and that wrapping FO builtins as ordinary functions would destroy structure. The helpers were written *for* the transfer table. The table does not see them on `main`.

---

## 6. What this does to the thesis premise

Discard, with evidence:

- “Variance dies on *a* call” as a universal. Same-file `id1`, `ret_only`, and `add2` keep FO on both engines. `gum_cross_function` still `PASS`.

Do not discard:

- Any Madaros uncertainty that crosses a helper with three or more `f64` parameters.
- Any Madaros uncertainty that crosses an imported helper, including identity.
- The duration: this has been the shape of the FO path since 2026-07-26, not a late-August accident.

If a thesis number was produced under **lean_single** and only crossed same-file 1–2 arg *additive* helpers, this receipt does not impeach it. If it was produced under **Madaros** and crossed `epistemic::fo` or any other imported / ≥3-arg helper, the printed variance can be a silent zero. That is older than the two cells grok-cli2 named. It is not every number. It is every number on that side of the cut.

lean_single is the reference **for this additive matrix**. It is not a universal FO oracle: `tests/run-pass/madaros_gum_fo_interproc.sio` is `PASS` on Madaros (`scale2=0.010000`, `mul=0.032500`) and `FAIL` on lean_single (`scale2=0.002500`, `mul=0.005000`). Expression-body `2.0 * x` through a same-file helper is a different classification than `let r = x + y; return r`. Do not invert the engines globally.

---

## 7. Commands

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUC=./bin/souc

# same-file
SOUNIO_SOUC_ENGINE=lean_single $SOUC compile docs/audit/repro/fo_var_samefile.sio -o /tmp/fo_same.ls.elf
$SOUC compile docs/audit/repro/fo_var_samefile.sio -o /tmp/fo_same.md.elf

# imported (cwd must see fo_var_callee.sio)
( cd docs/audit/repro && SOUNIO_SOUC_ENGINE=lean_single $SOUC compile fo_var_import.sio -o /tmp/fo_imp.ls.elf )
( cd docs/audit/repro && $SOUC compile fo_var_import.sio -o /tmp/fo_imp.md.elf )
```

July control: `tests/run-pass/gum_cross_function.sio` — both engines `PASS`.

---

## 8. What was not done

- No patch. No Madaros rebuild. No E230.
- `gum_fo_across_call.sio` was not read.
- Thesis PBPK drivers were not re-run here. The cut in §6 is the premise those lanes should use; it is not a census of which dissertation figures sit on the wrong side.
- No LLM-offload: this receipt does not introduce a PK/PD formula or a clinical pathway. The expected column is textbook independent-sum variance.
