<!-- docs:meta
topic_id: repo.docs.audit.vec-new-path-call-fabrication-census-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.vec-new-path-call-fabrication-census-2026-08-17
-->

# Vec::new path-call fabrication census

**Date:** 2026-08-17  
**Family:** same honesty class as #1788 / #1792 (fabricated zero) and #1784 (silent truncation)  
**Question:** Madaros accepts `Type::method()` when `Type` does not exist; the call evaluates to **0**. Concrete case: `Vec::new` — there is **no** `struct Vec` / `type Vec` in the repository (collections ship `IntVec` / `FloatVec` / `HeapVec` / `Native*Vec`). Enumerate active sites, intent, and live vs dead under source-built Madaros.

## Mechanism (measured on source-built Madaros)

```text
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
export MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros SOUNIO_STDLIB_PATH=$PWD/stdlib

# probe: bare Vec::new
./bin/souc check tests/run-pass/vec_new_nonexistent_type_eval_zero.sio
# → check: OK   (no W44, no E137)

./bin/souc run tests/run-pass/vec_new_nonexistent_type_eval_zero.sio
# → prints: 0
```

Same result on prebuilt `bin/madaros-linux-x86_64`.

**Engine split (CI, 2026-08-17):** the Full Test Suite job runs `souc-stage2` via `scripts/ci/souc-native-wrapper.sh` (lean_single). On that engine the same witness exits 1 with empty stdout — so `tests/run-pass/vec_new_nonexistent_type_eval_zero.sio` is `//@ known-failure` until both engines refuse, or the suite engine is Madaros and refuse has landed. The 19-site census above is a Madaros measurement and is unchanged.

### Checker path

In `self-hosted/check/check.sio` associated-function handling (`Type::method`):

1. Lookup `fn_sig_table_find_method_semantic` for the path.
2. On **miss**, fall through: the path callee types as a named/unknown type.
3. Call arm `is_error_or_unknown(callee_ty)` checks args and **returns that type** — no hard error.
4. Native lowering of the missing call becomes a zero-producing stub (same family as E219 extern stubs).

### W44 status — warning is not wired

`print_warning_message` code 44 text says:

> associated-function call on an unknown type… compiles and evaluates to 0

But a repo-wide search finds **no** `report_warning_at(..., 44, ...)` (or mut equivalent) for this case. Error code 44 is `break` outside loop. So the “W44 dispatch” is a **message table entry without an emitter**: the fabrication is **silent** on the measured path, not merely a soft warning.

## Inventory method

- Scan `stdlib/`, `examples/`, `tests/`, `self-hosted/` for bare `Vec::new` (not `IntVec::new` / `FloatVec::new` / …).
- Strip `//` line comments and `/* */` block comments (many examples hide Rust-dialect bodies in block comments and ship a stub `main`).
- **Active** = appears outside comments.

## Counts

| Scope | Active `Vec::new` sites |
|---|---:|
| `stdlib/` | **19** |
| `examples/` (outside comments) | **2** |
| `tests/` | **1** |
| `self-hosted/` | **0** |
| **Total active** | **22** |
| Inside `/* */` only (examples stubs) | 22 additional historical bodies — **not active** |

No `struct Vec` / generic `Vec<T>` definition exists under `stdlib/` or `self-hosted/`. Legitimate collections: `stdlib/collections/vec.sio` → `IntVec`, `FloatVec`; also `HeapVec`, `NativeI64Vec`, `NativeF64Vec`.

## Live vs dead under Madaros

**Live path (language):** any program that typechecks can call `Vec::new()` and get **0** at runtime. Witness: `tests/run-pass/vec_new_nonexistent_type_eval_zero.sio` (source-built Madaros).

**Dead as module body (current Madaros):** every **active** site sits in a host file that **does not** `check: OK` under source-built Madaros:

| File | Sites | Madaros module status | Why dead today |
|---|---:|---|---|
| `stdlib/epistemic/meta.sio` | 9 | fails check (~41 errors) | Rust-dialect (`&[T]`, `.len()`, `vec![]`, `.clone()`) |
| `stdlib/nn/optimizers_quaternion.sio` | 5 | fails check (~168 errors) | incomplete / non-Sounio surface |
| `stdlib/compiler/epistemic/confidence_metadata.sio` | 3 | **parse fail** | depends on unparseable sibling |
| `stdlib/qnn/training.sio` | 1 | **parse fail** | Rust-ish module |
| `stdlib/pbpk/regulatory.sio` | 1 | **parse fail** | parse errors early in file |
| `examples/csv/mod_demo.sio` | 2 | fails check (~21 errors) | Rust `String` / `Writer` dialect |
| `tests/test_rusty_sounio.sio` | 1 | fails check (intentional) | **deliberate anti-pattern sample** for lint harness |

So: **0 of 22 active sites ride a currently-green Madaros execution path inside their host module.** They are **debt** (and traps when those modules are ported), not today’s silent zeros in shipped green programs.

That does **not** make the language bug debt-only: the probe proves fabrication is available to any new green code.

## Site-by-site intent (stdlib 19)

### `stdlib/epistemic/meta.sio` (9) — meta-analysis empties

| Line | Function | Intent |
|---:|---|---|
| 162, 166 | `fixed_effects` | `k==0` early return: empty `Provenance.steps`, empty `weights: Vec<f64>` |
| 222 | `fixed_effects` | normal path: empty provenance step list on pooled `Epistemic` |
| 251, 255 | `random_effects` | same pattern for RE empty-k |
| 324 | `random_effects` | empty provenance steps on pooled result |
| 370, 376 | `bayesian_pool` | empty-k prior path: empty steps + empty weights |
| 420 | `bayesian_pool` | empty provenance steps on pooled result |

**Meant:** growable `Vec` of f64 weights and provenance step records.  
**Should become:** fixed arrays / `FloatVec` / explicit empty-slice constants once types exist — never a missing-type call.  
**Callers internal to file:** `pool` helpers call `fixed_effects` / `random_effects` / `bayesian_pool`; **no external green Madaros caller** found.

### `stdlib/nn/optimizers_quaternion.sio` (5) — empty quaternion buffers

| Line | Function | Intent |
|---:|---|---|
| 18 | `empty_qvec` | construct empty `Vec<Quaternion>` via wrapper struct init |
| 24 | `empty_qvecvec` | empty `Vec<Vec<Quaternion>>` |
| 438–439 | `create_empty_gradients` | empty `weight_grads` / `bias_grads` |
| 594 | `compute_gradients` | empty `bias_grads` in returned `QuaternionGradients` |

Comment in-file admits workaround: “Vec::new() only type-checks in struct init”.  
**Internal callers only** (`create_empty_gradients`, `compute_gradients`, layer helpers). Demo `examples/nn/optimizers_quaternion_demo.sio` does not import these empty helpers into a green path measured here.  
**Meant:** empty gradient / activation buffers for quaternion Adam.

### `stdlib/compiler/epistemic/confidence_metadata.sio` (3)

| Line | Function | Intent |
|---:|---|---|
| 178 | `TypeConfidenceMetadata::new` | empty `provenance` step list |
| 193 | `::certain` | same |
| 208 | `::annotated` | same |

**Meant:** empty inference-provenance log. **No external uses** of `TypeConfidenceMetadata` outside this file.

### `stdlib/qnn/training.sio` (1)

| Line | Function | Intent |
|---:|---|---|
| 191 | `train_epoch` | `var predictions: Vec<Quat> = Vec::new()` before a commented-out forward pass |

**Meant:** batch prediction buffer. Forward pass is still a stub comment — even if the file parsed, this is already a fabricated empty prediction list.

### `stdlib/pbpk/regulatory.sio` (1)

| Line | Function | Intent |
|---:|---|---|
| 619 | `generate_fda_report` | `var reasons = Vec::new()` then `reasons.push(format!(...))` for failed qualification criteria |

**Meant:** human-readable failure reasons for FDA-style report. **High severity if this module ever goes green** — a “not qualified” report with fabricated empty reasons is a regulatory lie.

### Non-stdlib active

| Site | Intent | Live? |
|---|---|---|
| `examples/csv/mod_demo.sio:65,70` | CSV header/row `Vec<String>` for writer demo | Module fails check — dead under Madaros |
| `tests/test_rusty_sounio.sio:10` | **Intentional bad code** labeled “Using Vec (should be fixed array)” for Rust-ism linter | Dead by design; documents the anti-pattern |

## Severity classification

| Class | n | Meaning |
|---|---:|---|
| **Language fabrication (live)** | 1 mechanism | `Vec::new` → 0 with check OK; W44 not emitted |
| **Active stdlib debt (dead module)** | 19 | Will become fabricated zeros the day the host file typechecks without fixing the call |
| **Active example/test debt** | 3 | same |
| **Comment-only historical bodies** | ~22 in examples | Not executed; stub `main` prints checksum 36 |

Closest to “live fabricated science value” when revived: **`generate_fda_report` reasons** and **`epistemic/meta` weights/provenance** — both epistemic/regulatory surfaces.

## What is *not* claimed

- That today’s green dissertation gates already print zeros from these 19 stdlib lines (host modules fail check).
- That `IntVec::new` is broken (it is real and checks OK).

## Recommended next honesty step (out of this census scope)

1. **Refuse** unknown `Type::method` associated calls (hard error, E137-class or dedicated code) — warning text already admits eval-to-0; wire refuse, not W44.
2. Replace or quarantine the 19 stdlib sites when those modules are ported (`FloatVec` / fixed arrays / real `Vec` only after it exists).
3. Gate: compile-fail witness that `Vec::new` must not `check: OK`.

## Reproduction artifacts

- Source-built ELF: `artifacts/self-hosted/madaros` (session build 2026-08-17)
- Probe: `tests/run-pass/vec_new_nonexistent_type_eval_zero.sio`
- Machine list: `.scratch/vec_new_active_sites.json`

## Bottom line

| Question | Answer |
|---|---|
| How many active `Vec::new` in stdlib? | **19** (22 repo-wide outside comments) |
| Does `Vec` exist? | **No** |
| Does Madaros accept it and eval to 0? | **Yes** (source-built and prebuilt) |
| Does W44 fire? | **No emitter found — silent** |
| Any of the 19/22 on a green Madaros runtime path today? | **No — all host modules fail parse/check** |
| Debt or fabricated value? | **Debt in-tree; fabrication is live at the language boundary** — same class as #1788/#1792 the moment any host goes green or any new code uses the form |

