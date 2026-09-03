<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave15-prebuilt-fire-measurement-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave15-prebuilt-fire-measurement-2026-07-22
-->

# Madaros Wave15 PREBUILT FIRE measurement (2026-07-22)

## Claim under test

Public tip `origin/main` stock `bin/madaros-linux-x86_64` (sha prefix `b097dc08…` after #1405) is RED because squash merges landed a **stale prebuilt** while tip **source** already carries Waves 13–14 fixes.

## Method

1. Branch from `origin/main` at `3e7ed9f52` (`fix(madaros): Wave13e pure paramful single-stmt global element-list fold (#1405)`).
2. Rebuild modular Madaros from tip sources:
   ```bash
   export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
   export SOUNIO_BUILD_LOCK=/tmp/sounio-w15-prebuilt-fire.lock
   bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-w15-prebuilt
   ```
   (Private lock path only to avoid multi-agent stampede on the global lock; same `souc-build-lock` mechanism.)
3. Compare sha256 of rebuild vs stock `bin/madaros-linux-x86_64` / `git show HEAD:bin/madaros-linux-x86_64`.
4. Run the Wave15 gate matrix under the rebuild.

## Result: prebuilt lag hypothesis **FALSIFIED**

| Object | sha256 | bytes |
|---|---|---:|
| `git show HEAD:bin/madaros-linux-x86_64` | `b097dc085ecd4b0d74f8368801c3e2af2607512a044c0d4009d2fa5716d5c715` | 100888549 |
| tip rebuild `artifacts/self-hosted/madaros-w15-prebuilt` | `b097dc085ecd4b0d74f8368801c3e2af2607512a044c0d4009d2fa5716d5c715` | 100888549 |

Bit-identical. Tip rebuild does **not** produce a newer prebuilt. Stock already encodes tip modular sources at `3e7ed9f52`.

## Gate matrix (rebuild = stock, complete tree)

| Gate | Script | Verdict | Class |
|---|---|---|---|
| dual | `scripts/madaros_dual_import_gate.sh` | **PASS** | prebuilt-ok / tip-ok |
| order_spread | `scripts/madaros_order_spread_native_gate.sh` | **PASS** | prebuilt-ok / tip-ok |
| cd_exact e2e | `scripts/madaros_cd_exact_e2e_gate.sh` | **PASS** (ZD PROVED) | prebuilt-ok / tip-ok |
| global_array_init | `scripts/dev/madaros_global_array_init_gate.sh` | **PASS** | prebuilt-ok / tip-ok |
| bare_float_arith | `scripts/madaros_bare_float_arith_gate.sh` | **PASS** | prebuilt-ok / tip-ok |
| root2 multimodule method | `scripts/madaros_root2_multimodule_method_gate.sh` | **PASS** | prebuilt-ok / tip-ok |
| imported_f64_const | `scripts/ci/madaros_imported_f64_const_gate.sh` | **FAIL** | **source residual** |
| wave13 showcase | `scripts/dev/madaros_wave13_showcase_gate.sh` | **FAIL** | cascade of imported_f64 |

### wave13 showcase sub-matrix

| Sub-gate | Verdict |
|---|---|
| wave12_tip_green | FAIL (`imported_f64` / lognormal) |
| dual | PASS |
| order_spread | PASS |
| k95 / epistemic_trust | PASS |
| cd_exact generic i64 | PASS |
| cd_exact e2e | PASS |

## Source residual (not prebuilt)

`scripts/ci/madaros_imported_f64_const_gate.sh`:

- Minimal multi-mod Defect A arm: **PASS** (`IMPORTED_F64_GLOBAL_CONST_OK`).
- Lognormal science vertical: **FAIL** — `FAIL lognormal_pdf (imported DE_LN_SQRT_2PI likely zero)` with `lnpdf_bits 118622047889322840`.

This is the sole required Wave13 showcase failure path under tip. Rebuilding the prebuilt cannot close it.

## Dirty-tree counter-example (do not ship)

A concurrent worktree built `madaros-w15a` with **dirty** `self-hosted/parser/items.sio` (+117/−54 vs tip). That ELF differs:

| Object | sha256 | bytes |
|---|---|---:|
| dirty w15a | `e9f65341373f9c1b567b28962bb0a1af138a4a540b59d58140c8968753acb3fa` | 100891712 |

Under that binary, `global_array` **failed** on `call_list_args_multistmt_residual` (`out=10 2`) while pure tip **passes** — evidence that local parser drift is not a safe prebuilt substitute.

## Worktree hygiene note

This measurement worktree initially had many deleted tracked paths (stdlib cores, `.claude/`, papers, bootstrap seeds). After `git restore` from `origin/main`, the matrix above greened for dual/order_spread/global_array/bare_float/root2. Incomplete trees can look like “stock RED” when the failure is missing inputs, not a stale ELF.

## Disposition

- **No prebuilt PR**: nothing to commit; binary already tip-bit-identical.
- **Do not rewrite compiler source in this lane** (Wave15 Agent 0 rule): remaining red is source residual for a separate f64/BSS science vertical lane.
- Next owner for green showcase: close lognormal `DE_LN_SQRT_2PI` imported-const defect (see `docs/audit/MADAROS_NATIVE_V2_F64_REMAINING_BUGS_2026-07-20.md` family), then re-run `madaros_wave13_showcase_gate`.

## Commands (reproducible)

```bash
git rev-parse HEAD   # expect 3e7ed9f52…
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-w15-prebuilt
sha256sum artifacts/self-hosted/madaros-w15-prebuilt bin/madaros-linux-x86_64
bash scripts/madaros_dual_import_gate.sh
bash scripts/madaros_order_spread_native_gate.sh
bash scripts/madaros_cd_exact_e2e_gate.sh
bash scripts/dev/madaros_global_array_init_gate.sh
bash scripts/madaros_bare_float_arith_gate.sh
bash scripts/madaros_root2_multimodule_method_gate.sh
bash scripts/ci/madaros_imported_f64_const_gate.sh
bash scripts/dev/madaros_wave13_showcase_gate.sh
```
