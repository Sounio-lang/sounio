<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-array-byvalue-913-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-array-byvalue-913-2026-07-21
-->

# #913 closeout — imported `[f64;N]` by-value is preserved (Wave14 Agent B)

**Date:** 2026-07-21  
**Issue:** https://github.com/Sounio-lang/sounio/issues/913  
**Branch:** `fix/madaros-w14b-913-array-byvalue`  
**Engine:** default `./bin/souc` → **Madaros v0.80.0** multi-module native  
**Verdict:** **CLOSED on tip** — no compiler change required; regression gate + receipt only.

## Defect (historical)

Passing a fixed array `[f64;N]` **by value** to an *imported* function delivered a
**zeroed** array to the callee. Same-module by-value and cross-module `&[f64;N]`
by-ref were fine. Original science symptom: `fit_simple_linear_regression(x, y, 3)`
on `y = 2x` printed `slope=0` instead of `2`.

## Measurement (origin/main tip)

| Probe | Engine | Result |
|---|---|---|
| Minimal multi-mod leaf: sum/elem/`[f64;4]` + `[f64;100]` by value | Madaros default | **GREEN** — payload preserved |
| OLS slope leaf over `[f64;100]` by value (`y=2x` → slope bits of `2.0`) | Madaros default | **GREEN** |
| Same minimal + OLS witness | `SOUNIO_SOUC_ENGINE=lean_single` | **GREEN** |
| dual gum+knowledge | Madaros default | **GREEN** (`MADAROS_DUAL_IMPORT_GATE_OK`) |
| cd_exact e2e | Madaros default | **GREEN** (`MADAROS_CD_EXACT_E2E_GATE_OK`, ZD PROVED) |

Bit oracles (little-endian IEEE-754):

| value | `f64_to_bits` |
|---:|---:|
| 1.0 | `4607182418800017408` |
| 2.0 | `4611686018427387904` |
| 6.0 | `4618441417868443648` |

Tip identity at measurement:

- `git_sha` = `1b289fda3` (`origin/main` at Wave14 B ship)
- `raw_elf` = `bin/madaros-linux-x86_64`
- `raw_elf_sha256` = `86a1742864e8989347ea7ba1be2ee1e4828db9aab8a6eedbb8496b461ea5de51`

## Gate

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true
bash scripts/ci/madaros_imported_array_byvalue_gate.sh
# → MADAROS_IMPORTED_ARRAY_BYVALUE_GATE_OK
```

Witnesses:

- `tests/run-pass/imported_array_byvalue.sio`
- `tests/run-pass/fixtures/imported_array_byvalue_leaf.sio`

Machine receipt: `docs/audit/receipts/madaros_imported_array_byvalue_913_2026-07-21.json`

## Claim boundary (explicit)

**Claimed closed:**

- Cross-module call ABI for **`[f64;N]` by value** (N=4 and N=100 measured) under
  default Madaros multi-module native: callee sees the caller's payload, not zeros.
- Science-shaped acceptance without `stats::regression::linear`: imported OLS slope
  over by-value `[f64;100]` yields slope bits of `2.0` for `y = 2x` (N=3 points).

**Not claimed / residual (separate lanes):**

- Importing full `stats::regression::linear` under Madaros multi-mod currently fails
  **parse** on that module's `impl` methods (`expected token` cluster ~L162+). That is
  **not** the by-value array ABI defect measured here.
- Under lean_single, a direct `fit_simple_linear_regression` import still reported
  slope bits `0` while the isolated by-value OLS leaf was green — treat that as a
  **distinct** stdlib-path residual (struct return / model field / denser call graph),
  not as evidence that the simple by-value array ABI is still zeroing.
- Do not fight Wave13 residual lanes: into-acc, bare f64 Ident, global list args,
  specialize_generics DCE (#1397).

## Why no `self-hosted/` fix

The characterized defect is already green on the committed prebuilt Madaros on tip
(`#1392` and prior multi-mod native work). Shipping a **fail-closed regression gate**
is the correct closeout; a speculative compiler patch would risk dual/cd_exact without
evidence of a remaining ABI zeroing path.

## AI disclosure

Measurement, gate, and audit by Wave14 Agent B under human direction. No math claim
beyond bit-identity of f64 literals. GAIDeT-ICMJE 2025.
