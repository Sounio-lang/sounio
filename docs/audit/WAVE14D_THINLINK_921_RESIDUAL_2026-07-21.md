<!-- docs:meta
topic_id: repo.docs.audit.wave14d-thinlink-921-residual-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.wave14d-thinlink-921-residual-2026-07-21
-->

# Wave14 Agent D — #921 thin-link residual measure (2026-07-21)

**Issue:** #921 (Defect B from `HYPERCOMPLEX_651_ROOTCAUSE_2026-07-14.md`)  
**Lane:** Wave14 residual wave (post-#1392 specialize collapse)  
**Isolation:** dedicated worktree on `origin/main`  
**Compiler:** stock `./bin/souc` → Madaros v0.80.0 (`bin/madaros-linux-x86_64`)

## Mission

Attack the multimodule compact-IR ELF writer residual that historically failed with
`rc=12` when importing `math::rational` alongside a second module
(`algebra::cayley_dickson::cd_sigma`).

## Measured result (reproducible)

### Default path — filed fail class **CLOSED**

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc compile docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio -o /tmp/mm.elf
# → using full IR path; compact experimental path disabled by default
# → Merged IR: 113 functions; Compilation successful!
/tmp/mm.elf
# → 11
# rc=0
```

Controls:

| Program | Result |
|---|---|
| `math::rational` alone | compile+run `1\n` |
| `algebra::cayley_dickson::{cd_sigma}` alone | compile+run `1\n` |
| both together (handoff repro) | compile+run `11\n` |
| `prob::distributions` scale probe (~210 fns) | compile+run `m=5.000000` |
| `epistemic::knowledge` import | compile OK |

Log markers on the handoff repro (default):

- `module_native_driver: using full IR path; compact experimental path disabled by default`
- `imported_compile: lower_done` / `final_fn_count 113`
- **No** `multimodule native thin-link compilation failed`
- **No** hard `Failed to write native binary rc=12`

Mechanism of close (already on `main`, not introduced by this wave):

1. **PR #1236** (`c62c142df`) — default multi-module route skips the unfinished compact
   imported-simple-IR emitter; always uses `module_frontend_compile_imported_to_file`.
2. Subsequent full-IR multimodule fixes (kind-9 parity #1271, arena rebox, transitive call
   rebind A14, specialize collapse #1392, etc.) keep the full-IR path viable for this pairing.

### Experimental compact path — residual **classified, not hard rc=12**

```bash
SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1 ./bin/souc compile \
  docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio -o /tmp/mm_c.elf
```

Observed:

```
module_native_driver: imported source uses compact modular IR table path (experimental; ...)
Native compilation failed: imported_simple_ir_emit_failed
module_native_driver: compact IR ELF write failed; rc=1
; falling back to full IR path
... full IR succeeds ...
Compilation successful!
```

| Field | Value |
|---|---|
| `compact_fail_class` | `compact_emit_failed` (`imported_simple_ir_emit_failed`) |
| Compact ELF write | fails with `rc=1` (internal simple-IR emit) |
| Full-IR fallback | **yes** — program still compiles |
| Hard thin-link `rc=12` | **no** |
| Final ELF run | `11\n` correct |

The compact emitter is an intentional unfinished stub (comment block in
`self-hosted/compiler/module_native_driver.sio` around the
`SOUNIO_ENABLE_COMPACT_IMPORTED_IR` gate): it only lowers a small table of
simple function shapes. Real stdlib bodies (`Rational` returns, multi-print
main, etc.) correctly refuse emit rather than silently printing hardcoded
`"42\n"` (the silent-corruption class fixed by #1236). Fail-closed + fallback
is the correct residual behaviour until a real compact emitter exists.

### Spec-list DCE branch (`origin/fix/madaros-spec-list-dce`)

Available and not required for this closeout. It reduces specialized ItemFn
count (measured 50→19 on its own lane) and helps memory/thinlink *pressure* on
larger specialize-collapse graphs; the #921 handoff pairing is already green
on stock `origin/main` full IR without it. Not cherry-picked to avoid colliding
with Wave13 Agent B.

## What this does **not** claim

- Compact imported-simple-IR is production-ready (it is not; opt-in only).
- All of D3 is closed — exclusive-ref fragile chains / other multi-module
  residuals may remain; only the **#921 filed thin-link rc=12 pairing** is closed
  on the default path.
- Generic `cd_exact` over `Rational` runtime correctness (pending651 still
  prints `RAT-ZD FAIL` at runtime — orthogonal science residual, not thin-link).

## Gate

```bash
bash scripts/madaros_thinlink_921_residual_gate.sh
# → MADAROS_THINLINK_921_RESIDUAL_GATE_OK
# receipt: artifacts/compiler/madaros_thinlink_921_residual_receipt.v1.json
```

## Acceptance vs #921 body

| Acceptance item from issue | Status |
|---|---|
| `rational` + second module native-compiles on Madaros default | **PASS** |
| Handoff repro no longer hard-fails thin-link | **PASS** |
| Single-module controls stay green | **PASS** |
| Compact experimental completeness | **out of scope** (residual classified) |

## Next action

1. Land residual gate + limitation note (this PR).
2. Close or re-label #921 with evidence pointing at this audit + gate.
3. Leave compact emitter rewrite as a separate experimental lane (not fail-open).
