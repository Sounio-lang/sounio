# Bootstrap Chain

The Sounio compiler is self-hosting: its provenance root is the committed seed described under "Provenance Root" below. `stage0.c` is the historical C seed of the chain.

## Provenance Root

The root of the current compiler is the **committed seed**
`bin/souc-lean-single-x86_64`, with its receipt
`bin/souc-lean-single-x86_64.SeedReceipt.json`. It is checked by
`scripts/ci/verify_lean_seed.sh`; add `SOUNIO_SEED_DDC=1` for the diverse
double-compilation leg. See [ADR-028](../docs/decisions/adr-028-bootstrap-provenance-root.md).

The C chain below (`stage0.c` → … → `boot4`) is kept as the historical minimal
bootstrap. **`boot4` does not analyse the current `self-hosted/compiler/lean_single.sio`**
(measured 2026-11-04: its node cap of 262143 is below the 291756 nodes that file needs,
and `boot4` also lacks non-zero global initializers and `f64`), so the chain is not a
route to the current fixed point.

## The Chain

| Stage | Source | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| **Seed (stage0)** | `stage0.c` | (C code) | `boot0.elf` | C seed compiler — cannot be lost |
| **boot0** | `boot0.sio` | `boot0.elf` | `boot1.elf` | Minimal Sounio source |
| **boot1** | `boot1.sio` | `boot1.elf` | `boot2.elf` | Slightly enriched |
| **boot2g** | `boot2g.sio` | `boot2g.elf` | (→ boot3) | Graphical/variant staging |
| **boot3** | `boot3.sio` | `boot3.elf` | (→ boot4) | Extended feature support |
| **boot4_a1** | `boot4_a1.sio` | `boot4.elf` | `gen1.elf` | Last stage of the historical C chain; does not analyse the current `lean_single.sio` |

## Compiler Sources (Self-Hosted)

All files in `self-hosted/` are Sounio source code. The **canonical compiler driver** is:

```
self-hosted/compiler/lean_single.sio
```

This is the source the bootstrap chain compiles to demonstrate fixed-point achievement.

## Fixed-Point Verification

```
make build
```

This runs the 3-stage fixed-point check:

```
bin/souc-linux-x86_64 compiles lean_single.sio → gen1.elf
gen1.elf compiles lean_single.sio → gen2.elf
gen2.elf compiles lean_single.sio → gen3.elf

# Check if gen2 == gen3 (by MD5)
# If they match: FIXED POINT OK
```

When `gen2 ≡ gen3` (bit-identical), the bootstrap chain is **self-hosting**: the compiler can compile itself.

## Artifacts

All compiled ELF binaries are stored in `artifacts/bootstrap/`:
- `boot4.elf` — compiler of the historical C chain; not the seed of the current compiler (see Provenance Root)
- `souc-native-v1.0.0.elf` — named release artifact
- `micro_bootstrap.elf` — alternative minimal bootstrap

## Naming Rationale

- `boot*` stages use increasing numbers (0,1,2,3,4) to show progression
- `boot2g` = "graphical" variant (explored but superseded)
- `boot4_a1` = "revision a1" of boot4 (the current production version)
- `gen*` stages in the fixed-point check are numbered 1,2,3 to show compilation iterations

## Reference

- Full bootstrap design: See CLAUDE.md section "Bootstrap chain"
- Self-hosting architecture: `docs/compiler/SELF_HOSTING_DESIGN.md`
- Build entry point: `Makefile` targets `build`, `check`
