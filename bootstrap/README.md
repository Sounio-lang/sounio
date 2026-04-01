# Bootstrap Chain

The Sounio compiler is self-hosting: it compiles itself from a minimal seed compiler written in C (`stage0.c`).

## The Chain

| Stage | Source | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| **Seed (stage0)** | `stage0.c` | (C code) | `boot0.elf` | C seed compiler — cannot be lost |
| **boot0** | `boot0.sio` | `boot0.elf` | `boot1.elf` | Minimal Sounio source |
| **boot1** | `boot1.sio` | `boot1.elf` | `boot2.elf` | Slightly enriched |
| **boot2g** | `boot2g.sio` | `boot2g.elf` | (→ boot3) | Graphical/variant staging |
| **boot3** | `boot3.sio` | `boot3.elf` | (→ boot4) | Extended feature support |
| **boot4_a1** | `boot4_a1.sio` | `boot4.elf` (**production**) | `gen1.elf` | **CURRENT PRODUCTION HEAD** |

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
boot4.elf compiles lean_single.sio → gen1.elf
gen1.elf compiles lean_single.sio → gen2.elf
gen2.elf compiles lean_single.sio → gen3.elf

# Check if gen2 == gen3 (by MD5)
# If they match: FIXED POINT OK
```

When `gen2 ≡ gen3` (bit-identical), the bootstrap chain is **self-hosting**: the compiler can compile itself.

## Artifacts

All compiled ELF binaries are stored in `artifacts/bootstrap/`:
- `boot4.elf` — the current seed for rebuilding the compiler
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
