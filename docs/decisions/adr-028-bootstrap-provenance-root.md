<!-- docs:meta
topic_id: repo.docs.decisions.adr-028-bootstrap-provenance-root
authority: repo_only
audience: users
last_validated: 2026-11-04
validated_by: docs/decisions/PENDING.md (section D-10, as measured there) and the measurements in this ADR
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-028-bootstrap-provenance-root
-->

# ADR-028: The committed seed is the provenance root of the bootstrap

## Context

`bootstrap/README.md` described a C chain, `stage0 → boot0 → … → boot4`, and said
that `boot4.elf` compiles `self-hosted/compiler/lean_single.sio` to produce `gen1.elf`
(D-10, issue #725). That is no longer true, and it was asked whether the chain should
be made to reproduce the current fixed point again or the documentation should say what
is actually the root.

Measured on 2026-11-04 on the workspace, with the committed `artifacts/bootstrap/boot4.elf`
(md5 prefix `5ab5c0160deb`) and the current `lean_single.sio` (2,683,159 bytes, 406,624
tokens):

- `timeout 190 artifacts/bootstrap/boot4.elf self-hosted/compiler/lean_single.sio g1.elf`
  does not finish (rc=124) and loops on `NODE OVERFLOW idx=262143`. `boot4.sio` has
  `nd_cap() = 262143`; the program parses to 291,756 nodes.
- Raising that cap is not enough. With the node arrays at 1,048,575, `boot4` compiles
  `lean_single.sio` in 2.4 s (rc=0) but the compiler it produces does not run, and each
  step uncovered another gap in `boot4` that is not a capacity:
  1. the data segment base is fixed at `0x700000` in about 35 places, so code above
     3 MiB overlaps its own data (SIGSEGV);
  2. non-zero global initializers are not emitted (`var X: i64 = 5` reads 0), while
     `lean_single.sio` has 65 scalar and 59 array-fill initializers (for example
     `SRC_CAP`);
  3. the ELF program header did not count the entry trampoline, so code that follows a
     trampoline with initializers is partly unmapped;
  4. `boot4` has no `f64` support: `v / 2.0` compiles to an integer division and
     faults with SIGFPE. `lean_single.sio` uses `f64` in its own float-literal folding
     (`flc_tok_bits`, `flc_slow`, `flc_scale2`, `flc_pow10_exact`).
- The chain that `make build` actually runs starts from `bin/souc-linux-x86_64`, not from
  `boot4.elf`: `gen2` and `gen3` have the same md5 (prefix `a5ab9a1f06ac`), and it is the
  md5 of the committed `bin/souc-lean-single-x86_64`.

## Decision

The committed seed `bin/souc-lean-single-x86_64`, with its receipt
`bin/souc-lean-single-x86_64.SeedReceipt.json`, is the **provenance root** of the
self-hosted compiler. It is checked by `scripts/ci/verify_lean_seed.sh`; the diverse
double-compilation leg runs with `SOUNIO_SEED_DDC=1`.

The C chain `stage0 → boot4` stays in the repository as the historical minimal bootstrap.
It is not claimed to reproduce the current fixed point, and `boot4` is not claimed to
analyse the current `lean_single.sio`.

`bootstrap/README.md` is rewritten to say so.

## Relationship to other decisions

- **ADR-006 (fixed point as trust anchor) is amended, not replaced.** `gen2 == gen3`
  stays the check. ADR-006 names `artifacts/bootstrap/boot4.elf` as the curated anchor and
  requires the chain `boot4 → gen1 → gen2 → gen3`; that chain cannot run on the current
  source (see Context). The chain starts at the committed seed instead.
- **ADR-027 §4.1 (diverse double-compiling)** already records the seed SHA in
  `bootstrap-receipt.json`. This ADR is consistent with it.
- **Madaros bit-identical requirement (board epic SIO-2).** The requirement that the
  sha256 of Madaros `gen2` through `gen7` be identical applies to Madaros, whose chain
  starts from this seed (`make build-madaros`). The C chain is not part of it. The founder
  confirmed this reading on 2026-11-04.

## Consequences

- The trust argument for the compiler rests on the seed, its receipt and the DDC leg. A
  reader who needs a root that is auditable from C source alone does not have one today.
- Restoring the C chain (making `boot4` compile the current `lean_single.sio`) would need,
  at least: larger node arrays, a data base that does not depend on code size, global
  initializers, a corrected program header, and `f64` support or a `lean_single.sio` that
  does not use `f64` itself. That is a decision for a future ADR, to be opened only if the
  C chain becomes a trust requirement. The partial attempt is not in the repository.

## Acceptance

1. `bootstrap/README.md` names the seed and the receipt as the root, states that `boot4`
   does not analyse the current `lean_single.sio`, and no longer says that `boot4.elf`
   compiles it.
2. `docs/decisions/PENDING.md` no longer contains D-10.
3. ADR-006 carries a pointer to this ADR.
4. `SOUNIO_SEED_DDC=1 scripts/ci/verify_lean_seed.sh` is unchanged by this ADR and keeps
   passing.
