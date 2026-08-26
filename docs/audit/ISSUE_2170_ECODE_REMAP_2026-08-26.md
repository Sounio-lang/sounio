<!-- docs:meta
topic_id: repo.docs.audit.issue-2170-ecode-remap-2026-08-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.issue-2170-ecode-remap-2026-08-26
-->

# Issue #2170 - Collision-free E2xx remap

## Decision

Preserve the published lean_single/catalogue identities on E208, E217, E218,
and E219. Move the four unrelated Madaros reuses to fresh global identities:

| Published identity kept | Madaros identity moved | Fresh code |
|---|---|---|
| E208 refinement integer predicate | malformed ZD locus | E247 |
| E217 invalid function body span | unsupported f128/f256 conversion | E248 |
| E218 tail type mismatch | V0-A f128/f256 reservation | E249 |
| E219 function pass mismatch | unsupported extern C call | E250 |

E210 and E220 were present in the original census but are no longer live
collisions on the implementation base. Their unrelated Madaros uses already
moved to E246 and E245 respectively.

## Evidence boundary

`scripts/ci/diagnostic_remap_2170_gate.sh` checks the semantic message-to-code
pairing, exact emitter counts, parser tag, catalogue rows, and explanation
files. It then sabotages each mapping back to its former colliding number; all
four mutants must be refused.

Source-fresh live witnesses cover E247, E249, and E250. E248 is structural-only
in this wave because the earlier E249 parser reservation makes a source-level
wide-float cast unreachable. This is recorded as a boundary, not reported as a
live conversion refusal.

The namespace census remains a ratchet, not a claim that every historical
collision is resolved. This wave removes exactly four documented-vs-reissue
collisions. Other collision classes retain separate ownership.

## Executed receipts

- Diagnostic identity ratchet: `collisions=21`, down from 25;
  `undocumented=140`; `orphaned=25`. The four additional orphan rows are the
  preserved lean_single identities whose messages are still emitted without a
  code tag.
- Remap gate: all four static mappings passed and all four sabotage mutations
  were rejected.
- Source-fresh Madaros: SHA-256
  `b9a4dd9ec5a0a46fa4e0613df3e1a20f3f871ea5552f159af5794cd76eec0099`,
  101388577 bytes. The build used the independently fixed-pointed #1678 seed
  `455365f19b6c96506991cfac5fed3d86ca655a324567d71bc9309ae5cd2aa759`.
- Shipped Madaros: `bin/madaros-linux-x86_64` is that exact source-fresh ELF.
  Its gate receipt names source commit `aac119fb941906c65764167fc82dae3a510c5475`
  and records `madaros_full_gate.sh=pass`.
- Live remap witnesses: malformed ZD locus refused with E247; reserved
  wide-float source refused with E249; unsupported extern call refused with
  E250. Each returned `rc=1` and none printed `check: OK`. These witnesses also
  passed through the public default `bin/souc` path after artifact promotion.
- Extern live gate: implemented control checked clean; unsupported call refused
  with E250 and produced no ELF.
- CI wiring: Contracts runs the structural mapping and all four sabotage
  controls; Madaros Witness Gate reruns the live source-reachable witnesses
  against `/tmp/madaros-ci.elf` built from the commit under test.
- Website diagnostic modules: targeted TypeScript check passed. Full Astro
  check could not allocate its WebAssembly instance in the pod; whole-project
  `tsc` also retains the unrelated baseline implicit-any in `feed.xml.ts`.

The default modular build route first derived a seed and then segfaulted while
compiling `main.sio` (`rc=139`). The independent #1678 seed crossed that point
and produced the source-fresh ELF above. Both observations are retained; the
successful alternative does not erase the default-route failure.
