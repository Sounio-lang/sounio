<!-- docs:meta
topic_id: repo.docs.audit.souc-provenance-local-build-gate-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.souc-provenance-local-build-gate-2026-08-31
-->

# `souc --version` said "COMMITTED" about a local build — the provenance line now tells the truth, and can refuse

**Date:** 2026-08-31. **Trigger:** issue #2318 (closed as not reproducible) and its audit
`MADAROS_LOCAL_SLOT_OVERFLOW_2026-08-31.md`: a day of compiler measurements — a filed defect, a
bisection, a "silent wrong values" claim — were made through `bin/souc` in a lane worktree that
resolved `artifacts/self-hosted/madaros` (a **gitignored local build dated 2026-08-16**, md5
`709acf97`) ahead of the committed `bin/madaros-linux-x86_64`, while every `souc --version` in
those logs printed

```
provenance: this ELF is the COMMITTED binary; it is not built from the tree above.
```

Nothing in the chain could say otherwise, and nothing could be told to refuse. The same trap
had already bitten once (#2315). This is, in the paper's own vocabulary, an anti-garbling by the
toolchain: a certainty ("COMMITTED") manufactured about something never measured (the ELF's
identity).

## Change (additive; no resolution order changed)

`bin/souc`
- `_madaros_provenance_kind <elf>` → `committed` iff the resolved ELF **is**
  `bin/madaros-linux-x86_64` or is byte-identical to it (`cmp -s`); otherwise
  `local artifact built <mtime>` (for `artifacts/self-hosted/madaros`) or
  `override <path> built <mtime>` (for `MADAROS_RAW_BIN`/`SOUNIO_MADAROS_BIN`).
- `souc --version` prints the truth on stderr (stdout unchanged — ~657 call sites parse it):
  ```
  provenance: this ELF is a LOCAL BUILD (local artifact built 2026-08-16T04:46), NOT the committed bin/madaros-linux-x86_64 (md5=ff69dae4); it was resolved ahead of the committed ELF.
  provenance: set SOUNIO_REQUIRE_COMMITTED_MADAROS=1 to refuse local builds; a claim about the committed compiler must not be measured on this one.
  ```
- **Strict mode, opt-in:** `SOUNIO_REQUIRE_COMMITTED_MADAROS=1` makes `bin/souc` (every verb)
  refuse to run any ELF that is not the committed prebuilt — exit 78, naming the resolved path
  and its kind. Intended for measurement sessions, claim gates and CI jobs whose subject is *the
  committed compiler*.

`bin/madaros` — the same strict check after `_resolve_modular_elf`, so scripts that call
`bin/madaros` directly are covered.

Cost: the committed path pays nothing (path equality short-circuits); a local artifact pays one
`cmp` of ~100 MB per invocation (~50 ms), which is the point.

## Gate

`scripts/ci/compiler_override_fail_closed_gate.sh` grows from 13 to 18 cases (7 → 9 refusals):
- strict mode refuses a byte-different ELF, in `souc` and in `madaros` (`local.elf` = the
  committed ELF plus one trailing byte — still runs, no longer *is* the committed binary);
- strict mode honours a byte-identical copy at another path (`ok.elf`);
- `--version` names a local build honestly (says `LOCAL BUILD`, never `is the COMMITTED
  binary`) and recognises a byte-identical copy as committed.

Verified on the real fixture too: the 2026-08-16 stale artifact, passed as `MADAROS_RAW_BIN`, is
reported `LOCAL BUILD (override … built 2026-08-16T04:46)` and is refused under strict mode.

## What it does not do

It does not change which ELF is resolved (local builds still win, by design, for development),
and it does not know whether a local build is *newer or older* than the committed one — only
that it is *not* the committed one. "Build from source for a claim about compiler behaviour"
stays the rule; this makes the other rule enforceable: "for a claim about the committed compiler,
run the committed compiler".
