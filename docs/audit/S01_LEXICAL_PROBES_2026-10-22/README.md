<!-- docs:meta
topic_id: repo.docs.audit.s01-lexical-probes-2026-10-22.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.s01-lexical-probes-2026-10-22.readme
-->

# Lexical probes — raw material, 2026-10-22

Raw material for the [lexical execution receipt](../../spec/2026-10/18-lexical-execution-receipt.md)
(chapter 18 of the 2026-10 edition).

- `gen_probes.sh` writes the 125 probe programs into `probes/` (regenerated, not committed).
- `run_probes.sh [out.tsv]` runs each probe; one record per probe and engine:
  `probe  engine  rc  condensed output`. `MADAROS_ELF=path` runs that Madaros binary
  directly; `ENGINES="madaros lean_single"` selects engines. Probes run from a scratch
  directory.
- `results_committed.tsv` — committed Madaros ELF (`bin/madaros-linux-x86_64.gz`) and the
  seed (`bin/souc-lean-single-x86_64`), all 125 probes.
- `results_source_built.tsv` — Madaros built from `main@27afa340`, all 125 probes.
- `results_fixed.tsv` — the same build with the companion lexer change (receipt §4).
- `seed_fallback_sweep.tsv` — every `.sio` file under `tests/`, `stdlib/` and `examples/` that a
  seed refusing *every* token at `compile_primary`'s last arm would refuse, with the current seed's
  exit status (0 = it compiles today). Receipt §6.

`diff <(cut -f1,3 results_source_built.tsv) <(cut -f1,3 results_fixed.tsv)` lists the
probes whose exit status the change moves.

These are measurements, not conformance tests: no probe asserts an expected outcome until
the rulings in [ADR-015](../../decisions/adr-015-lexical-structure.md) (formerly `PENDING.md` D-33) are made.
