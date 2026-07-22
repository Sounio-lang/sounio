# Madaros Imported Layout Identity Acceptance

`issue_901_nested_field_chain_main.sio` proves that an imported chained field
access retains the declared nominal type of the intermediate field. The leaf
places `family_id` at a different offset than the unrelated outer layouts, so
the expected runtime value is `520`, not the adjacent `protocol_id` value
`8400`.

`issue_901_known_layout_miss_main.sio` is the complementary negative witness.
Its inner type has no `family_id`, while an unrelated imported type does. The
compiler must reject that known-layout miss; it may not reuse the global field
name fallback.

For a behavioral replay against an explicitly supplied raw ELF, run
`scripts/ci/madaros_imported_runtime_acceptance_gate.sh` with `MADAROS_RAW_BIN`.
That replay proves only the supplied ELF's behavior. For current-source
evidence, run `scripts/ci/madaros_imported_runtime_source_fresh_gate.sh`; it
builds the raw ELF in the clean checkout and records the source-to-binary
receipt before running this fixture pair.
