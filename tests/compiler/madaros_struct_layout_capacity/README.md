# Madaros Struct Layout Capacity Boundary

`scripts/ci/madaros_struct_layout_capacity_gate.sh` generates two imported
native witnesses rather than committing hundreds of mechanically repeated
struct declarations.

The lowerer pre-registers `Knowledge` in `lowerer_new_with_epistemic`, so the
relevant catalog counts are:

| Fixture | Custom imported layouts | Pre-registered layout | Total catalog layouts |
| --- | ---: | ---: | ---: |
| lower boundary | 255 | `Knowledge` | 256 |
| overflow boundary | 256 | `Knowledge` | 257 |

Each witness imports the final generated struct and reads its uniquely named
final field. That makes a dropped final layout observable: the legacy
field-name hash fallback cannot accidentally satisfy the expected value.

The default `baseline` mode is a classification witness for the unrepaired
compiler: the 256-entry total must execute, while the 257-entry total must not
look like a successful exact runtime witness. `resolved` mode requires both
native witnesses to execute and print their exact markers using one explicit
Madaros ELF through `MADAROS_RAW_BIN`; this focused gate does not make a
separate native-dispatch fallback claim.

This fixture is support evidence for `BLK-20260720-D11-D12-IR-SUMMARY`; it does
not by itself establish D6, D11, or D12 imported runtime parity.
