# Exact executable-function census joins

Strict parser19800:
- WIDE: functions19, instructions634, arguments37.
- Scalar: functions14, instructions79, arguments6.

| Capture | fi/name | Instructions | Registers | Actual params | Rounds | Missing |
| --- | --- | --- | --- | --- | --- | --- |
| WIDE19724 |0 wide_core_literal |265 |134 |3 |2 |3 |
| WIDE19724 |1 wide_core_hoist |263 |133 |3 |2 |3 |
| WIDE19724 |2 main |106 |84 |0 |2 |0 |
| Scalar19737 |0 core |54 |29 |1 |2 |3 |
| Scalar19737 |1 main |25 |17 |0 |2 |0 |

Every listed instruction is reachable in these captures. Each WIDE core lacks
reg16 at instructions82/92/97; scalar core lacks reg8 at30/40/45. All remaining
captured descriptors (WIDE16, scalar12) are explicitly listed as empty in the
same complete logs. There are no DATA descriptors in these two baselines.

The strict parser's counts join the full decoded census, then the per-function
audits and explicit empty/DATA counts join the global summary. Final missing totals
are6 and3 respectively; both audits exit1. The C++ component gate itself exits0
only after proving those expected baseline refusals and41 separate controls.

Complete original captures are retained compressed under19800 with their original
decompressed SHA in CAPTURES.json. Per-control audit logs are compressed byte-exact
(including their original trailing newline bytes); AUDIT-FILES.json records both
compressed and decompressed identity. These transformations do not change input IR
or the underlying component results.
