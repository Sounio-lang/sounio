# Fault-injection results

Base commit: `c90e6cd7d3053a129cb501487f72a656c384094b`

| Mutation | Sounio | Python | Cross-diff | Formal audit | Claim gate | Detected |
|---|---|---|---|---|---|---|
| M1 | DETECTED | UNCHANGED | DETECTED | MISSED | MISSED | YES |
| M2 | DETECTED | DETECTED | MISSED | MISSED | DETECTED | YES |
| M3 | DETECTED | UNCHANGED | DETECTED | MISSED | DETECTED | YES |
| M4 | MISSED | UNCHANGED | DETECTED | MISSED | MISSED | YES |
| M5 | MISSED | MISSED | MISSED | MISSED | DETECTED | YES |
| M6 | MISSED | MISSED | MISSED | DETECTED | MISSED | YES |
| M7 | MISSED | MISSED | MISSED | DETECTED | DETECTED | YES |
| M8 | MISSED | UNCHANGED | DETECTED | MISSED | DETECTED | YES |
| M9 | MISSED | UNCHANGED | DETECTED | MISSED | MISSED | YES |
| M10 | MISSED | MISSED | MISSED | MISSED | DETECTED | YES |

## Limitations

- Lean kernel execution was not available in this sandbox; M6/M7 use deterministic source and theorem-contract audits.
- M9 is a controlled compiler-output corruption proxy, not evidence of a newly discovered souc compiler bug.
- The claim gate is deterministic contract checking, not a blinded human review experiment.
