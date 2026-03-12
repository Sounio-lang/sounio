# Cross-Skill Index

Maps skills to the programs they are primary or supporting for.
`PRIMARY` = the skill was created to own this program's domain.
`X` = skill is relevant to this program's work.
`—` = not applicable.

## Program Matrix

| Skill | Render Platform (A) | Frontend Bootstrap (B) | PGO Pipeline (C) | Native Codegen (D) | Core Language |
|-------|--------------------|------------------------|------------------|--------------------|---------------|
| `sounio-render` | **PRIMARY** | — | — | — | — |
| `sounio-bootstrap` | — | **PRIMARY** | — | — | — |
| `sounio-pgo` | — | — | **PRIMARY** | X | — |
| `sounio-native-codegen` | — | — | X | **PRIMARY** | — |
| `sounio-examples-hygiene` | X | X | — | — | X |
| `sounio-language` | X | — | — | — | **PRIMARY** |
| `sounio-stdlib-dev` | X | — | — | — | X |
| `sounio-codegen-backends` | X | X | X | X | — |
| `sounio-typeck-effects` | — | X | — | — | **PRIMARY** |
| `sounio-compiler-mir` | — | — | X | — | — |
| `sounio-l0-core` | — | — | — | — | **PRIMARY** |
| `sounio-epistemic-types` | X | — | — | — | X |
| `sounio-tests-suite` | X | X | X | X | X |
| `sounio-tooling` | X | — | — | — | X |
| `sounio-ontology` | — | — | — | — | X |
| `model-dispatch` | X | X | X | X | X |

## Skill → Program Coverage Count

| Skill | Programs covered |
|-------|----------------|
| `model-dispatch` | 5 (all) |
| `sounio-tests-suite` | 5 (all) |
| `sounio-codegen-backends` | 4 |
| `sounio-examples-hygiene` | 3 |
| `sounio-pgo` | 2 (primary C) |
| `sounio-native-codegen` | 2 (primary D) |
| `sounio-render` | 1 (primary A) |
| `sounio-bootstrap` | 1 (primary B) |
| `sounio-language` | 2 |
| `sounio-typeck-effects` | 2 |
| `sounio-tooling` | 2 |
| `sounio-stdlib-dev` | 2 |
| `sounio-epistemic-types` | 2 |
| `sounio-l0-core` | 1 |
| `sounio-compiler-mir` | 1 |
| `sounio-ontology` | 1 |

## Sprint → Skill Mapping

| Sprint | Skills active |
|--------|--------------|
| 53 (render contract) | `sounio-render`, `sounio-examples-hygiene`, `sounio-stdlib-dev` |
| 54 (GPU contract) | `sounio-render`, `sounio-codegen-backends` |
| 55 (website) | `sounio-render`, `sounio-tooling` |
| 56 (frontend corpus) | `sounio-bootstrap`, `sounio-typeck-effects` |
| 57 (IR fidelity) | `sounio-bootstrap`, `sounio-compiler-mir`, `sounio-codegen-backends` |
| 58 (bootstrap binary) | `sounio-bootstrap`, `sounio-codegen-backends` |
| 59 (new skills) | `model-dispatch` |
| 60 (dispatch update) | `model-dispatch` |
| 61 (coverage gate) | `model-dispatch`, all 15 skills |
| 62 (4-preg regalloc) | `sounio-native-codegen`, `sounio-pgo` |
| 63 (disp8 encoding) | `sounio-native-codegen` |
| 64 (compact frame) | `sounio-native-codegen` |
| 65 (peephole wiring) | `sounio-native-codegen` |
| 66 (native codegen skill) | `sounio-native-codegen`, `model-dispatch` |
