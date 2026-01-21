---
name: sounio-l0-core
description: "L0 epistemic core of Sounio: language identity, epistemic invariants, and effect discipline; use when explaining or extending the L0 language semantics."
---

# Sounio L0 Core

## Workflow

1) Anchor on language identity
- Read `docs/MV_CORE_CHECKLIST.md` first.
- Confirm: explicit effects, epistemic integrity, and Sounio-native syntax.

2) Epistemic invariants
- Use `stdlib/epistemic/SEMANTICS.md` for Knowledge/uncertainty/confidence/provenance rules.
- Never allow implicit Knowledge→T unwrapping.

3) Effects and purity
- Consult `.claude/commands/sounio-effects.md` for effect expectations.
- Ensure side effects are declared on functions (`with IO`, `with Panic`, etc.).

4) Units and refinements (when relevant)
- Use `.claude/commands/sounio-units.md` to keep unit syntax and checks consistent.

## References
- `docs/MV_CORE_CHECKLIST.md`
- `stdlib/epistemic/SEMANTICS.md`
- `spec/LANGUAGE_SPECIFICATION.md`
- `.claude/commands/sounio-effects.md`
- `.claude/commands/sounio-units.md`
