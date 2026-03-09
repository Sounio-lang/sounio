<!-- docs:meta
topic_id: website.docs.compiler.effect-system
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler.effect-system
-->

# Sounio Effect System Architecture

The effect system is both a language-facing contract and a dedicated implementation subsystem. The public surface is still best taught through explicit `with ...` clauses, while the contributor-facing implementation lives primarily in the self-hosted checker and effect modules.

## Current implementation map

- `self-hosted/check/effects.sio`: effect-aware semantic checking inside the main checker
- `self-hosted/check/effects_row.sio`: row-like or composition-oriented effect reasoning
- `self-hosted/effects/types.sio`: effect representations
- `self-hosted/effects/checker.sio`: effect-specific checking logic
- `self-hosted/effects/handlers.sio`: handler-oriented implementation support
- `docs/architecture/EFFECT_HANDLERS_IMPLEMENTATION.md`: broader design background

## Public contract that is safe to teach

- effects are explicit in function signatures
- `IO` remains the clearest everyday effect example
- effectful operations should not be described as silently available in pure contexts

Representative surface example:

```sio
fn read_config(path: string) -> string with IO {
    read_line()
}
```

## Contributor reading order

1. start from a checked example or failing fixture
2. inspect `self-hosted/check/effects.sio`
3. step into `self-hosted/effects/`
4. use `docs/architecture/EFFECT_HANDLERS_IMPLEMENTATION.md` only as design context, not as the primary current implementation map

## Documentation rules

- Keep user docs centered on the signature-level contract.
- Be explicit when you are discussing handlers or richer implementation work that may not be equally proven in the checked public artifact.
- Do not let the deeper design obscure the basic rule that effect boundaries are part of the type-level surface.
