# Loom Fleet Catalog v2

## Purpose

Catalog v1 persisted which provider-shaped slot should exist, but every slot
was reconstructed through the compatibility `agentd` launcher. Catalog v2
makes terminal authority explicit:

```text
desired custody = agentd | loom
```

This is desired state, not a label inferred from the process name. The
reconciler observes both authorities before it mutates either surface.

## Stored authority

Every v2 record stores:

- enabled state, slot, provider kind, agent identity, credential home, and cwd;
- the selected custody implementation and shared coordination authority;
- for Loom custody, a stable session UUID, optional model, and explicit unsafe
  policy;
- a private bootstrap prompt path and its SHA-256 digest.

The bootstrap prompt is copied to `<state>/fleet/prompts/<slot>.txt` with mode
`0600`. The descriptor contains only its path and digest. A missing, relocated,
or modified prompt refuses catalog loading before provider launch.

`coord_dir` is the absolute durable state directory for the coordination bus.
Enrollment derives it from the controlling Git worktree or accepts an explicit
`--coord-dir`. Reconciliation injects it as `SOUNIO_COORD_DIR` into both new and
recovered kernels. Therefore a lane whose provider cwd is another repository
still refreshes presence and endpoint records in the selected Sounio bus. A
legacy v2 entry without this field can be re-enrolled with `--replace`; until
then, mutation refuses rather than registering into an implicit local bus.

Version 1 records remain readable and mean `custody=agentd`. The next successful
enrollment rewrites them as version 2 without changing that authority.

## Reconciliation state machine

For `custody=agentd`:

```text
Loom active or recoverable -> REFUSE fleet-authority-conflict
agentd active             -> NOOP
agentd absent             -> PLAN start / APPLY launch-kind
```

For `custody=loom`:

```text
agentd active             -> REFUSE fleet-authority-conflict
Loom active               -> NOOP
Guardian active, kernel dead -> PLAN recover / APPLY recover
Loom absent               -> PLAN provider-open / APPLY provider-open
```

After mutation, the reconciler observes the selected authority again and
requires `active`. A launcher exit code alone is not acceptance evidence.

## Active adoption

`--adopt-active` brings a matching, already-running Loom lane under catalog
desired state. The command verifies:

- agent and lane identity;
- worktree path;
- Loom session UUID;
- logical provider executable.

Without the flag, enrollment refuses an uncatalogued active lane. With the flag
but any identity mismatch, enrollment also refuses. Prompt and descriptor are
published only after these checks.

Active adoption does not convert a live `agentd` or tmux PTY. Loom cannot seize
another supervisor's file descriptors without violating exclusive input
authority. A legacy lane must be stopped before a new Loom generation is
opened. That boundary is a product invariant, not an implementation gap to
hide with process discovery.

## Current boundary

Persistent catalog custody is initially available for Codex because Codex is
the first provider with a verified `provider-open` adapter. Other provider kinds
fail closed. Provider or Guardian loss permits a new physical generation from
the sealed bootstrap intent; it cannot preserve the dead provider process or
claim that the new conversation is the old one. Kernel-only loss preserves the
Guardian, provider PID, instance, PTY, output cursor, and conversation.

The executable sabotage gate is:

```sh
scripts/ci/sounio_loom_fleet_custody_selftest.sh
```

It proves prompt tamper refusal, forged-custody refusal, dual-authority refusal,
explicit active adoption, idempotent reconciliation, and stable-provider kernel
recovery.
