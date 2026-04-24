<!-- docs:meta
topic_id: repo.docs.governance.docs-conventions
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.governance.docs-conventions
-->

# Documentation Conventions

Minimal rules for contributing to Sounio docs.

## Frontmatter

Every doc page starts with a frontmatter comment block:

```html
<!-- docs:meta
topic_id: repo.docs.[section].[page-name]
authority: repo_only | dual
audience: users | contributors
last_validated: YYYY-MM-DD
validated_by: your-name
-->
```

## Status Badge

After the frontmatter, add a visible status line:

```
> **Status**: Production | **Last validated**: YYYY-MM-DD | **Source**: tests/run-pass/
```

Status values:
- **Production** — gate-backed, tested, reliable
- **Beta** — works for common patterns, edge cases may exist
- **Planned** — specified but not yet implemented

## Where Docs Live

| Directory | Purpose |
|---|---|
| `docs/guide/` | User-facing guides (getting started, tutorial, cookbook, syntax reference) |
| `docs/reference/` | API references (stdlib, Knowledge types) |
| `docs/stdlib/` | Standard library docs |
| `docs/spec/` | Language specification |
| `docs/compiler/` | Compiler architecture and internals |
| `docs/contributor-guide/` | Contributor workflow, style guide, benchmarks |
| `docs/COOKBOOK.md` | Task-oriented recipes |
| `docs/FAQ.md` | Frequently asked questions |
| `docs/MIGRATION_GUIDE.md` | Version upgrade guide |
| `docs/archived/` | Historical docs, replaced by current docs |
| `docs/internal/` | Process artifacts, sprint reports, NOT user-facing |

## Rules

1. **One getting-started doc**: `docs/guide/getting-started.md` is canonical. Do not create new quick-start or getting-started docs.

2. **Mark implementation status**: When describing language features, mark them as Production, Beta, or Planned. Do not present planned features as working.

3. **Verify code snippets**: Code examples should come from or be tested against `tests/run-pass/`. Run `python3 scripts/dev/sounio-lint.py` against docs.

4. **Use `sio` code fences**: Use ` ```sio ` for Sounio code blocks. ` ```sounio ` is also acceptable.

5. **No Rust syntax**: Sounio is not Rust. Never use Rust-isms in examples.

6. **No duplication**: If a topic is covered elsewhere, link to it instead of repeating.

7. **Keep internal docs separate**: Sprint reports, agent handoff logs, and implementation status belong in `docs/internal/` or `artifacts/`, not in user-facing paths.

## Archiving Docs

When a doc is superseded:
1. Move it to `docs/archived/`
2. Add a `README.md` in the archive directory explaining what replaced it
3. If the original path is referenced widely, leave a redirect stub

## Topic Registry

The detailed topic registry lives at `docs/governance/topic-registry.v1.json` and the authority matrix at `docs/governance/DOCS_AUTHORITY_MATRIX.md`. These are maintained for historical continuity but contributors do not need to update them for routine doc changes. Just add frontmatter and a status badge.
