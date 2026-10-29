# Type checking navigation (repo-local)

## Key directories

- Main checker: `self-hosted/check/`
- Parser: `self-hosted/parser/`

## Typical edit flows

- Add/adjust a type rule → update `self-hosted/check/`
- Tighten a rule → add a `tests/ui/` case
