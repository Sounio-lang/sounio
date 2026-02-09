# Standard Library Reference Entry Point

This page is the stable reference entrypoint linked from the repository README.

## Core References

- Full standard library inventory and metrics: `docs/STDLIB_REFERENCE.md`
- `Knowledge<T>` and uncertainty usage: `docs/reference/KNOWLEDGE_REFERENCE.md`
- Language specification: `spec/LANGUAGE_SPECIFICATION.md`

## API Doc Generation (`souniodoc`)

Generate docs from the repository root:

```bash
cargo run -p souc --bin souniodoc -- generate stdlib --output target/doc
```

This generates browsable API docs for stdlib modules.
