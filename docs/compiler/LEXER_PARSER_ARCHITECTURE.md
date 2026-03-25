<!-- docs:meta
topic_id: website.docs.compiler.lexer-parser
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler.lexer-parser
-->

# Sounio Lexer and Parser Architecture

The lexer and parser should now be described through the self-hosted tree, not through older Rust crate paths. The active implementation is split into many focused modules that separate tokenization, AST construction, expression parsing, item parsing, statement parsing, and recovery support.

## Current source map

Lexer-facing files:

- `self-hosted/lexer/cursor.sio`
- `self-hosted/lexer/reader.sio`
- `self-hosted/lexer/token.sio`
- `self-hosted/lexer/tables.sio`
- `self-hosted/lexer/numparse.sio`

Parser-facing files:

- `self-hosted/parser/ast.sio`
- `self-hosted/parser/exprs.sio`
- `self-hosted/parser/items.sio`
- `self-hosted/parser/stmts.sio`
- `self-hosted/parser/patterns.sio`
- `self-hosted/parser/types.sio`

Driver-facing wrappers:

- `self-hosted/compiler/lexer.sio`
- `self-hosted/compiler/parser.sio`

## What is stable to document

- the repo contains a real lexer and parser written in Sounio itself
- the implementation is intentionally decomposed by concern
- the safest syntax claims are the ones backed by current fixtures and `souc check`

## How to verify syntax claims

```bash
export SOUC_BIN="$(pwd)/bin/souc"
"$SOUC_BIN" check examples/hello.sio
```

Then move to more targeted run-pass or compile-fail fixtures for the syntax area you are documenting.

## Documentation rules

- Do not treat `crates/souc/src/lexer` or `crates/souc/src/parser` as the primary current explanation of syntax handling.
- Use small checked examples for public-facing grammar explanations.
- Treat recovery details, edge-case parsing behavior, and unsupported corners conservatively unless you have current fixtures for them.
