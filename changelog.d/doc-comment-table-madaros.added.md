Added (Madaros half of ADR-015 row 12): documentation comments are recorded with the token that follows them. `///` (not `////`), `//!`, `/** */` (not `/**/` or `/***`) and `/*! */` go into a side table in the lexer, in source order, each with its kind, line, text range and the index of the next token. The token stream the parser reads does not change, so a doc comment in any position (struct fields, enum variants, before end of file) cannot break a parse.

`madaros --emit-docs FILE` lists the table, one line per comment: `DOC <outer|inner>-<line|block> line=<L> next_line=<NL> next=<first token of the item> text=<text>`, with newlines in the text written as `\n`; `next=<eof>` when nothing follows. Inner comments (`//!`, `/*!`) document the enclosing item; `next` only gives their position. The table holds 65536 comments per file and `--emit-docs` says so if it overflows.

Pins: `scripts/ci/doc_comment_tokens_gate.sh` with `tests/doc_comments/basic.sio` and `basic.expected` (9 comments, plus the ordinary comments that must not be listed). The gate fails on a Madaros without the table. It is not wired into CI yet.

lean_single half: separate change.
