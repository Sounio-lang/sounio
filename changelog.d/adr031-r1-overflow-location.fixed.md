The 64-bit integer overflow panic now also says where the operation is:
`sounio: integer overflow (i64 +) at path/to/file.sio:12:9` (ADR-031 R1).

The lexer registers each file it lexes (path and line starts) in a source
map in `lexer::span`, and the parser stamps the file's id into every Span.
Lowering records the span of each checked arithmetic node in a new
`IrInstr.src_site`, which the IR arena carries through load/store, copy,
swap and compaction. The x86 backend emits one stub per (kind, site) and
function, with the whole line inline behind a `call`, so it uses no rodata.
A site it cannot resolve (synthesized spans, sources lexed from memory, a
full table) gets the message without the location; it never names a wrong
one. The i32/u32 range checks and lean_single still print the message
without a location.
