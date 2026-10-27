# stdlib/json

The owned API is available from the json root and direct modules:
`json::dom` provides JsonDom/JsonNode, checked parsing, exact token access,
decoded string/key access and explicit close. `json::writer` provides an
owned incremental regular-file writer.

The older flat JsonDoc/JsonBuf parser retains its `json::parser` path and
existing mod exports. It is a separate representation and is not renamed
to JsonDom.

## Owned DOM and writer

DOM parsing charges the input copy and node metadata to its explicit
payload budget and applies the supplied nesting limit. Exact number tokens
survive without implicit floating conversion. Check status before reading
nodes; close the owner explicitly.

The writer owns a NativeFile, a bounded output chunk and a depth stack.
Use begin/end object or array, key, UTF-8 string, i64, bool, null and exact
number-token operations. Finish validates one complete root and flushes;
close consumes the writer and file. Errors are sticky and can leave a
partial output file. Closing an unfinished writer reports 22.

Keys and strings validate complete UTF-8, escape NUL/control bytes and
preserve byte identity without normalization. NaN and Infinity are not JSON
number tokens. Number tokens are checked by the owned parser and emitted
unchanged; this API does not imply a lossy or round-trip f64 formatter.

[The root-import smoke fixture](../../tests/stdlib/massive/test_service_root_exports.sio)
writes a file and reads its value through the owned DOM. Detailed
[focused receipts](../../docs/evidence/stdlib-wave3-services/README.md)
include chunk-boundary, ownership, malformed-input and ordering controls.
