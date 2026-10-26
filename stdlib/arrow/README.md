# stdlib/arrow

Native Apache Arrow IPC, written in Sounio. Metadata is FlatBuffers
(`org.apache.arrow.flatbuf`, metadata versions V4 and V5). The stream and
file encapsulations match the Arrow columnar IPC format that pyarrow reads.

This is not the Arrow C Data Interface. `dataframe::arrow_available` stays
false; that hook is a separate FFI surface.

## Caps

Fail closed (status `-7`) rather than truncate:

- 8 flattened field nodes (roots and children together), 16 slots per node
- 16 rows at the root of a batch; a list child holds at most 16 values
- 8 record-batch blocks and 4 dictionary blocks in one file
- 4 dictionaries, 16 values each, value type has no children
- 4 custom-metadata pairs at each of field, schema, and message; keys ≤ 32 bytes, values ≤ 48
- field names, time zones, and string/binary cells at most 32 bytes
- IPC buffer 8192 bytes, record-batch body 4096 bytes

`arrow_ipc_read` reads one batch and returns `-7` when another batch is present.
`arrow_ipc_open` then `arrow_ipc_next` walk every batch (`0` = a batch, `1` = end).
`arrow_ipc_open_ex` / `arrow_ipc_next_ex` also load dictionary batches.
`arrow_ipc_file_begin` / `arrow_ipc_file_push` / `arrow_ipc_file_end` and the
matching `arrow_ipc_stream_*` calls write those batches. `*_push_dict` writes
a `DictionaryBatch`.

## Types

Every built-in type in `Schema.fbs`: null, int, floating point, binary, utf8,
bool, decimal, date, time, timestamp, interval, list, struct, union, fixed-size
binary, fixed-size list, map, duration, large binary, large utf8, large list,
run-end encoded, binary view, utf8 view, list view, and large list view.
Half and single floats stay raw bits. Decimal and interval values wider than
8 bytes live in the string cell. Nested fields are a pre-order depth-first
flatten: each node has a length and null count, and each buffer's `length`
is the logical size, not the padding.

A field may carry a dictionary id. The record batch then stores index buffers.
Dictionary value batches are separate messages (`isDelta` appends; a second
non-delta for the same id is replacement). A file rejects replacement. A
stream that replaces a dictionary must set feature `DICTIONARY_REPLACEMENT`
on the schema before `stream_begin`.

Custom metadata round-trips on the field, the schema, and the schema message.
The file footer copies the schema message metadata. Extension types are the
keys `ARROW:extension:name` and `ARROW:extension:metadata`.

## Canonical extensions

`arrow::ext` builds the [canonical extension types](https://arrow.apache.org/docs/format/CanonicalExtensions.html). Each setter checks the storage layout, then writes the extension name and the metadata string. `arrow_ext_check` reads that metadata back and checks it against the storage.

| Name | Storage |
| --- | --- |
| `arrow.uuid` | `FixedSizeBinary(16)`, empty metadata |
| `arrow.json` | Utf8, LargeUtf8, or Utf8View; empty metadata or `{}` |
| `arrow.bool8` | signed `Int8`; `0` is false and any other value is true |
| `arrow.opaque` | any storage; metadata `{"type_name":"...","vendor_name":"..."}` |
| `arrow.fixed_shape_tensor` | `FixedSizeList` whose size is the product of `shape` |
| `arrow.variable_shape_tensor` | struct fields `data` (a list) and `shape` (a fixed-size list) |
| `arrow.parquet.variant` | struct with non-null binary `metadata` and `value` and/or `typed_value` |
| `arrow.timestamp_with_offset` | struct `timestamp` (UTC) then `offset_minutes` (`Int16`) |

Tensor metadata uses the same compact JSON pyarrow writes (`shape`, then `permutation`, then `dim_names`). At most 4 dimensions. An extension name outside this list is kept and `arrow_ext_check` accepts it.

`arrow_batch_set_codec` selects lz4-frame or zstd body compression. Set
feature `COMPRESSED_BODY` before the schema message is written. Each buffer
is compressed on its own and prefixed with the uncompressed length, or with
`-1` when the payload is left uncompressed. Zstd decode accepts raw blocks,
RLE blocks, and compressed blocks whose literals are raw or RLE and whose
sequence tables are predefined or RLE. Huffman literals and FSE-compressed
sequence tables return `-4`. Big-endian schemas return `-4`.

## Status

`0` ok, `-1` overflow, `-2` bad magic, `-3` bad framing, `-4` unsupported,
`-5` truncated, `-6` malformed, `-7` over capacity, `-8` shape mismatch.

## Example

```sounio
use arrow::lib::{
    arrow_batch_new, arrow_batch_set_i64, arrow_batch_shape, arrow_buf_new,
    arrow_ipc_read, arrow_ipc_write_stream, arrow_schema_add, arrow_schema_new,
    arrow_type_int,
}

fn main() -> i32 with Mut, Panic, Div {
    var schema = arrow_schema_new()
    let _a = arrow_schema_add(&!schema, "id", 0, arrow_type_int(), 64, 1, 0)
    var batch = arrow_batch_new()
    let _b = arrow_batch_shape(&!batch, 1, 1)
    let _c = arrow_batch_set_i64(&!batch, 0, 0, 7, 0)
    var buf = arrow_buf_new()
    let _w = arrow_ipc_write_stream(&schema, &batch, &!buf)
    var schema2 = arrow_schema_new()
    var batch2 = arrow_batch_new()
    let rc = arrow_ipc_read(&buf, &!schema2, &!batch2)
    if rc != 0 { return 1 }
    0
}
```

`arrow_ipc_write_file` adds the `ARROW1` magic and footer. The embedded
stream ends with the spec EOS marker (`0xFFFFFFFF` then metadata length 0)
before that footer, the same marker `arrow_ipc_stream_end` writes.
`arrow_ipc_read` accepts either form. `arrow_ipc_write_path` writes the
file form with `write_file`.

## DoPut and DoExchange

`arrow::flight` wraps those encapsulated messages as Arrow Flight data
frames. The layout of each message is the [IPC encapsulated format](https://arrow.apache.org/docs/format/Columnar.html#format-ipc):
continuation `0xFFFFFFFF`, `metadata_size` (the FlatBuffer `Message` plus
padding), the metadata, then the body. A schema message has no body.
End-of-stream is not a Flight frame.

`do_put_write` turns an IPC stream into a DoPut client stream: the first
`FlightData` carries the descriptor and the schema `Message`, and each
following frame is one record batch (`data_header` plus `data_body`).
`put_result_write` is the DoPut reply. `do_exchange_write` is the same
`FlightData` sequence in either direction; a reply omits the descriptor,
and an empty IPC buffer with a descriptor is a command-only open.

Frames are uncompressed gRPC messages (one flag byte, a big-endian length,
then the protobuf). This is not an HTTP/2 server. At most 9 frames. A
compressed flag returns `-4`.
