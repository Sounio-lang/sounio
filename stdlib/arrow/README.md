# stdlib/arrow

Native Apache Arrow IPC, written in Sounio. Metadata is FlatBuffers
(`org.apache.arrow.flatbuf`, metadata versions V4 and V5). The stream and
file encapsulations match the Arrow columnar IPC format that pyarrow reads.

This is not the Arrow C Data Interface. `dataframe::arrow_available` stays
false; that hook is a separate FFI surface.

## Caps

Fail closed (status `-7`) rather than truncate:

- 8 fields, 16 rows per batch
- 8 record-batch blocks in one file; a stream holds as many as fit in the buffer
- field names and string/binary cells at most 32 bytes
- IPC buffer 8192 bytes, record-batch body 4096 bytes

`arrow_ipc_read` reads one batch and returns `-7` when another batch is present.
`arrow_ipc_open` then `arrow_ipc_next` walk every batch (`0` = a batch, `1` = end).
`arrow_ipc_file_begin` / `arrow_ipc_file_push` / `arrow_ipc_file_end` and the
matching `arrow_ipc_stream_*` calls write those batches.

## Types

Null, Int (8/16/32/64, signed or unsigned), FloatingPoint (half and single
are raw bits in the low part of the cell; double is IEEE-754), Binary, Utf8,
Bool. Dictionaries, nested children, compression, and big-endian schemas
return `-4`.

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
