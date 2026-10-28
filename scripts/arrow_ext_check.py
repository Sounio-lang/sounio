#!/usr/bin/env python3
"""Check canonical extension IPC written by test_arrow_ext.sio."""

import os
from pathlib import Path


def main() -> None:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        if os.environ.get("ARROW_REQUIRE_PYARROW") == "1":
            raise SystemExit("FAIL pyarrow required but not importable")
        print("EXT_PYARROW_SKIP")
        return

    def load(path: Path):
        with pa.memory_map(str(path), "r") as src:
            reader = ipc.open_stream(src)
            return reader.schema, reader.read_next_batch()

    schema, batch = load(Path("/tmp/sounio_arrow_ext.arrows"))
    if schema.field("id").type.extension_name != "arrow.uuid":
        raise SystemExit("FAIL uuid type")
    if schema.field("j").type.extension_name != "arrow.json":
        raise SystemExit("FAIL json type")
    if schema.field("ok").type.extension_name != "arrow.bool8":
        raise SystemExit("FAIL bool8 type")
    if schema.field("g").type.extension_name != "arrow.opaque":
        raise SystemExit("FAIL opaque type")
    if batch.column("ok").to_pylist() != [False, True]:
        raise SystemExit(f"FAIL bool8 {batch.column('ok').to_pylist()}")
    if batch.column("j").to_pylist() != ['{"a":1}', "null"]:
        raise SystemExit("FAIL json values")
    if str(batch.column("id")[0]) != "30313233-3435-3637-3839-616263646566":
        raise SystemExit(f"FAIL uuid {batch.column('id')[0]}")
    opaque = schema.field("g").type
    if opaque.type_name != "geometry" or opaque.vendor_name != "PostGIS":
        raise SystemExit(f"FAIL opaque params {opaque}")

    schema, batch = load(Path("/tmp/sounio_arrow_ext_tensor.arrows"))
    tensor = schema.field("t").type
    if tensor.extension_name != "arrow.fixed_shape_tensor":
        raise SystemExit(f"FAIL tensor {tensor}")
    if list(tensor.shape) != [2, 3]:
        raise SystemExit(f"FAIL shape {tensor.shape}")
    if list(tensor.permutation) != [1, 0]:
        raise SystemExit(f"FAIL permutation {tensor.permutation}")
    if list(tensor.dim_names) != ["a", "b"]:
        raise SystemExit(f"FAIL dim names {tensor.dim_names}")
    if batch.column("t").storage.to_pylist() != [[1, 2, 3, 4, 5, 6]]:
        raise SystemExit("FAIL tensor values")

    with pa.memory_map("/tmp/sounio_arrow_ext_escaped.arrows", "r") as src:
        escaped = ipc.open_stream(src).schema.field("t").type
    if list(escaped.dim_names) != ['a"b\n', "c\\d\t"]:
        raise SystemExit(f"FAIL escaped dim names {escaped.dim_names}")

    schema, _batch = load(Path("/tmp/sounio_arrow_ext_struct.arrows"))
    variant = schema.field("p").metadata
    if variant.get(b"ARROW:extension:name") != b"arrow.parquet.variant":
        raise SystemExit(f"FAIL variant {variant}")
    if variant.get(b"ARROW:extension:metadata") != b"":
        raise SystemExit("FAIL variant metadata")
    when = schema.field("when").metadata
    if when.get(b"ARROW:extension:name") != b"arrow.timestamp_with_offset":
        raise SystemExit(f"FAIL timestamp offset {when}")
    print("EXT_PYARROW_OK")


if __name__ == "__main__":
    main()
