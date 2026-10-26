#!/usr/bin/env python3
"""Logical check of the spec IPC files written by test_arrow_spec.sio."""

import sys
from pathlib import Path


def main() -> None:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        print("SPEC_PYARROW_SKIP")
        return

    def batches(path: Path):
        with pa.memory_map(str(path), "r") as src:
            reader = ipc.open_stream(src)
            out = []
            try:
                while True:
                    out.append(reader.read_next_batch())
            except StopIteration:
                return reader.schema, out

    schema, got = batches(Path("/tmp/sounio_arrow_nested.arrows"))
    if schema.metadata.get(b"app:ns") != b"s":
        raise SystemExit(f"FAIL schema metadata {schema.metadata}")
    ext = schema.field("col2").metadata
    if ext.get(b"ARROW:extension:name") != b"myorg.word":
        raise SystemExit(f"FAIL extension {ext}")
    if ext.get(b"ARROW:extension:metadata") != b"v1":
        raise SystemExit(f"FAIL extension metadata {ext}")
    col1 = got[0].column("col1").to_pylist()
    col2 = got[0].column("col2").to_pylist()
    if col1 != [{"a": 1, "b": [10, 11], "c": 1.5}, {"a": None, "b": [12], "c": None}]:
        raise SystemExit(f"FAIL nested {col1}")
    if col2 != ["ab", None]:
        raise SystemExit(f"FAIL utf8 {col2}")

    schema, got = batches(Path("/tmp/sounio_arrow_dict.arrows"))
    if [b.column(0).to_pylist() for b in got] != [["A", "B", "C", "B"], ["D", "C", "E", "A"]]:
        raise SystemExit("FAIL dictionary values")

    for name in ("lz4", "zstd"):
        schema, got = batches(Path(f"/tmp/sounio_arrow_{name}.arrows"))
        batch = got[0]
        if batch.column("v").to_pylist() != ["hi", "abcdefghijklm"]:
            raise SystemExit(f"FAIL {name} view")
        if batch.column("id").to_pylist() != [1, 2]:
            raise SystemExit(f"FAIL {name} id")
        if [s.value for s in batch.column("ts")] != [7, 8]:
            raise SystemExit(f"FAIL {name} timestamp")
        if str(schema.field("ts").type) != "timestamp[ns, tz=UTC]":
            raise SystemExit(f"FAIL {name} tz {schema.field('ts').type}")
    print("SPEC_PYARROW_OK")


if __name__ == "__main__":
    main()
