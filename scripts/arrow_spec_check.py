#!/usr/bin/env python3
"""Logical check of the spec IPC files written by test_arrow_spec.sio."""

import os
from decimal import Decimal
from pathlib import Path


def main() -> None:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        if os.environ.get("ARROW_REQUIRE_PYARROW") == "1":
            raise SystemExit("FAIL pyarrow required but not importable")
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

    schema, got = batches(Path("/tmp/sounio_arrow_ree.arrows"))
    if str(schema.field("r").type) != "run_end_encoded<run_ends: int32, values: int64>":
        raise SystemExit(f"FAIL ree type {schema.field('r').type}")
    if got[0].column("r").to_pylist() != [7, 7, 9, 9]:
        raise SystemExit(f"FAIL ree {got[0].column('r').to_pylist()}")

    schema, got = batches(Path("/tmp/sounio_arrow_scalars.arrows"))
    batch = got[0]

    def values(name: str) -> list:
        col = batch.column(name)
        out = []
        for i in range(len(col)):
            if not col[i].is_valid:
                out.append(None)
            else:
                out.append(col[i].value)
        return out

    if str(schema.field("d32").type) != "date32[day]" or values("d32") != [19000, None]:
        raise SystemExit(f"FAIL d32 {schema.field('d32').type} {values('d32')}")
    if str(schema.field("d64").type) != "date64[ms]" or values("d64") != [5, -9]:
        raise SystemExit(f"FAIL d64 {schema.field('d64').type} {values('d64')}")
    if str(schema.field("t32").type) != "time32[ms]" or values("t32") != [1000, -8]:
        raise SystemExit(f"FAIL t32 {schema.field('t32').type} {values('t32')}")
    if str(schema.field("t64").type) != "time64[ns]" or values("t64") != [7, 8]:
        raise SystemExit(f"FAIL t64 {schema.field('t64').type} {values('t64')}")
    if batch.column("dec64").to_pylist() != [Decimal("-0.42"), Decimal("0.42")]:
        raise SystemExit(f"FAIL dec64 {batch.column('dec64').to_pylist()}")
    if batch.column("dec128").to_pylist() != [
        Decimal("13610771366776895768476177004545508.5872"),
        Decimal("6405838452101818886974504219670769.8022"),
    ]:
        raise SystemExit(f"FAIL dec128 {batch.column('dec128').to_pylist()}")
    if batch.column("f32").to_pylist() != [-1.0, 1.0]:
        raise SystemExit(f"FAIL f32 {batch.column('f32').to_pylist()}")
    if str(schema.field("iym").type) != "month_interval":
        raise SystemExit(f"FAIL interval {schema.field('iym').type}")

    schema, got = batches(Path("/tmp/sounio_arrow_more.arrows"))
    batch = got[0]
    if batch.column("fb").to_pylist() != [b"xyz"]:
        raise SystemExit(f"FAIL fixed {batch.column('fb').to_pylist()}")
    if str(schema.field("m").type) != "map<int32, int64, keys_sorted>":
        raise SystemExit(f"FAIL map type {schema.field('m').type}")
    if batch.column("m").to_pylist() != [[(10, 100), (20, 200)]]:
        raise SystemExit(f"FAIL map {batch.column('m').to_pylist()}")
    if batch.column("lu").to_pylist() != ["pq"]:
        raise SystemExit(f"FAIL large utf8 {batch.column('lu').to_pylist()}")
    if batch.column("lv").to_pylist() != [[5, 6]]:
        raise SystemExit(f"FAIL list view {batch.column('lv').to_pylist()}")

    schema, got = batches(Path("/tmp/sounio_arrow_unions.arrows"))
    batch = got[0]
    if "sparse_union" not in str(schema.field("su").type):
        raise SystemExit(f"FAIL sparse type {schema.field('su').type}")
    if "dense_union" not in str(schema.field("du").type):
        raise SystemExit(f"FAIL dense type {schema.field('du').type}")
    if batch.column("su").to_pylist() != [4, 9] or batch.column("du").to_pylist() != [4, 9]:
        raise SystemExit(f"FAIL unions {batch.column('su').to_pylist()} {batch.column('du').to_pylist()}")
    if str(schema.field("ll").type) != "large_list<item: int16 not null>":
        raise SystemExit(f"FAIL large list {schema.field('ll').type}")
    if batch.column("ll").to_pylist() != [[99], [-5]]:
        raise SystemExit(f"FAIL large list values {batch.column('ll').to_pylist()}")
    print("SPEC_PYARROW_OK")


if __name__ == "__main__":
    main()
