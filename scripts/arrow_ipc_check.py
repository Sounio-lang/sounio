#!/usr/bin/env python3
"""Check Sounio Arrow IPC bytes, then compare with pyarrow when it imports."""

import sys
from pathlib import Path


def structural(path: Path, kind: str) -> None:
    data = path.read_bytes()
    if kind == "file":
        if not data.startswith(b"ARROW1") or not data.endswith(b"ARROW1"):
            raise SystemExit(f"FAIL magic {path}")
    else:
        if data[:4] != b"\xff\xff\xff\xff":
            raise SystemExit(f"FAIL continuation {path}")
        if not data.endswith(b"\xff\xff\xff\xff\x00\x00\x00\x00"):
            raise SystemExit(f"FAIL eos {path}")
    needles = [
        (bytes.fromhex("fdffffffffffffff"), "i64 -3"),
        (bytes.fromhex("000000000000f83f"), "f64 1.5"),
        (bytes.fromhex("00000000000002c0"), "f64 -2.25"),
        (b"abc", "utf8 abc"),
    ]
    for pat, name in needles:
        if pat not in data:
            raise SystemExit(f"FAIL missing {name} in {path}")


def try_pyarrow(file_path: Path, stream_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        print("PYARROW_SKIP")
        return

    expect = {
        "id": [1, 2, -3],
        "x": [1.5, None, -2.25],
        "ok": [True, False, None],
        "name": ["ab", None, "c"],
        "tiny": [-1, 0, 127],
        "code": [65535, 1, 256],
        "z": [None, None, None],
    }

    def load(path: Path, kind: str) -> None:
        with pa.memory_map(str(path), "r") as src:
            if kind == "file":
                reader = ipc.open_file(src)
            else:
                reader = ipc.open_stream(src)
            got = reader.read_all().to_pydict()
        if got != expect:
            raise SystemExit(f"FAIL pyarrow {path}: {got}")

    load(file_path, "file")
    load(stream_path, "stream")
    print("PYARROW_OK")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: arrow_ipc_check.py <file.arrow> <stream.arrows>")
    file_path = Path(sys.argv[1])
    stream_path = Path(sys.argv[2])
    structural(file_path, "file")
    structural(stream_path, "stream")
    print("STRUCT_OK")
    try_pyarrow(file_path, stream_path)


if __name__ == "__main__":
    main()
