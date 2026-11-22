#!/usr/bin/env python3
"""Check Sounio Arrow IPC bytes, then compare with pyarrow when it imports."""

import os
import sys
from pathlib import Path


def file_eos_before_footer(data: bytes) -> bool:
    """Spec file layout: embedded stream EOS, then the footer flatbuffer."""
    if len(data) < 18 or not data.startswith(b"ARROW1") or not data.endswith(b"ARROW1"):
        return False
    fsize = int.from_bytes(data[-10:-6], "little", signed=True)
    if fsize <= 0:
        return False
    eos_at = len(data) - 10 - fsize - 8
    if eos_at < 8:
        return False
    return data[eos_at : eos_at + 8] == b"\xff\xff\xff\xff\x00\x00\x00\x00"


def structural(path: Path, kind: str) -> None:
    data = path.read_bytes()
    if kind == "file":
        if not data.startswith(b"ARROW1") or not data.endswith(b"ARROW1"):
            raise SystemExit(f"FAIL magic {path}")
        if not file_eos_before_footer(data):
            raise SystemExit(f"FAIL eos before footer {path}")
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


def framing(path: Path, kind: str) -> None:
    data = path.read_bytes()
    if kind == "file":
        if not data.startswith(b"ARROW1") or not data.endswith(b"ARROW1"):
            raise SystemExit(f"FAIL magic {path}")
        if not file_eos_before_footer(data):
            raise SystemExit(f"FAIL eos before footer {path}")
    else:
        if data[:4] != b"\xff\xff\xff\xff":
            raise SystemExit(f"FAIL continuation {path}")
        if not data.endswith(b"\xff\xff\xff\xff\x00\x00\x00\x00"):
            raise SystemExit(f"FAIL eos {path}")


def check_multi(ipc, file_path: Path, stream_path: Path) -> None:
    import pyarrow as pa

    expect = [[1, 2], [30]]

    def batches(path: Path, kind: str) -> list:
        with pa.memory_map(str(path), "r") as src:
            if kind == "file":
                reader = ipc.open_file(src)
                return [reader.get_batch(i).column(0).to_pylist() for i in range(reader.num_record_batches)]
            reader = ipc.open_stream(src)
            out = []
            try:
                while True:
                    out.append(reader.read_next_batch().column(0).to_pylist())
            except StopIteration:
                return out

    for path, kind in ((file_path, "file"), (stream_path, "stream")):
        got = batches(path, kind)
        if got != expect:
            raise SystemExit(f"FAIL pyarrow multi {path}: {got}")


def try_pyarrow(file_path: Path, stream_path: Path, multi: tuple[Path, Path] | None) -> None:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        if os.environ.get("ARROW_REQUIRE_PYARROW") == "1":
            raise SystemExit("FAIL pyarrow required but not importable")
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
    if multi is not None:
        check_multi(ipc, multi[0], multi[1])
    print("PYARROW_OK")


def main() -> None:
    if len(sys.argv) not in (3, 5):
        raise SystemExit(
            "usage: arrow_ipc_check.py <file.arrow> <stream.arrows> [<multi.arrow> <multi.arrows>]"
        )
    file_path = Path(sys.argv[1])
    stream_path = Path(sys.argv[2])
    structural(file_path, "file")
    structural(stream_path, "stream")
    multi = None
    if len(sys.argv) == 5:
        multi_file = Path(sys.argv[3])
        multi_stream = Path(sys.argv[4])
        framing(multi_file, "file")
        framing(multi_stream, "stream")
        multi = (multi_file, multi_stream)
    print("STRUCT_OK")
    try_pyarrow(file_path, stream_path, multi)


if __name__ == "__main__":
    main()
