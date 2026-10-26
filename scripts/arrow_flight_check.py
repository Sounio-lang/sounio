#!/usr/bin/env python3
"""Check DoPut / DoExchange frames and, when pyarrow imports, the IPC inside them.

Payload layout: https://arrow.apache.org/docs/format/Columnar.html#format-ipc
Each frame is an uncompressed gRPC message (flag 0, big-endian length) around
a FlightData or PutResult protobuf. data_header is the IPC Message metadata
(metadata_size bytes, padding included). data_body is the message body.
"""

import os
import struct
import sys
from pathlib import Path


def frames(data: bytes) -> list[bytes]:
    out = []
    i = 0
    while i < len(data):
        if i + 5 > len(data):
            raise SystemExit(f"FAIL truncated frame at {i}")
        if data[i] != 0:
            raise SystemExit(f"FAIL compressed flag {data[i]}")
        n = int.from_bytes(data[i + 1 : i + 5], "big")
        if i + 5 + n > len(data):
            raise SystemExit("FAIL frame length past end")
        out.append(data[i + 5 : i + 5 + n])
        i += 5 + n
    return out


def parse_protobuf(buf: bytes) -> list[tuple[int, object]]:
    fields = []
    i = 0
    while i < len(buf):
        tag, i = _varint(buf, i)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            val, i = _varint_value(buf, i)
            fields.append((field, val))
        elif wire == 2:
            ln, i = _varint(buf, i)
            if ln < 0 or i + ln > len(buf):
                raise SystemExit("FAIL protobuf length past end")
            fields.append((field, buf[i : i + ln]))
            i += ln
        elif wire == 1:
            if i + 8 > len(buf):
                raise SystemExit("FAIL protobuf length past end")
            i += 8
        elif wire == 5:
            if i + 4 > len(buf):
                raise SystemExit("FAIL protobuf length past end")
            i += 4
        else:
            raise SystemExit(f"FAIL protobuf wire {wire}")
    return fields


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    val = 0
    while True:
        if i >= len(buf) or shift > 28:
            raise SystemExit("FAIL bad varint")
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if b < 128:
            return val, i
        shift += 7


def _varint_value(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    val = 0
    for n in range(10):
        if i >= len(buf):
            raise SystemExit("FAIL bad varint")
        b = buf[i]
        i += 1
        if n == 9 and b > 1:
            raise SystemExit("FAIL bad varint")
        val |= (b & 0x7F) << shift
        if b < 128:
            return val, i
        if n == 9:
            raise SystemExit("FAIL bad varint")
        shift += 7
    raise SystemExit("FAIL bad varint")


def flight_parts(frame: bytes) -> dict:
    parts = {"desc": None, "header": None, "meta": None, "body": None}
    for field, val in parse_protobuf(frame):
        if field == 1 and isinstance(val, bytes):
            parts["desc"] = val
        elif field == 2 and isinstance(val, bytes):
            parts["header"] = val
        elif field == 3 and isinstance(val, bytes):
            parts["meta"] = val
        elif field == 1000 and isinstance(val, bytes):
            parts["body"] = val
    return parts


def to_ipc(parts_list: list[dict]) -> bytes:
    blob = b""
    for part in parts_list:
        header = part["header"]
        if not header:
            if part["body"]:
                raise SystemExit("FAIL empty header with a body")
            continue
        body = part["body"] if part["body"] is not None else b""
        if len(body) % 8 != 0:
            raise SystemExit("FAIL body not padded to 8")
        pad = (8 - (len(header) % 8)) % 8
        blob += struct.pack("<ii", -1, len(header) + pad)
        blob += header + bytes(pad) + body
    if blob:
        blob += struct.pack("<ii", -1, 0)
    return blob


def check_put(path: Path) -> bytes:
    data = path.read_bytes()
    got = [flight_parts(frame) for frame in frames(data)]
    if len(got) != 3:
        raise SystemExit(f"FAIL do_put frames {len(got)}")
    desc = got[0]["desc"]
    if desc != bytes.fromhex("08011a04646174611a026964"):
        raise SystemExit(f"FAIL descriptor {desc.hex() if desc else None}")
    if got[1]["desc"] is not None or got[2]["desc"] is not None:
        raise SystemExit("FAIL descriptor on a later DoPut frame")
    if got[0]["body"] is not None:
        raise SystemExit("FAIL schema frame has a body")
    ipc = to_ipc(got)
    if not ipc.startswith(b"\xff\xff\xff\xff"):
        raise SystemExit("FAIL ipc continuation")
    return ipc


def check_results(path: Path) -> None:
    got = frames(path.read_bytes())
    if got != [bytes.fromhex("0a026f6b"), b""]:
        raise SystemExit(f"FAIL put results {[g.hex() for g in got]}")


def check_exchange(client: Path, server: Path) -> tuple[bytes, bytes]:
    cframes = [flight_parts(frame) for frame in frames(client.read_bytes())]
    if len(cframes) != 1 or cframes[0]["desc"] != bytes.fromhex("080212057175657279"):
        raise SystemExit("FAIL exchange client descriptor")
    sframes = [flight_parts(frame) for frame in frames(server.read_bytes())]
    if len(sframes) != 2 or sframes[0]["desc"] is not None or sframes[1]["desc"] is not None:
        raise SystemExit("FAIL exchange server descriptor")
    return to_ipc(cframes), to_ipc(sframes)


def try_pyarrow(put_ipc: bytes, client_ipc: bytes, server_ipc: bytes) -> None:
    try:
        import pyarrow.flight as flight
        import pyarrow.ipc as ipc
    except ImportError:
        if os.environ.get("ARROW_REQUIRE_PYARROW") == "1":
            raise SystemExit("FAIL pyarrow required but not importable")
        print("PYARROW_SKIP")
        return
    desc = flight.FlightDescriptor.deserialize(bytes.fromhex("08011a04646174611a026964"))
    if desc.path != [b"data", b"id"]:
        raise SystemExit(f"FAIL pyarrow path {desc.path}")
    cmd = flight.FlightDescriptor.deserialize(bytes.fromhex("080212057175657279"))
    if cmd.command != b"query":
        raise SystemExit(f"FAIL pyarrow command {cmd.command}")

    def batches(blob: bytes) -> list:
        reader = ipc.open_stream(blob)
        out = []
        try:
            while True:
                out.append(reader.read_next_batch().column(0).to_pylist())
        except StopIteration:
            return out

    if batches(put_ipc) != [[1, 2], [30]]:
        raise SystemExit(f"FAIL pyarrow do_put {batches(put_ipc)}")
    if batches(client_ipc) != []:
        raise SystemExit("FAIL pyarrow exchange client batches")
    if batches(server_ipc) != [[7]]:
        raise SystemExit(f"FAIL pyarrow exchange server {batches(server_ipc)}")
    print("PYARROW_OK")


def _reject_short(buf: bytes) -> None:
    try:
        parse_protobuf(buf)
    except SystemExit as exc:
        if "protobuf length past end" not in str(exc):
            raise
        return
    raise SystemExit("FAIL truncated protobuf accepted")


def _reject_truncated_protobuf() -> None:
    _reject_short(b"\x12\x08abc")
    _reject_short(b"\x09abc")
    _reject_short(b"\x0dabc")


def _varint_limits() -> None:
    _val, end = _varint_value(bytes.fromhex("808080808001"), 0)
    if end != 6:
        raise SystemExit("FAIL six-byte varint")
    try:
        _varint(bytes.fromhex("808080808001"), 0)
    except SystemExit as exc:
        if "bad varint" not in str(exc):
            raise
    else:
        raise SystemExit("FAIL long tag accepted")
    try:
        _varint_value(bytes.fromhex("80808080808080808002"), 0)
    except SystemExit as exc:
        if "bad varint" not in str(exc):
            raise
    else:
        raise SystemExit("FAIL varint overflow accepted")
    _ten, end = _varint_value(bytes.fromhex("80808080808080808001"), 0)
    if end != 10:
        raise SystemExit("FAIL ten-byte varint")


def _empty_header_ipc() -> None:
    if to_ipc([{"header": b"", "body": None}]) != b"":
        raise SystemExit("FAIL empty header emitted eos")
    if to_ipc([{"header": None, "body": None}]) != b"":
        raise SystemExit("FAIL absent header emitted eos")


def main() -> None:
    _reject_truncated_protobuf()
    _varint_limits()
    _empty_header_ipc()
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: arrow_flight_check.py <do_put> <put_result> <exchange_client> <exchange_server>"
        )
    put_ipc = check_put(Path(sys.argv[1]))
    check_results(Path(sys.argv[2]))
    client_ipc, server_ipc = check_exchange(Path(sys.argv[3]), Path(sys.argv[4]))
    print("STRUCT_OK")
    try_pyarrow(put_ipc, client_ipc, server_ipc)


if __name__ == "__main__":
    main()
