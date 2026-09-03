#!/usr/bin/env python3
"""SIO1 -> pyarrow.Table (or numpy / struct) reader -- Campaign C4 Arrow bridge.

Reads the flat little-endian columnar binary emitted by
`data::arrow_bridge::df_write_sio1` (stdlib/data/arrow_bridge.sio) and
reconstructs the DataFrame's columns.

Graceful degradation (this sandbox has neither pyarrow nor numpy):
  - pyarrow present -> build a pyarrow.Table, one field per column.
  - else numpy present -> np.frombuffer(col_bytes, dtype='<f8' | '<i8').
  - else -> struct.unpack('<{n}d' | '<{n}q', col_bytes).

SIO1 layout (all little-endian):
  [0..4)   magic "SIO1"
  [4..8)   version i32
  [8..16)  n_rows i64
  [16..24) n_cols i64
  next 8*ncols  col name_ids  i64[]
  next 8*ncols  dtype tags    i64[]  (0 = f64, 1 = i64)
  next 8*nrows*ncols  column-major data i64/f64 bits

Usage:
  arrow_bridge_bridge.py <file.sio1>          # human-readable dump
  arrow_bridge_bridge.py <file.sio1> --cells  # machine-readable PYCELL lines
                                              # + raw i64 bits for the
                                              # bit-exact gate cross-check.
"""
import struct
import sys

MAGIC = b"SIO1"

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pyarrow as _pa
except Exception:
    _pa = None


def read_sio1(path):
    """Parse a SIO1 file. Returns (meta, columns, bits_by_col).

    meta = dict(version, n_rows, n_cols, name_ids, dtypes)
    columns = list of per-column sequences (numpy arrays, or tuples for the
    pure-struct fallback). bits_by_col holds the raw per-cell i64 bit patterns
    for the exact gate check.
    """
    with open(path, "rb") as fh:
        data = fh.read()

    if data[0:4] != MAGIC:
        raise ValueError("bad magic: %r (expected %r)" % (data[0:4], MAGIC))

    (version,) = struct.unpack_from("<i", data, 4)
    (n_rows,) = struct.unpack_from("<q", data, 8)
    (n_cols,) = struct.unpack_from("<q", data, 16)

    off = 24
    name_ids = list(struct.unpack_from("<%dq" % n_cols, data, off)) if n_cols else []
    off += 8 * n_cols
    dtypes = list(struct.unpack_from("<%dq" % n_cols, data, off)) if n_cols else []
    off += 8 * n_cols

    columns = []
    bits_by_col = []  # raw i64 bit patterns, exactly as Sounio wrote them
    for c in range(n_cols):
        span = data[off : off + 8 * n_rows]
        off += 8 * n_rows
        tag = dtypes[c] if c < len(dtypes) else 0
        fmt_bits = "<%dq" % n_rows
        bits = list(struct.unpack(fmt_bits, span)) if n_rows else []
        bits_by_col.append(bits)

        if tag == 1:
            # int64 column
            if _np is not None:
                col = _np.frombuffer(span, dtype="<i8")
            else:
                col = struct.unpack("<%dq" % n_rows, span) if n_rows else ()
        else:
            # f64 column (v1 default)
            if _np is not None:
                col = _np.frombuffer(span, dtype="<f8")
            else:
                col = struct.unpack("<%dd" % n_rows, span) if n_rows else ()
        columns.append(col)

    meta = dict(
        version=version,
        n_rows=n_rows,
        n_cols=n_cols,
        name_ids=name_ids,
        dtypes=dtypes,
    )
    return meta, columns, bits_by_col


def to_arrow_table(meta, columns):
    """Build a pyarrow.Table if pyarrow is available, else return None."""
    if _pa is None:
        return None
    fields = {}
    for i in range(meta["n_cols"]):
        col = columns[i]
        fields["col_%d" % i] = _pa.array(list(col))
    return _pa.table(fields)


def main(argv):
    if len(argv) < 2:
        print("usage: arrow_bridge_bridge.py <file.sio1> [--cells]", file=sys.stderr)
        return 2
    path = argv[1]
    cells_mode = "--cells" in argv[2:]

    meta, columns, bits_by_col = read_sio1(path)

    if cells_mode:
        # Machine-readable form the gate parses. One line per cell, in
        # column-major order, matching the Sounio driver's CELL lines.
        print("PYMETA n_rows=%d n_cols=%d version=%d"
              % (meta["n_rows"], meta["n_cols"], meta["version"]))
        print("PYNAMES " + " ".join(str(x) for x in meta["name_ids"]))
        print("PYDTYPES " + " ".join(str(x) for x in meta["dtypes"]))
        for c in range(meta["n_cols"]):
            for r in range(meta["n_rows"]):
                val = columns[c][r]
                bits = bits_by_col[c][r]
                # PYCELL <col> <row> <exact-i64-bits> <reconstructed-float>
                print("PYCELL %d %d %d %r" % (c, r, bits, float(val)))
        return 0

    # Human-readable dump.
    backend = "pyarrow" if _pa else ("numpy" if _np else "struct")
    print("SIO1 file: %s" % path)
    print("backend  : %s" % backend)
    print("version  : %d" % meta["version"])
    print("n_rows   : %d" % meta["n_rows"])
    print("n_cols   : %d" % meta["n_cols"])
    print("name_ids : %s" % meta["name_ids"])
    print("dtypes   : %s  (0=f64, 1=i64)" % meta["dtypes"])

    table = to_arrow_table(meta, columns)
    if table is not None:
        print("--- pyarrow.Table ---")
        print(table)
    else:
        print("--- columns (%s reconstruction) ---" % backend)
        for i in range(meta["n_cols"]):
            vals = list(columns[i])
            print("col_%d (name_id=%s): %s"
                  % (i, meta["name_ids"][i] if i < len(meta["name_ids"]) else "?", vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
