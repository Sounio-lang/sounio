# stdlib/io

The root and direct modules expose checked owned regular-file I/O:
`NativeFile` from `io::native_file` and `BufferedReader` from
`io::buffered_reader`. They use an explicit Linux platform context and
caller-owned budgeted byte buffers. Existing pure/binary/scientific helpers
retain their direct modules and the existing mod facade.

Open accepts a byte-buffer path and mode 0 read,1 truncate/write or2 append.
Reads append; a successful positive-length read with count 0 means EOF.
Single writes expose short progress, and write-all loops with checked
ranges. Seek, sync and explicit close are available. Native failures use
POSIX errno values. Close files before closing their platform context.

BufferedReader owns its file and refill buffer. Its constructor checks a
positive refill size within the payload budget. Next returns a byte or
status 61 at EOF; offset counts bytes delivered to the caller, including
bytes consumed from prefetched data. Explicit close releases both owners.
It is a regular-file reader and does not promise asynchronous cancellation.

Compose BufferedReader with `data::csv_stream` for owned UTF-8 CSV rows or
`arrow::native_stream` for per-message Arrow IPC streaming. Their format
and per-batch limits remain separate from I/O refill size.

[The service smoke fixture](../../tests/stdlib/massive/test_service_root_exports.sio)
uses the public roots for actual file, CSV and JSON operations. See
[service receipts](../../docs/evidence/stdlib-wave3-services/README.md).
