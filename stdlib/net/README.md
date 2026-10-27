# stdlib/net

The checked native API implements Linux IPv4 TCP and UDP through an explicit
`ffi::platform::Platform` context. Its `NativeSocket`, `PollSet` and
`native_*` functions are available from the net root and their direct
`net::native` / `net::poll_set` modules.

`TcpConfig`, `UdpConfig` and address values remain pure configuration types.
The historical `net::ffi` wrappers keep their previous behavior:
`raw_socket_available()` is false, `tcp_connect` returns failure and
`udp_send` returns false. They are distinct APIs; no legacy stub is renamed
or advertised as a native socket.

## Native ownership and deadlines

Create a platform context and check its status. Close every native socket
and readiness set before closing the platform. Resource IDs are stable
logical handles and are never recycled into another resource. Socket tokens
are non-owning; their owners must remain alive for useful I/O.

TCP includes listen/connect/accept, nonblocking connect-start/finish,
readiness polling, checked buffer transfers and shutdown. IPv4 values use
the host-order integer representation (127.0.0.1 is 2130706433); port 0 requests
an ephemeral bound port. There is no DNS, TLS or IPv6 claim.

`native_socket_send_all_until` and `native_socket_read_exact_until` use an
absolute monotonic nanosecond deadline. Retries retain that deadline and
return completed byte counts on errors. Read-exact appends to the caller's
owned byte buffer; early EOF is 61 and timeout 110. Zero deadline polls
immediately. The older relative-timeout single-transfer functions retain
their names and behavior.

## Readiness and datagrams

PollSet registers socket interests 1/2 (read/write) with a caller tag.
Its payload budget is 24 bytes/entry; bounded native container/snapshot
overhead is separate. Wait polls all registrations against one absolute
deadline. No events returns count 0/status 0. Flags are readable 1, writable 2,
error 4, hangup 8 and closed/invalid16. A close during a blocked wait may be
observed only after an event or deadline. Serialize registration, wait and
event-field reads.

UDP destinations are explicit IPv4/port values. The sender getter reads one
atomic latest endpoint. `native_udp_receive_from` takes an exclusive typed
socket borrow across receive and sender retrieval. Preserve that exclusive
owner/serialization discipline; a separate receive/getter pair cannot
associate metadata with a packet under concurrent receivers.

Zero-length UDP payloads remain datagrams. Truncation consumes the datagram
and reports 90 with sender metadata, while the incomplete prefix is not
appended to the public buffer. Native status values are POSIX errno values.
The qualified runtime must provide platform ABI operations through 112.

The root-import native example is
[the service smoke fixture](../../tests/stdlib/massive/test_service_root_exports.sio).
Focused receipts and limits are recorded in
[service evidence](../../docs/evidence/stdlib-wave3-services/README.md).
