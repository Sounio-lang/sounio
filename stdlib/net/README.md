# stdlib/net

Network address types + TCP/UDP config (pure data structures) + FFI stubs.

## Key Types
- `IPv4Addr`: 4-octet IPv4 address (loopback/private/any predicates, `to_u32`)
- `SocketAddr`: `IPv4Addr` + port
- `TcpConfig`: TCP connection configuration (port, backlog, nodelay, keepalive, timeout)
- `UdpConfig`: UDP socket configuration (port, broadcast, max packet size)

## Key Functions
- `ipv4_new(a, b, c, d)` / `ipv4_loopback()` / `ipv4_any()`: construct `IPv4Addr`
- `socket_addr_new(ip, port)`: construct `SocketAddr`
- `tcp_config_default()` / `udp_config_default()`: default configs

## FFI (`net::ffi::*`)

No real OS socket syscalls are wired up yet. `tcp_connect`/`udp_send` are stubs
that always report failure, mirroring `stdlib/distributed/ffi/wrapper.sio`.
`raw_socket_available()` reports `false`.

## Tests

`tests/stdlib/net/test_net_core.sio` (check-only, Madaros gate).

`tests/stdlib/net/test_addr_e2e.sio` is a pre-existing `run-pass` test that
hits a native multi-module IR-lowering wall (SIGSEGV) unrelated to this
module's logic — a known, out-of-scope compiler gap, not fixed here.
