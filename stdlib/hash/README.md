# stdlib/hash

Single-shot cryptographic hash functions over `&RawBuf`-backed buffers.

## Functions
- `sha1(buf: &RawBuf, len: i64) -> [u8; 20]`
- `sha256(buf: &RawBuf, len: i64) -> [u8; 32]`
- `sha384(buf: &RawBuf, len: i64) -> [u8; 48]`
- `sha512(buf: &RawBuf, len: i64) -> [u8; 64]`

This module exists for the TLS/X.509 path's need to hash large `RawBuf`-backed buffers (e.g. a DER-encoded certificate, up to `DER_MAX_LENGTH` = 65536 bytes) in one shot. `stdlib/crypto/` (SHA-256 + HMAC-SHA256, incremental, fixed `[u8; 256]` buffers) predates this module and is not being replaced or retired by it — use `stdlib/crypto/` for incremental or keyed-hash needs over small buffers, and `stdlib/hash/` for one-shot hashing of large `RawBuf` data.
