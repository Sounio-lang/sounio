<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-08-25-madaros-tls13-handshake-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-08-25-madaros-tls13-handshake-design
-->

# Madaros TLS 1.3 Handshake + Record Layer Design

## Purpose

This is the final sub-project of the TLS-on-Madaros effort. It ties together
everything already built and merged on this branch — X.509 chain validation
(`stdlib/x509/chain.sio`), AEAD ciphers (`stdlib/crypto/aead.sio`), X25519 key
exchange (`stdlib/crypto/x25519.sio`), and the HKDF/TLS 1.3 key schedule
(`stdlib/crypto/hkdf.sio`) — into a working TLS 1.3 client that can complete a
real handshake against a real, unmodified modern server (nginx/Caddy
defaults) and exchange encrypted HTTP/1.1 traffic. This unblocks the original
goal: an HTTPS-capable web-search tool for the Conclave chat app.

Scope is **TLS 1.3 only**. TLS 1.2 fallback (its own separate PRF, RFC 5246
§5, an entirely different construction from TLS 1.3's HKDF-based schedule)
is explicitly deferred to a future sub-project, per the earlier "TLS 1.3
primeiro" decision.

Role is **TLS client only** — dialing out, never accepting inbound
connections. The design keeps role-agnostic pieces (record-layer framing,
most of transcript-hash accumulation, the key schedule) structured so a
future TLS *server* sub-project could reuse them, but this sub-project builds
and tests only the client path.

## Scope Decisions (already made, do not re-litigate)

- **Signature schemes**: this sub-project adds both **RSA-PSS**
  (`rsa_pss_rsae_sha256/384/512`) and **ECDSA P-256**
  (`ecdsa_secp256r1_sha256`) verification, since TLS 1.3 §4.2.3 forbids the
  already-implemented plain RSASSA-PKCS1-v1.5 in `CertificateVerify` even for
  RSA certificates, and a real, unmodified server could present either
  scheme.
- **Top-level interface**: a lower-level `TlsConnection` handle
  (`tls_connect`/`tls_send`/`tls_recv`/`tls_close`), mirroring
  `stdlib/net/socket.sio`'s `TcpSocket`/`tcp_connect`/`tcp_send`/`tcp_recv`/
  `tcp_close` pattern — not a one-shot `https_get()` mirroring
  `http_client.sio`'s `http_get()`. A future sub-project (or the Conclave
  integration itself) is expected to drive HTTP/1.1 request/response framing
  over this connection, the way `http_client.sio` already does over
  `TcpSocket`.
- **ClientHello extensions**: `server_name` (SNI), `application_layer_
  protocol_negotiation` = `"http/1.1"`, `supported_versions` = TLS 1.3 only,
  `key_share` = X25519 only, `signature_algorithms` = the RSA-PSS and ECDSA
  schemes above, `supported_groups` = X25519 only.
- **Post-handshake records**: `tls_recv` silently discards NewSessionTicket
  messages (resumption is out of scope) and transparently handles KeyUpdate
  (RFC 8446 §4.6.3: re-derives the next application traffic secret via the
  existing `hkdf.sio` machinery — no new key-schedule construction). A
  `close_notify` alert makes `tls_recv` return 0 (EOF), matching `tcp_recv`'s
  convention.
- **HelloRetryRequest**: handled for the cookie/retry case (re-send
  ClientHello with the server's echoed cookie extension, same X25519
  key_share). If the server's HRR instead demands a key-exchange group other
  than X25519, fail closed with a clear error — no other curve is
  implemented.

## Out of Scope

- TLS 1.2 (any part of it — fallback negotiation, RFC 5246's PRF, its
  distinct handshake message set).
- Session resumption, 0-RTT, PSK, `early_exporter_master_secret`,
  `resumption_master_secret`, `binder_key` (all already excluded by
  `hkdf.sio`'s own scope).
- The TLS *server* role — no ServerHello construction, no server-side
  certificate selection, no server-side CertificateVerify signing, no
  `tcp_accept`-driven handshake flow. (Some pieces are structured to be
  reusable by a future server sub-project; none are built or tested here.)
- Client certificates / mutual TLS.
- Any curve other than X25519 for key exchange.
- Any signature scheme other than RSA-PSS and ECDSA P-256 for
  CertificateVerify (e.g. Ed25519/Ed448, ECDSA over P-384/P-521).
- `exporter_master_secret`, TLS exporters.
- ALPN negotiation of anything other than advertising `"http/1.1"` — this
  sub-project does not implement HTTP/2 and does not need to parse an ALPN
  response beyond confirming the server didn't force an unsupported
  protocol.

## Architecture

Six new files, plus one small necessary extension to the existing X.509
parser discovered during implementation planning: `stdlib/x509/cert.sio`
currently has no representation for an EC public key at all (only RSA
`modulus`/`public_exponent`), and neither it nor `stdlib/x509/oid.sio`
recognizes the `id-ecPublicKey`/`prime256v1` OIDs — a certificate presenting
an ECDSA public key has nowhere to be stored today. Since ECDSA P-256
CertificateVerify is in scope, this is closed as part of this sub-project
rather than deferred. Each file has one clear responsibility:

| File | Responsibility |
|---|---|
| `stdlib/crypto/rsa_pss.sio` | RSASSA-PSS signature verification (RFC 8017 §8.1.2/§9.1.2), built on the existing RSA modular-exponentiation machinery already used by `stdlib/crypto/pkcs1.sio`. |
| `stdlib/crypto/ecdsa_p256.sio` | NIST P-256 (secp256r1) elliptic-curve point arithmetic (Weierstrass form — a different curve shape from X25519's Montgomery curve, sharing no code with it) and ECDSA signature verification (FIPS 186-4 / SEC1). |
| `stdlib/tls/record.sio` | TLS 1.3 record-layer framing: `TLSPlaintext`/`TLSCiphertext` wire structures, the `TLSInnerPlaintext` wrapping (real content type + zero padding, wrapped as an outer `application_data`-typed record once encryption starts), per-direction sequence-number-derived nonce construction, and record encryption/decryption via `stdlib/crypto/aead.sio`. Bidirectional by construction (a `RecordLayerState` tracks read and write keys/IVs/sequence numbers independently) so a future server role can reuse it unchanged. |
| `stdlib/tls/transcript.sio` | A running transcript-hash accumulator: feeds raw handshake-message wire bytes in as they're sent/received, and exposes the current hash on demand for `hkdf.sio`'s `derive_secret` calls. Hash algorithm (SHA-256 vs SHA-384) is fixed by the negotiated cipher suite. |
| `stdlib/tls/handshake.sio` | Wire encode/decode for every TLS 1.3 handshake message this sub-project touches: ClientHello, ServerHello, HelloRetryRequest, EncryptedExtensions, Certificate, CertificateVerify, Finished, NewSessionTicket (decode-and-discard only), KeyUpdate. Pure encode/decode — no I/O, no crypto, no state. |
| `stdlib/tls/client.sio` | The orchestrator: `TlsConnection` struct, `tls_connect`/`tls_send`/`tls_recv`/`tls_close`. Drives the handshake state sequence, wires `record.sio` + `transcript.sio` + `handshake.sio` + the existing `x509_verify_chain` + `aead_seal`/`aead_open` + `x25519`/`x25519_base_point_mul` + `hkdf.sio`'s full ladder + `rsa_pss.sio`/`ecdsa_p256.sio` together against a real `TcpSocket`. |
| `stdlib/x509/{oid.sio,cert.sio}` (extended, not new) | Adds `id-ecPublicKey`/`prime256v1` OID recognition and EC public-key-point extraction to the existing X.509 parser, so a certificate's `Certificate` struct can carry an `EcPoint` alongside its existing RSA fields. |

## Data Structures (key interfaces)

```sio
// stdlib/tls/record.sio
pub struct RecordLayerState {
    read_key: RawBuf, read_iv: RawBuf, read_seq: i64,
    write_key: RawBuf, write_iv: RawBuf, write_seq: i64,
    key_len: i64, iv_len: i64, aead_algo: i32,   // AEAD_AES_128_GCM etc, from stdlib/crypto/aead.sio
}
pub fn record_layer_rekey(state: &!RecordLayerState, read_key: &RawBuf, read_iv: &RawBuf, write_key: &RawBuf, write_iv: &RawBuf) with Mut
pub fn record_encrypt(state: &!RecordLayerState, content_type: u8, plaintext: &RawBuf, plaintext_len: i64) -> RawBuf with Mut, IO
pub fn record_decrypt(state: &!RecordLayerState, ciphertext_record: &RawBuf, record_len: i64) -> (u8, RawBuf, i64, bool) with Mut, IO   // (content_type, plaintext, plaintext_len, ok)

// stdlib/tls/transcript.sio
pub struct TranscriptHash { hash_algo: i32, buf: RawBuf, len: i64 }   // buf accumulates raw message bytes; hashed on demand
pub fn transcript_new(hash_algo: i32) -> TranscriptHash with IO
pub fn transcript_append(t: &!TranscriptHash, msg_bytes: &RawBuf, msg_len: i64) with Mut, IO
pub fn transcript_current_hash(t: &TranscriptHash) -> RawBuf with IO

// stdlib/crypto/rsa_pss.sio
pub fn rsa_pss_verify(modulus: &RawBuf, modulus_len: i64, exponent: &RawBuf, exponent_len: i64, message: &RawBuf, message_len: i64, signature: &RawBuf, signature_len: i64, hash_algo: i32) -> bool with IO

// stdlib/crypto/ecdsa_p256.sio
pub fn ecdsa_p256_verify(pubkey_x: &RawBuf, pubkey_y: &RawBuf, message: &RawBuf, message_len: i64, signature: &RawBuf, signature_len: i64) -> bool with IO   // signature is DER-encoded per RFC 8446 §4.2.3, decoded internally

// stdlib/tls/client.sio
pub struct TlsConnection { sock: TcpSocket, records: RecordLayerState, /* ...negotiated cipher suite, peer identity, etc. */ }
pub fn tls_connect(host: &RawBuf, host_len: i64, port: u16) -> (TlsConnection, i64) with IO   // i64: 0 on success, negative sentinel on failure (mirrors tcp_connect)
pub fn tls_send(conn: TlsConnection, buf: &RawBuf, len: i64) -> (TlsConnection, i64) with IO
pub fn tls_recv(conn: TlsConnection, buf: &RawBuf, cap: i64) -> (TlsConnection, i64) with IO   // 0 = EOF (close_notify), negative = error
pub fn tls_close(conn: TlsConnection) with IO
```

Exact field lists and every `handshake.sio` message struct will be finalized
during planning — this section fixes the module boundaries and the
public interface shape, not every internal field.

## Data Flow

1. `tls_connect(host, port)`: `tcp_connect`s, builds and sends a ClientHello
   (fresh random via `crypto_os_random_bytes`, a fresh X25519 ephemeral
   keypair via `x25519_base_point_mul`, SNI = `host`, ALPN = `"http/1.1"`,
   `supported_versions`/`key_share`/`signature_algorithms`/
   `supported_groups` per the Scope Decisions above), and starts a
   `TranscriptHash` from these exact wire bytes.
2. Reads the response record. If it decodes as a HelloRetryRequest: append
   to the transcript, re-send ClientHello with the echoed cookie extension
   (same X25519 share), loop once more. If it decodes as ServerHello:
   append to the transcript, extract the server's X25519 public share and
   negotiated cipher suite, compute the X25519 shared secret, and feed it
   plus the transcript hash into `hkdf.sio`'s ladder
   (`tls13_early_secret` → `tls13_handshake_secret` →
   `derive_secret("c hs traffic"/"s hs traffic")`) to get both handshake
   traffic secrets. `record_layer_rekey`s for the handshake phase.
3. Reads and decrypts EncryptedExtensions, Certificate, CertificateVerify,
   and the server's Finished, in order — each appended to the transcript as
   it arrives, before the transcript hash is read for the *next* message's
   verification (per RFC 8446's precise message-boundary semantics).
   Certificate's chain goes through `x509_verify_chain`. CertificateVerify's
   signature is checked with `rsa_pss_verify` or `ecdsa_p256_verify`
   (whichever the message's stated `signature_algorithm` selects) against
   the transcript hash through Certificate. The server's Finished is
   checked against an independently HMAC-computed expected value.
4. Computes and sends the client's Finished. Derives the Master Secret and
   both application traffic secrets via `hkdf.sio`'s
   `tls13_master_secret`/`derive_secret("c ap traffic"/"s ap traffic")`.
   `record_layer_rekey`s for the application phase.
5. `tls_send`/`tls_recv` wrap/unwrap caller bytes as `application_data`
   records via `record_encrypt`/`record_decrypt`. `tls_recv` discards
   NewSessionTicket records transparently, re-derives on KeyUpdate via
   `hkdf.sio`, and returns 0 on `close_notify`.
6. `tls_close` sends its own `close_notify`, then `tcp_close`s.

## Error Handling

Sentinel bool/i64 convention throughout, matching every prior sub-project on
this branch — never `Result<T,E>`/`Option<T>`. Any handshake failure
(signature mismatch, chain rejection, decrypt/MAC failure, malformed
message, unexpected message in the wrong state, unsupported HelloRetryRequest
group) sends the RFC-appropriate fatal alert where the protocol calls for
one, then fails the connection closed — `tls_connect` returns a negative
sentinel, never a partially-established connection.

## Testing Strategy

- **`rsa_pss.sio`, `ecdsa_p256.sio`**: published NIST CAVP / FIPS 186-4 test
  vectors, independently re-verified against the primary source (fetched
  fresh where needed) before being trusted in committed tests — no
  self-generated expected values. P-256 point arithmetic gets the same
  independent-model-plus-mutation-testing treatment X25519's field
  arithmetic received in the X25519 sub-project.
- **`record.sio`, `transcript.sio`**: verified byte-for-byte against RFC
  8448's published wire-level hex dumps of a real handshake's records and
  messages — no live network peer required for these two modules.
- **`handshake.sio`**: encode/decode round-tripped against RFC 8448's
  published message bytes.
- **`client.sio`**: the only module needing a live peer. Tested against (a)
  a local test TLS 1.3 server for repeatable development-time runs, (b) the
  actual target server for final interop validation, and (c), if feasible,
  one well-known public HTTPS endpoint as an independent real-world check.
  Adversarial cases (forged/expired certificate, bad CertificateVerify
  signature, corrupted/truncated records, unexpected message ordering) are
  tested wherever they can be simulated without a cooperating adversarial
  server.
- Standard per-task review plus the mandatory final whole-plan adversarial
  review (mutation-testing brief, most capable available model), matching
  every prior sub-project — this branch's own track record shows that pass
  has found a real, previously-missed bug in every sub-project so far.

## Global Constraints (carry into the implementation plan)

- RawBuf-based throughout (`net::socket::*`).
- Sentinel bool/i64 error convention; never Result/Option.
- The Madaros compiler defect `docs/handoff/souc_v0800_defects.md` §D8
  (`var x = *ref` on a fixed-array reference aliases instead of copying)
  applies to every new file here — never write that pattern.
- Bare `use <filename>::{name|*}` imports, matching the existing codebase
  convention.
- No AI-attribution line in any commit; Conventional Commits style.
- Every test's expected value from a published, independently-verified
  source — never self-generated.
- `bash scripts/run_sio_test_suite.sh --filter-prefix <prefix>_ --jobs 2`
  for this sub-project's own tests — never the whole-repo suite as a
  checkpoint. `--jobs 2` specifically, per this branch's established
  finding that higher parallelism produces spurious timeouts on this test
  runner right after a rebuild.
