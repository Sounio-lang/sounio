# Unresolved primitive-source witnesses

These are unchanged positive witnesses retained as reviewable evidence. They are outside the stdlib acceptance corpus because they describe unresolved compiler/primitive-string capabilities, and are not counted as passing or ignored tests.

`utf8_source_string.sio` expects a primitive source string containing embedded NUL to preserve bytes after NUL. The current primitive representation is a NUL-terminated C string; the witness still fails its original assertion under the adopted narrow compiler (Slurm18238). Owned Utf8Text from checked ByteBuffer/explicit bytes preserves U+0000 and is separately native-qualified. The from_string conversion explicitly documents the primitive C-string boundary. No general string ABI rewrite was approved or claimed.

`source_unicode_char_witness.sio` expects actual non-ASCII character literals é and 😀 to decode as Unicode scalar values. Fresh source compiler checking rejects the literal continuation byte with E271 (Slurm18218). This is distinct from actual non-ASCII string literal byte encoding, which passed native execution.

Neither file has had assertions weakened or annotation removed. Exact original files and earlier diagnostic overlays are also retained in durable core-evidence-20261027-02.tar.gz, SHA256 fe3aaf2f9417a903b5da5213c9b0f8ee423002e0f0b6bf94092ac5221c82b9d1. Run the witnesses explicitly against the compiler being assessed and preserve the resulting failure until the underlying capability is implemented.

