Fixed (Madaros half of ADR-015 row 4): a character literal is its Unicode scalar value, `'é'` is 233, `'€'` is 8364 and `'😀'` is 128512 (UTF-8 decoded, malformed sequences refused). A one-character literal with no closing quote and `''` are refused (E281); `'ab'` is refused too. `'name` with no closing quote is a lifetime and yields no token, so `&'static str` (stdlib/pbpk/regulatory.sio) still lexes as before. An unknown escape in a character literal stays E274.

Previously `let c = 'a` compiled and read the character, and `'é'` stopped the lexer with E271. lean_single still refuses every character literal; that half is a separate change.

Pins: `tests/compile-fail/chr_{unclosed,two_chars,unknown_escape}_madaros.sio`, `tests/run-pass/chr_scalar_value_madaros.sio`, `tests/run-pass/lifetime_static_still_lexes_madaros.sio`.
