# stdlib/toml

Minimal flat TOML parser (`toml::parser`).

- Up to 32 key-value pairs, keys ≤63 bytes, strings ≤127 bytes
- Types: string, integer, float, bool
- Skips `#` comments and blank lines; no tables or arrays

## Entry points

- `toml_doc_new`, `toml_parse`, `toml_get_*`, `toml_self_test`

## Tests

`tests/stdlib/toml/test_toml.sio`