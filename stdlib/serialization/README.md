# stdlib/serialization

Umbrella module for configuration and data interchange formats.

## Submodules

| Format | Path | Status |
|--------|------|--------|
| TOML | `toml::parser` | flat key-value parser |
| YAML | `yaml::parser` | flat key-value parser |
| JSON | `json::parser` | flat key-value parse + serialize |
| MessagePack | `msgpack::lib` | binary pack/unpack |

## Usage

```sounio
var doc = serialization::toml_doc_new()
// or: toml::parser::toml_doc_new()
```

## Tests

- `tests/stdlib/serialization/test_serialization.sio` — umbrella self-tests (check-only)
- `tests/stdlib/toml/test_toml.sio`
- `tests/stdlib/yaml/test_yaml.sio`
- `tests/stdlib/json/test_json.sio`
- `tests/stdlib/msgpack/test_msgpack.sio`