# stdlib/yaml

Minimal flat YAML parser (`yaml::parser`).

- Up to 32 `key: value` lines
- Types: string (quoted/bare), number, bool (`true`/`false`/`yes`/`no`), null (`null`/`~`)

## Entry points

- `yaml_doc_new`, `yaml_parse`, `yaml_get_*`, `yaml_self_test`

## Tests

`tests/stdlib/yaml/test_yaml.sio`