# Add Sounio language support

## Summary

[Sounio](https://github.com/sounio-lang/sounio) is an L0 systems + scientific programming language for epistemic computing. It has its own file extension (`.sio`), syntax, and type system distinct from any existing recognized language.

## Language details

- **File extension**: `.sio`
- **Type**: programming
- **Color**: `#4B8BBE`
- **Notable features**: effect system, `Knowledge<T>` epistemic types, linear types, units of measure, refinement types

## Evidence of use

- GitHub repository: https://github.com/sounio-lang/sounio
- 700+ `.sio` source files in the repo (stdlib, compiler, tests, examples)
- VS Code extension published: https://open-vsx.org/extension/sounio-lang/sounio-vscode
- Active development since 2024, self-hosted compiler (Sounio compiled in Sounio)

## Files

### `lib/linguist/languages.yml` entry

```yaml
Sounio:
  type: programming
  color: "#4B8BBE"
  extensions:
    - ".sio"
  tm_scope: source.sounio
  ace_mode: text
  codemirror_mode: null
  codemirror_mime_type: null
  language_id: 0  # assigned by maintainers
  aliases:
    - sio
  wrap: false
```

### Sample files

See `samples/Sounio/` directory in this PR:
- `hello.sio` — minimal program demonstrating basic syntax
- `closure_fn_ref.sio` — function references (higher-order programming)
- `epistemic_bmi.sio` — epistemic types with uncertainty propagation

## TextMate grammar

Available at: https://github.com/sounio-lang/sounio/blob/main/.github/linguist/sounio.tmLanguage.json
