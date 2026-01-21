# Drift scan patterns

Use this when examples/docs start looking “Rust-like” and drift away from Sounio.

## Quick `rg` searches

- Rust macros in `.sio`/docs: `rg -n \"\\b(println!|assert!)\\b\" docs examples tests stdlib`
- `&mut` (Sounio uses `&!`): `rg -n \"\\b&mut\\b\" docs examples tests stdlib`
- `let mut` (Sounio uses `var`): `rg -n \"\\blet\\s+mut\\b\" docs examples tests stdlib`
- Attribute syntax (not supported): `rg -n \"#\\[\" docs examples tests stdlib`
- `.d` references (doc drift): `rg -n \"\\.d\\b\" *.md docs`

## Scripted scan

Run `python3 skills/sounio-language/scripts/scan_syntax_drift.py` for a single consolidated report.
