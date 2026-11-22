# Standard library conventions

Rules for new stdlib code. They describe the forms the tree already uses. They are not a claim that every existing file complies, and they are not evidence that a function runs.

## Names

- Functions and variables: `snake_case`.
- Structs and other types: `PascalCase`.
- Constants: `SCREAMING_SNAKE_CASE` when a module uses that form.

## Layout

A top-level directory under `stdlib/` is a module area. It does not have to contain all of `mod.sio`, `lib.sio`, `README.md`, and `EXAMPLES.md`.

On 2026-10-27, `bash scripts/stdlib/scan_stdlib.sh` found 1,741 `.sio` files, 102 `lib.sio`, 79 `mod.sio`, and no `*.sio.disabled`. Only `chemistry`, `core`, `database`, and `dataframe` have all four of those files at the top of the directory.

Where a module has a public surface, put callable exports in `lib.sio` or in the file the generator already indexes. `scripts/build/gen_stdlib_api_md.sh` lists `pub fn` and skips a directory that has none.

## Epistemic values

The checked epistemic struct is `Epistemic` in `stdlib/epistemic/knowledge.sio`. Its fields are `val`, `variance`, and `confidence` (`i64`, 0..1000).

Use the free functions. There is no checked `+` or `*` overload on `Epistemic`.

- `ep_measured(val, std_dev)` stores `variance = std_dev * std_dev` and confidence 900.
- `ep_add` and `ep_sub` are the uncorrelated GUM sum and difference. They scale the minimum confidence by 99/100.
- `ep_mul` scales it by 98/100. `ep_div` scales it by 97/100 and requires `Div` and `Panic`.
- `ep_scale` and `ep_shift` keep confidence. `ep_merge` averages the two confidences.

`print` and `println` take one argument. Print a number with `print_f64`. `ep_std` requires the caller to declare `Mut`, `Div`, and `Panic`.

Not every numeric function returns `Epistemic`. Returning `f64` is normal. Return `Epistemic` only when the function is propagating a measurement.

## Tests

Prefer a `//@ run-pass` file under `tests/stdlib/<module>/` for a function you claim is callable. A `pub fn` line is not that claim.

## What not to copy

`compute_mean` is not a function in this tree. Do not add a docstring example that calls it. Do not require a DOI on every `pub fn`; most of the tree does not have one.
