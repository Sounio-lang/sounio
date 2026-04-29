# souniodoc — Self-Hosted API Documentation Generator

Generates the Sounio standard library API reference from source code. Written in Sounio itself.

## Usage

```bash
SOUC=./bin/souc

# Generate full stdlib API reference
$SOUC run tools/souniodoc/docgen.sio -- stdlib/ docs/stdlib/STDLIB_API_REFERENCE.md
```

## What It Does

1. Scans `stdlib/**/*.sio` for `pub fn`, `pub struct`, `pub enum`, and `pub type` declarations
2. Associates preceding `///` doc comments with each item
3. Groups items by module directory
4. Writes a markdown reference to the output file

## Input Format

Expects standard Sounio doc comments:

```sio
/// Compute the shortest path from source to all nodes.
/// Uses Dijkstra's algorithm with a priority queue.
pub fn dijkstra(adj: &[[f64; 100]; 100], n: usize, source: usize) -> [f64; 100] with Panic, Div, Mut {
    // ...
}
```

## Output

The generated file replaces `docs/stdlib/STDLIB_API_REFERENCE.md`. It is committed to the repo and can be regenerated at any time.

## Replaces

This tool replaces the previous bash-based generator at `scripts/build/gen_stdlib_api_md.sh`. The bash script is retained for fallback.
