# Sounio Formatter

The Sounio formatter (`souc fmt`) provides automatic code formatting with configurable style options. It parses code into an AST, applies formatting rules, and outputs consistently styled code while preserving comments.

## Quick Start

```bash
# Format a file
souc fmt src/main.sio

# Format entire project
souc fmt src/

# Check formatting without changes
souc fmt --check src/

# Show diff of changes
souc fmt --diff src/main.sio
```

## Usage

### Format Files

```bash
# Single file
souc fmt path/to/file.sio

# Multiple files
souc fmt src/lib.sio src/main.sio

# Directory (recursive)
souc fmt src/

# All Sounio files in project
souc fmt .
```

### Check Mode

Verify formatting without modifying files (useful in CI):

```bash
souc fmt --check src/

# Exit code 0 if formatted, 1 if changes needed
```

### Diff Mode

Show what would change:

```bash
souc fmt --diff src/main.sio
```

Output:

```diff
--- original
+++ formatted
 fn main() {
-    let x=1+2
+    let x = 1 + 2
 }
```

### Custom Configuration

```bash
# Use specific config file
souc fmt --config .souniofmt.toml src/

# Override specific options
souc fmt --max-width 80 --indent 2 src/
```

## Configuration

### Configuration Files

The formatter searches for configuration in this order:

1. `sounio.toml` `[format]` section
2. `.souniofmt.toml`
3. `souniofmt.toml`
4. `.dfmt.toml` (legacy)
5. `.souniofmt.json`

Search starts from the file's directory and walks up to the root.

### Configuration Options

Create `.souniofmt.toml` in your project root:

```toml
# Line width
max_width = 100

# Indentation
indent_width = 4
use_tabs = false
tab_width = 4

# Line endings
end_of_line = "lf"  # "lf", "crlf", or "cr"

# Final newline
insert_final_newline = true

# Trailing commas
trailing_comma = "multiline"  # "never", "multiline", or "always"

# Brace style
brace_style = "same_line"  # "same_line", "new_line", or "prefer_same_line"

# Single-line functions
single_line_fn = true

# Imports
group_imports = true
sort_imports = true

# Blank lines
blank_lines_between_items = 1
max_blank_lines = 2

# Comments
format_comments = true
wrap_comments = false
normalize_doc_comments = true

# Spacing
space_after_colon = true
space_before_colon = false
spaces_inside_brackets = false
spaces_inside_parens = false
spaces_around_operators = true

# Method chains
chain_method_break = "auto"  # "never", "auto", or "always"

# Arrays
array_layout = "auto"  # "auto", "single_line", "multi_line", or "one_per_line"
```

### In `sounio.toml`

Include formatting options in your project manifest:

```toml
[package]
name = "my-project"
version = "0.1.0"

[format]
max_width = 100
indent_width = 4
trailing_comma = "always"
```

### JSON Format

Alternative JSON configuration (`.souniofmt.json`):

```json
{
  "max_width": 100,
  "indent_width": 4,
  "use_tabs": false,
  "trailing_comma": "multiline",
  "brace_style": "same_line"
}
```

## Formatting Rules

### Indentation

Default: 4 spaces per level.

```sio
fn example() {
    if condition {
        do_something()
    }
}
```

With `indent_width = 2`:

```sio
fn example() {
  if condition {
    do_something()
  }
}
```

### Line Width

Default: 100 characters. Long lines are wrapped:

```sio
// Before (one long line)
let result = very_long_function_name(argument1, argument2, argument3, argument4, argument5)

// After (wrapped)
let result = very_long_function_name(
    argument1,
    argument2,
    argument3,
    argument4,
    argument5,
)
```

### Trailing Commas

Three modes available:

**`never`** - No trailing commas:

```sio
let arr = [
    1,
    2,
    3
]
```

**`multiline`** (default) - Trailing comma on multiline:

```sio
let arr = [
    1,
    2,
    3,
]
```

**`always`** - Always add trailing comma:

```sio
let arr = [1, 2, 3,]
```

### Brace Style

**`same_line`** (default, K&R style):

```sio
fn example() {
    if condition {
        do_something()
    }
}
```

**`new_line`** (Allman style):

```sio
fn example()
{
    if condition
    {
        do_something()
    }
}
```

**`prefer_same_line`** - Same line unless body is single expression:

```sio
fn short() { single_expression }

fn longer() {
    multiple
    statements
}
```

### Import Formatting

With `sort_imports = true` and `group_imports = true`:

```sio
// Before
use local::module
use std::io
use external::lib
use std::collections::HashMap

// After (sorted and grouped)
use std::collections::HashMap
use std::io

use external::lib

use local::module
```

### Spacing

**Operators** (`spaces_around_operators = true`):

```sio
let x = 1 + 2 * 3
let y = a && b || c
```

**Colons** (`space_after_colon = true`, `space_before_colon = false`):

```sio
struct Point {
    x: f64,
    y: f64,
}
```

**Brackets** (`spaces_inside_brackets = false`):

```sio
let arr = [1, 2, 3]
let val = arr[0]
```

### Method Chains

**`auto`** (default) - Break when exceeding line width:

```sio
// Short chains stay on one line
let result = data.filter(pred).map(f).collect()

// Long chains break
let result = data
    .filter(|x| x.value > threshold)
    .map(|x| transform(x))
    .fold(initial, |acc, x| acc.combine(x))
```

**`always`** - Always break chains:

```sio
let result = data
    .filter(pred)
    .map(f)
    .collect()
```

### Function Definitions

Single-line allowed for short functions (`single_line_fn = true`):

```sio
fn square(x: i64) -> i64 { x * x }

fn longer_function(x: i64, y: i64) -> i64 {
    let sum = x + y
    sum * sum
}
```

Parameters wrap when exceeding line width:

```sio
fn many_params(
    first: i64,
    second: f64,
    third: string,
    fourth: bool,
) -> Result<Output, Error> {
    // body
}
```

### Comments

With `normalize_doc_comments = true`:

```sio
// Before
/** This is a doc comment */

// After
/// This is a doc comment
```

With `wrap_comments = true`, long comments wrap at `max_width`.

### Arrays

**`auto`** (default) - Format based on content:

```sio
// Short arrays stay on one line
let short = [1, 2, 3]

// Long arrays wrap
let long = [
    "first element",
    "second element",
    "third element",
]
```

**`one_per_line`** - Each element on its own line:

```sio
let arr = [
    1,
    2,
    3,
]
```

## Preset Configurations

### Strict Mode

For CI enforcement:

```toml
# .souniofmt.toml
max_width = 100
indent_width = 4
use_tabs = false
trailing_comma = "always"
sort_imports = true
group_imports = true
```

Use programmatically:

```rust
let config = FormatConfig::strict();
```

### Minimal Mode

Preserve original style where possible:

```toml
format_comments = false
normalize_doc_comments = false
sort_imports = false
group_imports = false
```

## Editor Integration

### VS Code

The Sounio VS Code extension integrates formatting:

```json
// settings.json
{
  "editor.formatOnSave": true,
  "[sounio]": {
    "editor.defaultFormatter": "sounio.sounio-vscode"
  }
}
```

### Neovim

With null-ls or conform.nvim:

```lua
require("conform").setup({
  formatters_by_ft = {
    sounio = { "souc_fmt" },
  },
  formatters = {
    souc_fmt = {
      command = "souc",
      args = { "fmt", "--stdin" },
      stdin = true,
    },
  },
})
```

### Emacs

```elisp
(use-package sounio-mode
  :hook (sounio-mode . (lambda ()
    (add-hook 'before-save-hook #'sounio-format-buffer nil t))))
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/sh
souc fmt --check $(git diff --cached --name-only --diff-filter=ACM | grep '\.sio$')
```

Or use pre-commit framework:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: sounio-fmt
        name: Sounio Format
        entry: souc fmt --check
        language: system
        files: '\.sio$'
```

## CI Integration

### GitHub Actions

```yaml
- name: Check formatting
  run: souc fmt --check src/
```

### GitLab CI

```yaml
format:
  script:
    - souc fmt --check src/
  rules:
    - changes:
        - "**/*.sio"
```

## Troubleshooting

### Parse Errors

If the formatter reports parse errors, the file has syntax issues:

```bash
souc check src/file.sio
```

Fix syntax errors before formatting.

### Unexpected Changes

Run with `--diff` to see what will change:

```bash
souc fmt --diff src/file.sio
```

### Ignoring Files

Create `.souniofmtignore`:

```
# Ignore generated files
src/generated/
target/

# Ignore specific file
src/legacy.sio
```

### Preserving Format

Use `// @souniofmt:off` and `// @souniofmt:on`:

```sio
// @souniofmt:off
let matrix = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
// @souniofmt:on
```

## See Also

- [CLI Reference](cli.md) - All compiler commands
- [Language Server](lsp.md) - Editor integration
- [Package Manager](package-manager.md) - Project configuration
