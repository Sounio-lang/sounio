# Sounio Language Server

The Sounio Language Server (`sounio-lsp`) provides IDE features through the Language Server Protocol (LSP). It enables rich editing experiences in any editor that supports LSP.

## Features

The language server provides:

- **Real-time diagnostics** - Syntax and type errors as you type
- **Hover information** - Type signatures, documentation, and epistemic metadata
- **Go to definition** - Jump to symbol definitions
- **Find all references** - Locate all usages of a symbol
- **Code completion** - Context-aware suggestions
- **Semantic highlighting** - Accurate syntax coloring
- **Signature help** - Parameter hints for function calls
- **Document symbols** - File outline/structure
- **Code actions** - Quick fixes and refactorings
- **Inlay hints** - Inline type annotations
- **Rename symbol** - Project-wide renaming
- **Folding ranges** - Code folding regions

## Installation

Build the language server:

```bash
cd compiler
cargo build --release --features lsp

# The binary is at target/release/sounio-lsp
```

Verify installation:

```bash
sounio-lsp --version
sounio-lsp --help
```

## Usage

The language server communicates via stdin/stdout using JSON-RPC:

```bash
sounio-lsp --stdio
```

This is the mode used by all editors. You typically do not run it directly.

## Editor Configuration

### VS Code

Install the Sounio VS Code extension or configure manually:

**Using extension (recommended):**

1. Install the `sounio-vscode` extension from the marketplace
2. The extension automatically finds and starts `sounio-lsp`

**Manual configuration:**

Create `.vscode/settings.json`:

```json
{
  "sounio.serverPath": "/path/to/sounio-lsp",
  "sounio.trace.server": "verbose"
}
```

### Neovim

#### Using nvim-lspconfig

Add to your Neovim configuration (Lua):

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Define the Sounio language server
if not configs.sounio then
  configs.sounio = {
    default_config = {
      cmd = { 'sounio-lsp', '--stdio' },
      filetypes = { 'sounio', 'sio' },
      root_dir = function(fname)
        return lspconfig.util.root_pattern('sounio.toml', '.git')(fname)
          or lspconfig.util.path.dirname(fname)
      end,
      settings = {},
    },
  }
end

-- Enable the server
lspconfig.sounio.setup({
  on_attach = function(client, bufnr)
    -- Enable completion triggered by <c-x><c-o>
    vim.api.nvim_buf_set_option(bufnr, 'omnifunc', 'v:lua.vim.lsp.omnifunc')

    -- Key mappings
    local opts = { noremap = true, silent = true, buffer = bufnr }
    vim.keymap.set('n', 'gd', vim.lsp.buf.definition, opts)
    vim.keymap.set('n', 'K', vim.lsp.buf.hover, opts)
    vim.keymap.set('n', 'gr', vim.lsp.buf.references, opts)
    vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, opts)
    vim.keymap.set('n', '<leader>ca', vim.lsp.buf.code_action, opts)
  end,
  capabilities = require('cmp_nvim_lsp').default_capabilities(),
})
```

#### File type detection

Add to `~/.config/nvim/ftdetect/sounio.lua`:

```lua
vim.api.nvim_create_autocmd({ 'BufRead', 'BufNewFile' }, {
  pattern = { '*.sio', '*.sounio' },
  callback = function()
    vim.bo.filetype = 'sounio'
  end,
})
```

### Emacs

#### Using lsp-mode

Add to your Emacs configuration:

```elisp
(require 'lsp-mode)

;; Define Sounio language server
(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection '("sounio-lsp" "--stdio"))
  :major-modes '(sounio-mode)
  :server-id 'sounio-lsp))

;; Auto-start LSP for Sounio files
(add-hook 'sounio-mode-hook #'lsp)
```

#### Using eglot (Emacs 29+)

```elisp
(require 'eglot)

(add-to-list 'eglot-server-programs
             '(sounio-mode . ("sounio-lsp" "--stdio")))

(add-hook 'sounio-mode-hook 'eglot-ensure)
```

#### Sounio major mode

Create `~/.emacs.d/lisp/sounio-mode.el`:

```elisp
(define-derived-mode sounio-mode prog-mode "Sounio"
  "Major mode for editing Sounio source files."
  (setq-local comment-start "// ")
  (setq-local comment-end ""))

(add-to-list 'auto-mode-alist '("\\.sio\\'" . sounio-mode))

(provide 'sounio-mode)
```

### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "sounio"
scope = "source.sounio"
file-types = ["sio"]
roots = ["sounio.toml"]
language-server = { command = "sounio-lsp", args = ["--stdio"] }
comment-token = "//"
indent = { tab-width = 4, unit = "    " }

[[grammar]]
name = "sounio"
source = { git = "https://github.com/sounio-lang/tree-sitter-sounio", rev = "main" }
```

### Sublime Text

Install the LSP package, then add to LSP settings:

```json
{
  "clients": {
    "sounio": {
      "enabled": true,
      "command": ["sounio-lsp", "--stdio"],
      "selector": "source.sounio"
    }
  }
}
```

### Zed

Add to settings:

```json
{
  "lsp": {
    "sounio": {
      "binary": {
        "path": "sounio-lsp",
        "arguments": ["--stdio"]
      }
    }
  },
  "languages": {
    "Sounio": {
      "language_servers": ["sounio"]
    }
  }
}
```

## Hover Information

The language server provides rich hover information:

### Type Information

Hovering over a variable shows its type:

```
let measurement: Knowledge<f64>
```

### Keyword Documentation

Hovering over keywords shows explanations:

```
fn - Declares a function

Functions in Sounio can have effect annotations:
  fn read_file(path: string) -> string with IO { ... }
```

### Epistemic Types

Hovering over epistemic values shows confidence and source:

```
Knowledge<f64>
  Confidence: 0.95
  Source: laboratory measurement
```

### Tensor Types

Hovering over tensors shows shape information:

```
Tensor<f32, (batch, 784)>
  Element type: f32
  Shape: (batch, 784)
  Operations: matrix multiply, transpose, reshape
```

## Code Completion

Completion triggers automatically and is context-aware:

### Top-Level Items

At module level, suggests:
- `fn`, `struct`, `enum`, `trait`, `impl`
- `effect`, `handler`, `kernel`
- `use`, `mod`, `pub`

### Types

In type position, suggests:
- Primitive types: `i32`, `f64`, `bool`, `string`
- Generic types: `Vec`, `Option`, `Result`, `HashMap`
- Epistemic types: `Knowledge`, `Uncertain`, `Quantity`

### Expressions

In expression position, suggests:
- Keywords: `let`, `if`, `match`, `for`, `while`
- Effect operations: `perform`, `handle`, `sample`, `observe`

### Effects

After `with`, suggests available effects:
- `IO`, `Mut`, `Alloc`, `Panic`
- `Async`, `GPU`, `Prob`, `Div`

### Units

In numeric contexts, suggests scientific units:
- Mass: `kg`, `g`, `mg`
- Volume: `L`, `mL`
- Concentration: `mol/L`, `mg/mL`
- Time: `s`, `ms`, `h`

## Diagnostics

Real-time error reporting with rich context:

```
error[E0308]: type mismatch
  --> src/main.sio:15:12
   |
15 |     let x: i32 = "hello"
   |            ^^^   ^^^^^^^ expected `i32`, found `string`
```

Diagnostics are updated incrementally as you type.

## Code Actions

Quick fixes available via lightbulb or keyboard shortcut:

- **Add missing import** - Insert `use` statement
- **Fix typo** - Correct misspelled identifiers
- **Add type annotation** - Insert explicit type
- **Implement trait** - Generate trait method stubs
- **Extract function** - Refactor selection to function

## Inlay Hints

Optional inline annotations showing inferred types:

```sio
let x/* : i32 */ = compute()
let data/* : Vec<Knowledge<f64>> */ = load_measurements()
```

Enable/disable in editor settings.

## Troubleshooting

### Server Not Starting

Check that the binary exists and is executable:

```bash
which sounio-lsp
sounio-lsp --version
```

### No Completions

Verify the file has correct extension (`.sio`) and the language is detected:

```bash
# Check LSP logs (varies by editor)
# VS Code: Output panel -> Sounio Language Server
# Neovim: :LspLog
```

### Slow Performance

For large projects, ensure incremental analysis is enabled:

```toml
# sounio.toml
[build]
incremental = true
```

### Debug Logging

Enable verbose logging:

```bash
RUST_LOG=debug sounio-lsp --stdio
```

Or configure trace in editor:

```json
{
  "sounio.trace.server": "verbose"
}
```

## See Also

- [CLI Reference](cli.md) - Command-line compiler
- [REPL](repl.md) - Interactive development
- [Formatter](formatter.md) - Code formatting
