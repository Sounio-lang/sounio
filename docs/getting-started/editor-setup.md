---
title: Editor Setup
description: Configure your editor for Sounio development with LSP and syntax highlighting
prerequisites: installation.md
reading_time: 8 minutes
---

# Editor Setup

This guide helps you configure your editor for Sounio development. Sounio provides a Language Server Protocol (LSP) implementation for rich editor features like autocompletion, error highlighting, and go-to-definition.

## Visual Studio Code

VS Code provides the best Sounio development experience through its LSP integration.

### Installing the Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
3. Search for "Sounio" (if a published extension exists)
4. Click Install

If there is no published extension yet, you can configure the LSP manually.

### Manual LSP Configuration

First, build the LSP server:

```bash
cd compiler
cargo build --release --features lsp
```

The server binary will be at `compiler/target/release/sounio-lsp`.

Create or edit `.vscode/settings.json` in your project:

```json
{
    "sounio.serverPath": "/path/to/sounio/compiler/target/release/sounio-lsp",
    "files.associations": {
        "*.sio": "sounio"
    }
}
```

### Syntax Highlighting (Manual)

If syntax highlighting is not automatic, create a TextMate grammar:

1. Create a directory: `~/.vscode/extensions/sounio-syntax/`
2. Create `package.json`:

```json
{
    "name": "sounio-syntax",
    "displayName": "Sounio Language",
    "version": "0.1.0",
    "engines": {
        "vscode": "^1.70.0"
    },
    "contributes": {
        "languages": [{
            "id": "sounio",
            "aliases": ["Sounio", "sounio"],
            "extensions": [".sio"],
            "configuration": "./language-configuration.json"
        }],
        "grammars": [{
            "language": "sounio",
            "scopeName": "source.sounio",
            "path": "./syntaxes/sounio.tmLanguage.json"
        }]
    }
}
```

3. Create `language-configuration.json`:

```json
{
    "comments": {
        "lineComment": "//",
        "blockComment": ["/*", "*/"]
    },
    "brackets": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"]
    ],
    "autoClosingPairs": [
        { "open": "{", "close": "}" },
        { "open": "[", "close": "]" },
        { "open": "(", "close": ")" },
        { "open": "\"", "close": "\"" },
        { "open": "'", "close": "'" }
    ],
    "surroundingPairs": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"],
        ["\"", "\""],
        ["'", "'"]
    ]
}
```

4. Create `syntaxes/sounio.tmLanguage.json`:

```json
{
    "scopeName": "source.sounio",
    "name": "Sounio",
    "patterns": [
        { "include": "#comments" },
        { "include": "#keywords" },
        { "include": "#strings" },
        { "include": "#numbers" },
        { "include": "#types" }
    ],
    "repository": {
        "comments": {
            "patterns": [
                {
                    "name": "comment.line.double-slash.sounio",
                    "match": "//.*$"
                },
                {
                    "name": "comment.block.sounio",
                    "begin": "/\\*",
                    "end": "\\*/"
                }
            ]
        },
        "keywords": {
            "patterns": [
                {
                    "name": "keyword.control.sounio",
                    "match": "\\b(if|else|match|for|while|loop|break|continue|return|in)\\b"
                },
                {
                    "name": "keyword.other.sounio",
                    "match": "\\b(fn|let|var|const|struct|enum|impl|trait|type|module|import|use|pub|with|effect|handler|handle|perform|kernel|async|await)\\b"
                },
                {
                    "name": "storage.modifier.sounio",
                    "match": "\\b(linear|affine|move|copy)\\b"
                }
            ]
        },
        "strings": {
            "patterns": [
                {
                    "name": "string.quoted.double.sounio",
                    "begin": "\"",
                    "end": "\"",
                    "patterns": [
                        {
                            "name": "constant.character.escape.sounio",
                            "match": "\\\\."
                        }
                    ]
                }
            ]
        },
        "numbers": {
            "patterns": [
                {
                    "name": "constant.numeric.sounio",
                    "match": "\\b[0-9][0-9_]*\\.?[0-9_]*([eE][+-]?[0-9_]+)?\\b"
                }
            ]
        },
        "types": {
            "patterns": [
                {
                    "name": "storage.type.sounio",
                    "match": "\\b(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|f32|f64|bool|char|string|Self)\\b"
                },
                {
                    "name": "support.type.sounio",
                    "match": "\\b(Knowledge|EpistemicValue|Option|Result|Vec|HashMap|HashSet)\\b"
                }
            ]
        }
    }
}
```

5. Restart VS Code

## Neovim

Neovim provides excellent LSP support through `nvim-lspconfig`.

### Prerequisites

- Neovim 0.8 or later
- `nvim-lspconfig` plugin

### LSP Configuration

Add to your `init.lua`:

```lua
-- Sounio LSP configuration
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Define the Sounio LSP
if not configs.sounio_lsp then
    configs.sounio_lsp = {
        default_config = {
            cmd = { '/path/to/sounio/compiler/target/release/sounio-lsp' },
            filetypes = { 'sounio' },
            root_dir = function(fname)
                return lspconfig.util.find_git_ancestor(fname)
                    or lspconfig.util.path.dirname(fname)
            end,
            settings = {},
        },
    }
end

-- Enable the server
lspconfig.sounio_lsp.setup({
    on_attach = function(client, bufnr)
        -- Enable completion triggered by <c-x><c-o>
        vim.api.nvim_buf_set_option(bufnr, 'omnifunc', 'v:lua.vim.lsp.omnifunc')

        -- Mappings
        local opts = { noremap=true, silent=true, buffer=bufnr }
        vim.keymap.set('n', 'gd', vim.lsp.buf.definition, opts)
        vim.keymap.set('n', 'K', vim.lsp.buf.hover, opts)
        vim.keymap.set('n', 'gr', vim.lsp.buf.references, opts)
        vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, opts)
    end,
})
```

### Filetype Detection

Add to your `init.lua`:

```lua
vim.filetype.add({
    extension = {
        sio = 'sounio',
    },
})
```

### Syntax Highlighting with Tree-sitter

If a Tree-sitter grammar is available for Sounio:

```lua
require('nvim-treesitter.configs').setup({
    ensure_installed = { 'sounio' },
    highlight = { enable = true },
})
```

Otherwise, create a basic syntax file at `~/.config/nvim/syntax/sounio.vim`:

```vim
" Sounio syntax highlighting

if exists("b:current_syntax")
    finish
endif

" Keywords
syn keyword sounioKeyword fn let var const struct enum impl trait type
syn keyword sounioKeyword module import use pub with effect handler handle perform
syn keyword sounioKeyword kernel async await linear affine move copy
syn keyword sounioControl if else match for while loop break continue return in

" Types
syn keyword sounioType i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64
syn keyword sounioType bool char string Self
syn keyword sounioType Knowledge EpistemicValue Option Result Vec HashMap HashSet

" Constants
syn keyword sounioConstant true false

" Comments
syn match sounioComment "//.*$"
syn region sounioComment start="/\*" end="\*/"

" Strings
syn region sounioString start='"' end='"' contains=sounioEscape
syn match sounioEscape "\\." contained

" Numbers
syn match sounioNumber "\<[0-9][0-9_]*\>"
syn match sounioFloat "\<[0-9][0-9_]*\.[0-9_]*\>"

" Highlighting
hi def link sounioKeyword Keyword
hi def link sounioControl Conditional
hi def link sounioType Type
hi def link sounioConstant Constant
hi def link sounioComment Comment
hi def link sounioString String
hi def link sounioEscape Special
hi def link sounioNumber Number
hi def link sounioFloat Float

let b:current_syntax = "sounio"
```

## Emacs

### LSP Mode

If you use `lsp-mode`:

```elisp
(use-package lsp-mode
  :commands lsp
  :hook (sounio-mode . lsp))

;; Register Sounio with lsp-mode
(with-eval-after-load 'lsp-mode
  (add-to-list 'lsp-language-id-configuration '(sounio-mode . "sounio"))
  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection "/path/to/sounio/compiler/target/release/sounio-lsp")
    :major-modes '(sounio-mode)
    :server-id 'sounio-lsp)))
```

### Eglot (Built-in for Emacs 29+)

```elisp
(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs
               '(sounio-mode . ("/path/to/sounio/compiler/target/release/sounio-lsp"))))
```

### Major Mode

Create `~/.emacs.d/lisp/sounio-mode.el`:

```elisp
;;; sounio-mode.el --- Major mode for Sounio -*- lexical-binding: t; -*-

(defvar sounio-mode-syntax-table
  (let ((table (make-syntax-table)))
    ;; Comments
    (modify-syntax-entry ?/ ". 124" table)
    (modify-syntax-entry ?* ". 23b" table)
    (modify-syntax-entry ?\n ">" table)
    ;; Strings
    (modify-syntax-entry ?\" "\"" table)
    table))

(defvar sounio-keywords
  '("fn" "let" "var" "const" "struct" "enum" "impl" "trait" "type"
    "module" "import" "use" "pub" "with" "effect" "handler" "handle" "perform"
    "kernel" "async" "await" "linear" "affine" "move" "copy"
    "if" "else" "match" "for" "while" "loop" "break" "continue" "return" "in"))

(defvar sounio-types
  '("i8" "i16" "i32" "i64" "i128" "u8" "u16" "u32" "u64" "u128"
    "f32" "f64" "bool" "char" "string" "Self"
    "Knowledge" "EpistemicValue" "Option" "Result" "Vec" "HashMap" "HashSet"))

(defvar sounio-font-lock-keywords
  `((,(regexp-opt sounio-keywords 'words) . font-lock-keyword-face)
    (,(regexp-opt sounio-types 'words) . font-lock-type-face)
    ("\\<\\(true\\|false\\)\\>" . font-lock-constant-face)))

(define-derived-mode sounio-mode prog-mode "Sounio"
  "Major mode for editing Sounio source code."
  :syntax-table sounio-mode-syntax-table
  (setq-local comment-start "// ")
  (setq-local comment-end "")
  (setq-local font-lock-defaults '(sounio-font-lock-keywords)))

(add-to-list 'auto-mode-alist '("\\.sio\\'" . sounio-mode))

(provide 'sounio-mode)
;;; sounio-mode.el ends here
```

Add to your init file:

```elisp
(add-to-list 'load-path "~/.emacs.d/lisp/")
(require 'sounio-mode)
```

## LSP Features

The Sounio LSP server provides:

### Diagnostics

Real-time error and warning messages as you type.

### Hover Information

Hover over a symbol to see its type and documentation.

### Go to Definition

Jump to where a function, type, or variable is defined.

### Find References

Find all places where a symbol is used.

### Autocompletion

Context-aware suggestions for functions, types, and variables.

### Document Symbols

Navigate to functions and types in the current file.

## Running the LSP Server

Build with LSP support:

```bash
cd compiler
cargo build --release --features lsp
```

The server communicates over stdio. Start it directly to test:

```bash
./target/release/sounio-lsp
```

It will wait for JSON-RPC messages on stdin. Your editor's LSP client handles this communication automatically.

### Debugging the LSP

Set the `SOUNIO_LSP_LOG` environment variable for logging:

```bash
SOUNIO_LSP_LOG=debug ./target/release/sounio-lsp 2>lsp.log
```

## Troubleshooting

### LSP Not Starting

1. Verify the binary exists and is executable:
   ```bash
   ls -la /path/to/sounio-lsp
   ```

2. Check that it runs:
   ```bash
   /path/to/sounio-lsp --version
   ```

3. Ensure your editor configuration points to the correct path

### No Syntax Highlighting

1. Confirm the file has a `.sio` extension
2. Check that the filetype is recognized:
   - VS Code: Check status bar shows "Sounio"
   - Neovim: `:set filetype?` should show `sounio`
   - Emacs: `M-x describe-mode` should show `Sounio`

### LSP Errors in Editor

Check the LSP log output:
- VS Code: Output panel, select "Sounio Language Server"
- Neovim: `:LspLog`
- Emacs: `*lsp-log*` buffer

## Next Steps

With your editor configured, you are ready to develop Sounio projects effectively. Continue with:

- [Project Structure](./project-structure.md) - Organize larger projects
- [Language Reference](../LLM_PROGRAMMING_GUIDE.md) - Complete syntax guide

## See Also

- [Installation](./installation.md) - Building with different features
- [Your First Uncertainty](./your-first-uncertainty.md) - Sounio's core feature
