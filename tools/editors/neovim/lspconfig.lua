-- Neovim setup for the Sounio Language Server.
--
-- Requires nvim 0.8+ and the `neovim/nvim-lspconfig` plugin
-- (https://github.com/neovim/nvim-lspconfig).
--
-- Install: drop this file (or its contents) into your config, e.g.
--   ~/.config/nvim/lua/sounio.lua
-- and require it from init.lua:
--   require('sounio')
--
-- `souc` must be on PATH (or pass an absolute path via `cmd`).
-- `souc lsp --stdio` starts the checked preview LSP route in bin/souc; the
-- pure-Sounio server rebuild is tracked separately.

local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')
local util = require('lspconfig.util')

-- Filetype detection for .sio files.
vim.filetype.add({
  extension = {
    sio = 'sounio',
  },
})

-- Register the server only once.
if not configs.sounio_lsp then
  configs.sounio_lsp = {
    default_config = {
      cmd = { 'souc', 'lsp', '--stdio' },
      filetypes = { 'sounio' },
      root_dir = function(fname)
        return util.root_pattern('souc.toml', '.git')(fname)
          or util.path.dirname(fname)
      end,
      single_file_support = true,
      settings = {},
    },
    docs = {
      description = 'Sounio Language Server preview.',
      default_config = {
        root_dir = [[root_pattern("souc.toml", ".git")]],
      },
    },
  }
end

-- Bring up the server with default capabilities.
lspconfig.sounio_lsp.setup({
  on_attach = function(client, bufnr)
    -- Standard LSP keymaps. Override or remove to taste.
    local opts = { buffer = bufnr, silent = true }
    vim.keymap.set('n', 'K',     vim.lsp.buf.hover,           opts)
    vim.keymap.set('n', 'gd',    vim.lsp.buf.definition,      opts)
    vim.keymap.set('n', 'gr',    vim.lsp.buf.references,      opts)
    vim.keymap.set('n', '<F2>',  vim.lsp.buf.rename,          opts)
    vim.keymap.set('i', '<C-Space>', vim.lsp.completion.trigger or function() end, opts)
  end,
})
