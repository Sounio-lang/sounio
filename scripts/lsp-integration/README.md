# Sounio LSP Integration

This directory contains integration files for connecting the Sounio validator with Language Server Protocol (LSP) implementations.

## Overview

The Sounio LSP integration provides:
1. **Real-time validation** in code editors
2. **Code diagnostics** for Rust-isms and Sounio conventions
3. **Quick fixes** for common issues
4. **Integration** with existing Sounio LSP infrastructure

## Files

### `rustism_detector.sio`
Located at `self-hosted/lsp/rustism_detector.sio`

This module integrates the `check_sounio.sh` validator with the Sounio LSP system. It:
- Runs the external validator on Sounio files
- Parses the validator output
- Converts findings to LSP diagnostics
- Provides a public API for LSP integration

### Key Functions

#### `lspr_check_file(file_path: string) -> [LspdDiag; 256]`
Main entry point for LSP integration. Returns LSP diagnostics for a file.

#### `lspr_validate_file(file_path: string) -> bool`
Quick validation check for CI/CD integration.

#### `lspr_validate_files(file_paths: [string; 128], count: i64) -> (bool, i64)`
Batch validation for multiple files.

## Integration with Existing LSP

The rustism detector integrates with the existing Sounio LSP through:

### 1. Diagnostics Module
Adds rustism detections to the existing diagnostics pipeline in `diagnostics.sio`.

### 2. Quality Module
Extends the code quality analysis in `quality.sio` with Rust-ism detection.

### 3. Code Actions
Can be extended to provide quick fixes for detected issues.

## Usage in Editors

### VS Code
1. Ensure the Sounio LSP extension is installed
2. The rustism detector will automatically run on `.sio` files
3. Errors and warnings will appear in the Problems panel
4. Hover over issues to see suggested fixes

### Neovim / Vim
With coc.nvim or native LSP client:
```vim
" Example configuration for coc.nvim
{
  "languageserver": {
    "sounio": {
      "command": "sounio-lsp",
      "filetypes": ["sio"],
      "rootPatterns": ["*.sio", "Cargo.toml"],
      "settings": {
        "sounio": {
          "enableRustismDetection": true,
          "validationLevel": "strict"
        }
      }
    }
  }
}
```

### Emacs
With lsp-mode:
```elisp
(use-package lsp-mode
  :ensure t
  :hook ((sio-mode . lsp-deferred))
  :commands lsp)

(use-package lsp-sounio
  :ensure t
  :after lsp-mode
  :config
  (setq lsp-sounio-enable-rustism-detection t))
```

## Configuration

The rustism detector can be configured through LSP settings:

```json
{
  "sounio.lsp.rustismDetection": {
    "enabled": true,
    "severity": {
      "rustMethodCalls": "error",
      "mutRef": "error",
      "vecUsage": "error",
      "unnecessarySemicolons": "warning",
      "missingEffects": "warning",
      "missingReturns": "info"
    },
    "excludePatterns": [
      "**/*.disabled",
      "**/*.backup",
      "**/tests/**"
    ]
  }
}
```

## Extending the Detector

To add new detection patterns:

1. Add a new constant in the `RUSTISM KIND CONSTANTS` section:
```sounio
let LSPR_NEW_PATTERN: i64 = 9
```

2. Update the parsing logic in `lspr_parse_error_line` or `lspr_parse_warning_line`

3. Add detection creation in the appropriate section

4. Update the LSP diagnostics conversion if needed

## Performance Considerations

- The detector runs the external `check_sounio.sh` script
- For large files, this may cause slight delays
- Consider caching validation results
- Batch validation is available for CI/CD scenarios

## Testing

Test the integration with:

```bash
# Test single file
sounio-lsp check examples/hello.sio --rustism

# Test all files
find . -name "*.sio" -exec sounio-lsp check {} --rustism \;
```

## Troubleshooting

### Validator not found
Ensure `check_sounio.sh` is in the PATH or specify full path:
```sounio
let validator_path = "/full/path/to/check_sounio.sh"
```

### Permission denied
Make the validator executable:
```bash
chmod +x check_sounio.sh
```

### No diagnostics shown
Check LSP client logs and ensure:
1. The rustism detector module is compiled
2. LSP settings enable rustism detection
3. File has `.sio` extension

## Future Enhancements

1. **Inline validation**: Validate as you type without saving
2. **Quick fixes**: Automatically fix detected issues
3. **Configuration UI**: Visual configuration in editors
4. **Metrics**: Track validation statistics over time
5. **Custom patterns**: Allow users to define custom validation patterns
