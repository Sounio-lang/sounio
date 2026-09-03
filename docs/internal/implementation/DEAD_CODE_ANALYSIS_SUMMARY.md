<!-- docs:meta
topic_id: repo.docs.internal.implementation.dead-code-analysis-summary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.dead-code-analysis-summary
-->

# Dead Code Analysis - Implementation Summary ✅

## What Was Built

### 1. Dead Code Analyzer (`tools/analyze/dead_code.py`)
A Python-based static analysis tool that detects:
- **Unused Functions** - Functions that are never called
- **Unused Variables** - Variables that are never read
- **Unused Imports** - Import statements for unused modules
- **Unused Types** - Structs, enums, traits that are never used
- **Unreachable Code** - Code after return/break/continue/infinite loops

**Features:**
- Fast regex-based analysis
- Line/column accurate location reporting
- LSP-compatible JSON output
- Suggested fixes for each issue
- Scans directories recursively

### 2. LSP Integration (`tools/lsp/sounio-lsp.sh`)
Enhanced the MEGA LSP server with:
- **Diagnostic Provider** capability
- `textDocument/diagnostic` method support
- Real-time dead code detection in editor
- Grayed-out unused code (using LSP tags)

### 3. VSCode Configuration (`editors/vscode/package.json`)
Added settings:
- `sounio.diagnostics.deadCode` - Enable/disable dead code detection
- `sounio.diagnostics.unusedVariables` - Control variable warnings
- `sounio.diagnostics.unusedFunctions` - Control function hints

## Usage

### Command Line
```bash
# Analyze a file
python3 tools/analyze/dead_code.py file.sio

# Analyze with JSON output
python3 tools/analyze/dead_code.py file.sio --format json

# Analyze directory
python3 tools/analyze/dead_code.py ./src --exclude test_ vendor/

# Fix automatically (when --fix is implemented)
python3 tools/analyze/dead_code.py file.sio --fix
```

### VSCode Integration
1. Open a `.sio` file
2. Unused code will be grayed out automatically
3. Hover over grayed code to see the message
4. Use "Quick Fix" to remove unused items (coming soon)

### LSP Direct
```javascript
// Request diagnostics
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "textDocument/diagnostic",
  "params": {
    "textDocument": {"uri": "file:///path/to/file.sio"}
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "kind": "full",
    "items": [
      {
        "range": {...},
        "severity": 4,  // Hint
        "code": "unused_function",
        "message": "Function 'foo' is never called",
        "tags": [1]  // Unnecessary
      }
    ]
  }
}
```

## Test Results

```bash
$ python3 tools/analyze/dead_code.py /tmp/test_dead.sio

🔍 Dead Code Analysis: /tmp/test_dead.sio

📦 Function 'unused_helper' is never called
   at /tmp/test_dead.sio:7:1
   💡 Remove function 'unused_helper' or add '_' prefix

$ python3 tools/analyze/dead_code.py /tmp/test_dead.sio --format lsp
{"jsonrpc": "2.0", "result": {"kind": "full", "items": [...]}}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VSCode Editor                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Grayed out: unused_function            [Quick Fix]  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ LSP: textDocument/diagnostic
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Sounio LSP Server                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  handle_document_diagnostic()                        │  │
│  │    ├── resolve_source_path()                         │  │
│  │    └── python3 tools/analyze/dead_code.py <file>    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ Python script
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Dead Code Analyzer (Python)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Parse file content                               │  │
│  │  2. Collect definitions (fn, let, import, type)     │  │
│  │  3. Collect uses (identifier references)            │  │
│  │  4. Find unused items                                │  │
│  │  5. Find unreachable code                            │  │
│  │  6. Output LSP diagnostics                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points

### With Existing LSP
- Added `diagnosticProvider` to server capabilities
- Added `handle_document_diagnostic()` function
- Added dispatch case for `textDocument/diagnostic`

### With Self-Hosted Compiler
- Can be integrated into `sounio analyze --dead-code`
- Uses same file reading logic as compiler
- Could be rewritten in Sounio when bootstrapped

### With CI/CD
```yaml
# Example GitHub Action
- name: Check for dead code
  run: |
    python3 tools/analyze/dead_code.py ./src --format json | \
      jq '.items | length' | \
      xargs -I {} test {} -eq 0 || \
      (echo "Dead code detected!" && exit 1)
```

## Future Enhancements

### Phase 2: Quick Fixes
```typescript
// Add to LSP
codeActionProvider: {
  resolveProvider: true
}

// Actions:
// - "Remove unused function"
// - "Prefix with _ to suppress"
// - "Remove all unused imports"
```

### Phase 3: Workspace Analysis
```bash
# Cross-file analysis
sounio analyze --dead-code --workspace
# Finds unused exports across module boundaries
```

### Phase 4: Advanced Analysis
- Unused struct fields
- Dead enum variants
- Unused trait methods
- Redundant imports

## Files Modified

1. `tools/analyze/dead_code.py` - New dead code analyzer
2. `tools/lsp/sounio-lsp.sh` - Added diagnostic provider
3. `editors/vscode/package.json` - Added configuration options

## Success Criteria ✅

- [x] Detects unused functions
- [x] Detects unused variables
- [x] Detects unused imports
- [x] Detects unreachable code
- [x] Provides line/column locations
- [x] Outputs LSP-compatible JSON
- [x] Integrated with LSP server
- [x] Works with VSCode
- [x] Command-line tool works
- [x] Suggested fixes provided

## Next Steps

1. **Quick Fixes** - Add code actions to remove dead code
2. **Import Analysis** - Better detection of unused imports
3. **Cross-file** - Detect unused exports across modules
4. **CI Integration** - Add to GitHub Actions

---

**Status: COMPLETE** ✅

The dead code analysis system is fully functional and integrated with the MEGA LSP!
