<!-- docs:meta
topic_id: repo.docs.internal.implementation.quick-fixes-summary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.quick-fixes-summary
-->

# Quick Fixes for Dead Code - Implementation Summary ✅

## What Was Built

### 1. Enhanced Code Actions (`tools/lsp/sounio-lsp.sh`)

Updated the LSP server to provide intelligent quick fixes for dead code diagnostics:

**Available Quick Fixes:**

| Action | Kind | Description |
|--------|------|-------------|
| 🗑️ "Remove unused function 'X'" | `quickfix` | Deletes the entire function |
| 🔧 "Suppress with '_' prefix" | `quickfix` | Adds underscore to suppress warning |
| 🗑️ "Remove unused variable 'X'" | `quickfix` | Deletes the variable declaration |
| 🗑️ "Remove unused import 'X'" | `quickfix` | Deletes the import statement |
| 🧹 "Remove all N unused items" | `source.fixAll` | Batch removal of all dead code |
| 📦 "Organize imports" | `source.organizeImports` | Sorts and cleans imports |
| 🎨 "Format document" | `source.formatDocument` | Formats the entire file |

### 2. Enhanced Dead Code Analyzer (`tools/analyze/dead_code.py`)

Improved the analyzer to capture full function ranges:
- Now tracks complete function bodies (start line → end line)
- Properly calculates brace matching for nested functions
- Provides accurate ranges for LSP text edits

### 3. LSP Integration

**Diagnostic Provider:**
- Reports dead code as hints (severity 4)
- Tags items as "unnecessary" (tag 1)
- Shows grayed out in VSCode

**Code Action Provider:**
- Responds to `textDocument/codeAction` requests
- Associates fixes with specific diagnostics
- Generates proper text edits for removals

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  VSCode User Sees:                                          │
│  ╔═══════════════════════════════════════════════════════╗  │
│  ║  fn unused_helper() -> () with IO {     [Quick Fix]   ║  │
│  ║  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (grayed out)         ║  │
│  ║      print("unused")                                   ║  │
│  ║  }                                                    ║  │
│  ╚═══════════════════════════════════════════════════════╝  │
└────────────────────────┬────────────────────────────────────┘
                         │ Hover → Quick Fix menu
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Quick Fix Menu:                                            │
│  ┌──────────────────────────────────────┐                  │
│  │ 🗑️ Remove unused function            │                  │
│  │ 🔧 Suppress with '_' prefix          │                  │
│  │ ───────────────────────────────────  │                  │
│  │ 🧹 Remove all 3 unused items         │                  │
│  └──────────────────────────────────────┘                  │
└────────────────────────┬────────────────────────────────────┘
                         │ User clicks
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LSP Applies WorkspaceEdit:                                 │
│  {                                                          │
│    "changes": {                                             │
│      "file:///path/file.sio": [                             │
│        {                                                    │
│          "range": {"start": {"line": 6, "character": 0},   │
│                   "end": {"line": 10, "character": 0}},    │
│          "newText": ""                                      │
│        }                                                    │
│      ]                                                      │
│    }                                                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

## Demo

### Command Line Test
```bash
# Test code actions
$ python3 /tmp/test_code_action.py
Code Actions Received: 4 actions
  - [quickfix] Remove unused function 'unused_helper'
  - [quickfix] Suppress with '_' prefix
  - [source.organizeImports] Organize imports
  - [source.formatDocument] Format document
```

### VSCode Usage
1. Open a `.sio` file with unused code
2. Unused functions/variables appear grayed out
3. Hover over the grayed code
4. Click "Quick Fix" (or press `Ctrl+.`)  
5. Select an action:
   - **Remove**: Deletes the code
   - **Suppress**: Renames to `_unused_name`
   - **Remove all**: Cleans up entire file

## Text Edit Examples

### Remove Function
```json
{
  "title": "Remove unused function 'unused_helper'",
  "kind": "quickfix",
  "edit": {
    "changes": {
      "file:///test.sio": [
        {
          "range": {
            "start": {"line": 6, "character": 0},
            "end": {"line": 10, "character": 0}
          },
          "newText": ""
        }
      ]
    }
  }
}
```

### Suppress with Prefix
```json
{
  "title": "Suppress with '_' prefix",
  "kind": "quickfix",
  "edit": {
    "changes": {
      "file:///test.sio": [
        {
          "range": {
            "start": {"line": 6, "character": 3},
            "end": {"line": 6, "character": 3}
          },
          "newText": "_"
        }
      ]
    }
  }
}
```

### Batch Remove All
```json
{
  "title": "Remove all 5 unused items",
  "kind": "source.fixAll",
  "edit": {
    "changes": {
      "file:///test.sio": [
        {"range": {...}, "newText": ""},  // Item 5
        {"range": {...}, "newText": ""},  // Item 4
        {"range": {...}, "newText": ""},  // Item 3
        {"range": {...}, "newText": ""},  // Item 2
        {"range": {...}, "newText": ""}   // Item 1
      ]
    }
  }
}
```

## File Changes

1. **`tools/analyze/dead_code.py`** - Enhanced to track full function ranges
2. **`tools/lsp/sounio-lsp.sh`** - Updated code actions with dead code fixes

## VSCode Settings

Users can configure which quick fixes appear:

```json
{
  "sounio.diagnostics.deadCode": true,
  "sounio.diagnostics.unusedVariables": true,
  "sounio.diagnostics.unusedFunctions": true
}
```

## Testing

### Manual Test
```bash
# Start with test file
cat > /tmp/test.sio << 'EOF'
fn main() -> i32 with IO {
    let unused = 42
    print("hello")
    return 0
}

fn dead_func() -> () with IO {
    print("never called")
}
EOF

# Run analyzer
python3 tools/analyze/dead_code.py /tmp/test.sio

# Test LSP code actions
python3 /tmp/test_code_action.py
```

### Automated Tests
```bash
# All LSP smoke tests pass
bash tools/lsp/test_smoke.sh
# [lsp-smoke] PASS
```

## Future Enhancements

### Phase 3: More Fixes
- **Extract function**: Move code to new function
- **Inline variable**: Replace var with its value
- **Rename symbol**: Safe renaming across files
- **Add missing import**: Auto-import from stdlib

### Phase 4: Refactoring
- **Convert to trait impl**: Extract interface
- **Make function pure**: Remove effects
- **Add error handling**: Wrap with Result
- **Parallelize loop**: Convert to kernel

## Integration Points

### With LSP
- Uses `textDocument/codeAction` method
- Associates fixes with `textDocument/publishDiagnostics`
- Generates `WorkspaceEdit` for changes

### With VSCode
- Shows lightbulb icon on hover
- Provides "Quick Fix" menu
- Applies edits automatically

### With CI/CD
```yaml
- name: Check for dead code
  run: |
    if python3 tools/analyze/dead_code.py ./src --format json | jq -e '.summary.total > 0'; then
      echo "Dead code detected! Run 'sounio fix' to clean up."
      exit 1
    fi
```

## Success Criteria ✅

- [x] Quick fixes appear for dead code
- [x] Remove single item works
- [x] Suppress with prefix works
- [x] Remove all items works
- [x] Organize imports available
- [x] Format document available
- [x] Proper text edit ranges
- [x] LSP integration complete
- [x] All smoke tests pass

---

**Status: COMPLETE** ✅

Users can now click "Quick Fix" in VSCode to remove dead code automatically!
