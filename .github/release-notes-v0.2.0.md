# Sounio v0.2.0 Release Notes

## Overview

This release completes the core type system with critical bug fixes, 
security hardening, and production-ready package management.

## Highlights

### 🔒 Critical Bug Fixes (4)
- **Borrow Soundness**: Linear types can no longer be consumed while borrowed
- **Bounds Checking**: All refinement_id array accesses now bounds-checked
- **Shift Validation**: Right operand of shift operators now validated
- **Race Condition**: suppress_linear_consume_depth now unconditionally reset

### 🛡️ Security Hardening
- **Depth Limits**: MAX_TYPE_DEPTH (64), MAX_EXPR_DEPTH (256), MAX_LOOP_DEPTH (1024)
- **Complexity Budget**: MAX_TYPE_COMPLEXITY (1000) prevents resource exhaustion
- **Resource Limits**: All table accesses now bounds-checked with errors (E060-E069)

### 📦 Package Manager v0.2.0
Complete rewrite with full functionality:
- `sounio-pkg add name@version` - Registry dependencies
- `sounio-pkg add name --git <url>` - Git dependencies  
- `sounio-pkg add name --path <path>` - Path dependencies
- `sounio-pkg add name --dev` - Dev dependencies
- `sounio-pkg remove name [--clean]` - Remove with vendor cleanup

### ✨ Type System Enhancements
- Epsilon propagation in binary operations
- Enhanced error messages with help hints
- 10 new error codes for security limits

## Statistics

- Total LOC: 29,000+ (self-hosted)
- Test coverage: 139+ tests
- Error codes: 69 total (E001-E069)
- Package manager: 694 lines

## Breaking Changes

None - all changes are additive or bug fixes.

## Migration Guide

No migration needed from v0.1.0.

## Contributors

- Sounio Team

## Downloads

- Linux x86_64: `sounio-v0.2.0-linux-x86_64.tar.gz`
- macOS: `sounio-v0.2.0-darwin.tar.gz`
- Source: `sounio-v0.2.0-src.tar.gz`

## Verification

Stage 1≡Stage 2 bootstrap verification included.
