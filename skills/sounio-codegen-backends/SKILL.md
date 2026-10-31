---
name: sounio-codegen-backends
description: "Work on Sounio code generation and backends (LLVM, GPU, native ELF); use when editing `self-hosted/native/`, `self-hosted/gpu/`, or `self-hosted/llvm/`."
---

# Sounio Codegen + Backends

## Overview

Implement and debug lowering/codegen without breaking IR invariants.

## Workflow

### 1) Pick the backend path

- Native ELF emission: `self-hosted/native/`
- GPU: `self-hosted/gpu/`
- LLVM: `self-hosted/llvm/`

### 2) Keep IR invariants explicit

- If you change HLIR/MIR shape, validate passes and analyses that assume SSA form.
- Be conservative around effectful ops when doing backend optimizations.

### 3) Add targeted tests

- Prefer a focused program under `tests/run-pass/` or `tests/compile-fail/`.

## References

- Codegen map: `references/codegen-navigation.md`
