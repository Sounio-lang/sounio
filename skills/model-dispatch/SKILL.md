---
name: model-dispatch
description: "Choose which AI model to use for a given Sounio task: type-checker changes, render examples, PGO passes, docs/skills updates, bootstrap debugging, codebase sweeps, and paper sections."
---

# Sounio Model Dispatch Guide

## Workflow

1. Identify the task type from the Decision Flowchart or Quick Reference table below.
2. Assign the task to the indicated model.
3. If the result is uncertain or the first attempt fails, escalate via the Fallback Chain.
4. For multi-file compiler changes that touch formal semantics, enable Multi-Model Review.

## Decision Flowchart

- Touches 3+ files across compiler stages? -> Opus 4.6
- Requires formal reasoning / soundness? -> Opus 4.6
- Long autonomous run or prose writing? -> GPT-5.3-Codex
- Whole-codebase sweep or impact analysis? -> Gemini 3.1 Pro
- System-level debugging or cross-check? -> GLM-5
- Render example completion (self-contained, no imports)? -> Sonnet 4.6
- Touches self-hosted frontend (check/resolve) + correctness proof? -> Opus 4.6
- PGO pass (IR module, new opcodes, self-test)? -> Opus 4.6
- Website/docs/skills update? -> Sonnet 4.6
- Native backend single-file pass (peephole/frame/encode)? -> Sonnet 4.6
- Multi-file native pipeline change (codegen+lower_ir+encode)? -> Opus 4.6
- Everything else -> Sonnet 4.6

## Quick Reference

| Question | Model |
|----------|-------|
| Change the type checker | Opus 4.6 |
| Write a test | Sonnet 4.6 |
| Write a paper section | GPT-5.3-Codex |
| What will break if I change this? | Gemini 3.1 Pro |
| Why is the bootstrap failing? | GLM-5 |
| Refactor one stdlib file | Sonnet 4.6 |
| Add keyword to lexer+parser+checker+HLIR | Opus 4.6 |
| Second opinion on checker logic | GLM-5 |
| Name a new language feature | GPT-5.3-Codex |
| Find every use of X in 30 files | Gemini 3.1 Pro |
| Complete a render example (heatmap, DAG, 3D) | Sonnet 4.6 |
| Add rasterize_* primitive to stdlib/render/ | Sonnet 4.6 |
| Fix self-hosted --check failure | Opus 4.6 |
| Debug --native-compile bootstrap crash | Opus 4.6 + GLM-5 cross-check |
| Add a new PGO pass (new ir/*.sio module) | Opus 4.6 |
| Run sprint gate and triage failures | Sonnet 4.6 |
| Generate website SVG previews | Sonnet 4.6 |
| Add a new peephole pattern to peephole.sio | Sonnet 4.6 |
| Wire a new native pass in compile_ir_function | Opus 4.6 |
| Debug native binary crash / ELF reloc error | Opus 4.6 + GLM-5 cross-check |
| Add emit_* helper to encode.sio | Sonnet 4.6 |

## Multi-Model Review Triggers

Enable manually on: check/epistemic.sio, check/causal.sio, hlir/ir.sio type enums, bootstrap seed updates.

## Fallback Chain

Opus 4.6 -> GPT-5.3-Codex -> Gemini 3.1 Pro -> Sonnet 4.6. GLM-5 for cross-checks.

## Targets

- Opus first-compile success rate: >= 90%
- Sonnet first-compile success rate: >= 80%
- Sonnet-to-Opus escalation rate: < 15%
- Paper draft revision cycles: <= 2
