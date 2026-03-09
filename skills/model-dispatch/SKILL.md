# Sounio Model Dispatch Guide

## Decision Flowchart

- Touches 3+ files across compiler stages? -> Opus 4.6
- Requires formal reasoning / soundness? -> Opus 4.6
- Long autonomous run or prose writing? -> GPT-5.3-Codex
- Whole-codebase sweep or impact analysis? -> Gemini 3.1 Pro
- System-level debugging or cross-check? -> GLM-5
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

## Multi-Model Review Triggers

Enable manually on: check/epistemic.sio, check/causal.sio, hlir/ir.sio type enums, bootstrap seed updates.

## Fallback Chain

Opus 4.6 -> GPT-5.3-Codex -> Gemini 3.1 Pro -> Sonnet 4.6. GLM-5 for cross-checks.

## Targets

- Opus first-compile success rate: >= 90%
- Sonnet first-compile success rate: >= 80%
- Sonnet-to-Opus escalation rate: < 15%
- Paper draft revision cycles: <= 2
