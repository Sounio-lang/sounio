<!-- docs:meta
topic_id: repo.docs.llm-guide.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.readme
-->

# Sounio LLM Guide

How to teach an LLM to write correct Sounio code. Start here.

## The three layers

| Layer | File | When to use |
|-------|------|-------------|
| **Rules** | `../../llms.txt` | System prompt. Include in every session. Covers syntax rules + self-check. |
| **Reference** | `../guide/LLM_PROGRAMMING_GUIDE.md` | Full language reference. Source-cited. Read when the rules aren't enough. |
| **Cookbook** | `cookbook.md` (this dir) | Real patterns for numeric computation, testing, ODE solving, epistemic types. |
| **Stdlib** | `stdlib-index.md` (this dir) | What modules exist, what they export, how to import them. |
| **Errors** | `error-catalog.md` (this dir) | Error messages you'll see + how to fix them. |

## What an LLM needs to know

1. **Syntax differs from Rust** — same look, different rules. `llms.txt` covers this.
2. **Effects are a design discipline, not a hard gate (today)** — every function that divides, mutates, or prints *should* declare it (`with Div`/`Mut`/`IO`), and style/lint expects it. But note: as of Madaros v0.80.0 the default `souc check` does **not** reject a missing IO/Div/Observe effect (E035 is not wired under `check`). Write effects correctly for readability and forward-compat; do not rely on the compiler to catch a missing one.
3. **No dynamic features** — no generics, no closures, no traits, no dynamic dispatch.
4. **stdlib is extensive** — 95 modules. `stdlib-index.md` maps what's usable.

## Quick start for an LLM

Before writing any `.sio` file, read `llms.txt`. Then run the self-check at the bottom
before submitting. When you reach for a stdlib function, check `stdlib-index.md` first.
