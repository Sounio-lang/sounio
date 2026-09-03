<!-- docs:meta
topic_id: repo.docs.audit.souc-wrapper-bare-dash-o-swallow-dispatch-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.souc-wrapper-bare-dash-o-swallow-dispatch-2026-08-18
-->

# Dispatch: `bin/souc <file> -o <out>` swallows `-o` as the output name and exits 0

**Date:** 2026-08-18
**Status:** CLOSED — fail-closed refusal landed in `bin/souc`.
**Owner:** grok-cli1 (opened the dispatch; no other lane held `bin/souc`).
**Not this dispatch:** the E230 handle-table diagnostic, or the E230-patch
3-slot aggregate crash (minimax-cli1 / grok-cli5).
**Surface:** `bin/souc` raw positional `SRC OUT` route.

## Symptom (measured)

On the E230 v3 STAGE (`/orangefs/training/e230-v3-20260818T155411Z`):

```text
souc w2.sio -o w2.elf     # rc=0
# wrote a file literally named `-o` (36924 bytes)
# w2.elf was never created
# compile log: lean_single (`source: w2.sio`, `elf: -o 36924 bytes`)
```

The command reported success and did not produce the named output. Independent
confirm (grok-cli5 case D): lean_single ELF for that source is 36924 bytes —
the same artefact, two lanes.

## Mechanism

`bin/souc` treated any invocation whose first argument is `*.sio` and whose
argc ≥ 2 as the historical lean_single `SRC OUT` ABI and `exec`'d the seed
with the raw argv. `-o` became OUT.

`souc --version` still printed Madaros. A gate that keyed off `--version`
claimed Madaros while the compile ran on the seed.

## Correction (landed)

Before the lean_single `exec`, if the second argument looks like a flag
(`-*`, including `-o`), refuse:

- exit 2
- do not `exec` lean_single
- do not write a file named `-o`
- name the intended form: `souc compile <src> -o <out>`

Legitimate raw form `souc <src.sio> <out>` is unchanged (`scripts/ci/real_language_runner_gate.sh` 6/7).

## Radius

14 scripts under `scripts/ci` use some `.sio -o` form. Zero used the bare
no-verb form. The only in-tree caller was the E230 ceiling gate, rewritten
to the verb. This patch removes the trap for every future caller.

## AI disclosure

Symptom capture, localisation, and the fail-closed wrapper edit by AI agent
(grok-cli1) under human direction, 2026-08-18. GAIDeT-ICMJE 2025.
