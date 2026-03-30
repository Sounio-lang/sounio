<!-- docs:meta
topic_id: repo.docs.architecture.module-closure-truth
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.module-closure-truth
-->

# Module Closure Truth

## Summary

This note defines the current internal rule that resolved module closure is an
authority surface, not only a debugging witness.

The goal is to keep closure construction and execution aligned so the compiler
does not rediscover module truth differently at later stages.

## Bundle-As-Authority Rule

The hard rule is:

- resolve imports once during closure construction
- store the resolved result in the bundle
- make execution consume only bundle-resolved entries
- do not reintroduce execution-time path guessing or path recomposition

This is the current meaning of `bundle-as-authority` in the repo.

## Authoritative Per-Module Fields

Each module entry should carry, at minimum:

| Field | Meaning |
| --- | --- |
| `requested_spec` | import text or requested module spec |
| `resolved_path` | final resolved module path used by the closure |
| `resolution_kind` | how the module was resolved |
| `parent_module_id` | bundle parent that requested this module |
| `depth` | closure depth from the root |
| `src_start` | start offset in the assembled source space |
| `src_len` | copied byte length present in the closure |
| `requested_len` | requested byte length before truncation |
| `tk_start` | token-stream start for the module |
| `tk_end` | one-past-last token for the module |
| `exec_state` | current module execution state |
| `parse_status` | current parse outcome or failure class |

Additional fields such as hashes, local function counts, scan counts, or slice
metadata may exist, but the fields above are the minimum truth-bearing closure
contract for the current work.

## Execution States

The current state split is:

- `closure-only`
- `lexed`
- `parsed`

These states exist to prevent a module from being counted as "present" when it
was only resolved, not actually lexed or parsed.

## Why This Split Exists

The rebuild work established a recurring failure mode:

- a module can appear in the closure witness
- the compiler can still fail to execute that module truthfully
- path guessing during execution can hide whether the failure is about
  resolution, lexing, parsing, or capacity

`bundle-as-authority` exists to remove that ambiguity.

## Current M9 Interpretation

M9 gives the current large-surface interpretation of closure truth:

### Byte-Cap Failure

At `2 MiB`:

- `m4_large_surface_probe` fails by explicit closure truncation with
  `module_count=46` and `first_truncated_index=28`
- `ontology_witness_program_probe` fails by explicit closure truncation with
  `module_count=48` and `first_truncated_index=30`

This is a Closure Truth plus Capacity Truth result. The compiler knows which
world it tried to assemble and where byte truncation happened.

### Non-Byte Large-Surface Failure

At `4 MiB`, `8 MiB`, and `16 MiB`:

- byte truncation is gone
- import resolution is no longer the validated blocker on the probe set
- large-surface execution still fails with node and pool saturation

The observed stable pressure point is:

- `ND_COUNT=262143`
- `ovf_nd=1`
- `ovf_pool=1`

The first attributed large-surface parse failures currently land in:

- `self-hosted/compiler/module_loader.sio`

That means the next blocker class is large-surface execution capacity/model
truth, not import lookup truth.

## Practical Reading

The current closure model is strong enough to support these claims:

- the compiler can describe the intended module world
- the compiler can distinguish byte truncation from later large-surface failure
- the compiler can attribute at least some large-surface failures to a specific
  module path

The current closure model is not yet strong enough to claim:

- stable large-surface execution truth
- stable ontology-sized direct-driver semantic truth

## Related Docs

- [compiler-maturity-blueprint.md](./compiler-maturity-blueprint.md)
- [truth-layers.md](./truth-layers.md)
- [truth-frontier.md](./truth-frontier.md)
