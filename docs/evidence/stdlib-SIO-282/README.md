<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-282.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-282.readme
-->

# SIO-282: Python-free boundary metadata on stdlib execution routes

[Notion SIO-282](https://app.notion.com/p/4053f37f9371814f9ae6c8adbec77dee)
records ownership, acceptance, source, jobs and publication. This isolated branch
starts at remote main `a886ce9bdb9b6b9ca4f748028fca0248bfa9f125`.

## Trigger

Complete Git materialization in interrupted Slurm17597 restored the existing
Python boundary attestor, absent from the earlier selected packet. The wrapper
invoked it indirectly. ADR009 prohibits auxiliary Python, so the owner stopped
the job, retaining472 partial rows without synthesizing a complete627-row JSON
or declaring a valid baseline. Original17578 and earlier receipts remain intact.

This port implements operational policy, dependency closure, claim bindings
and artifact identity in Node. It does not calculate or certify a scientific
result. No Python ran as an oracle or helper for this port's qualification.

## Implementation and assurance

- `core.mjs`: fatal UTF8 with BOM preservation, physical non-strict paths,
  safe references, SHA256, Unicode-codepoint-sorted ASCII canonical JSON,
  receipt identity and fsynced atomic writes.
- `closure.mjs`: host import auditing and raw Madaros AST report validation.
  Host syntax remains non-authoritative. Missing, malformed, saturated or
  contradictory raw closure cannot grant strict OK. Exactly one status, capacity,
  saturated and parse_failed control is required; booleans must be literal.
  Duplicate, missing or invalid controls refuse UNKNOWN (deliberate hardening).
- `policy.mjs` and `claims.mjs`: quoted/multiline TSV, policy discovery and
  classification, TOML declarations, typed evidence and exact source/package/
  compiler/content bindings. Schema-invalid container metadata is UNKNOWN,
  preventing admission through ambiguous language-specific repr coercion.
- `attestor.mjs`: evaluate/finalize/verify, existing schema, identity-only
  limitations and refusal codes. Finalization additionally binds its requested
  source to the preflight root artifact and requires a recorded compiler.
  Graph/evidence arrays are checked explicitly: deliberate refusal hardenings.
- `manifest_value.mjs`: project name/entry lookup with original precedence
  and malformed-table refusal through the same parser.

`bin/madaros` uses Node on check/compile/run boundary and project-manifest routes.
Compiler source and selected numerical engine are unchanged. Explicit raw pins
retain priority; invalid explicit pins still refuse. The resolver checks them
before probing the host fallback. No runtime/storage speedup was measured from
this small lazy-probe cleanup.

Verified prebuilt materialization remains intact in both launchers. A full tree
may generate `bin/madaros-linux-x86_64` plus its verified stamp despite another
explicit raw pin. Those are generated artifacts, not tracked-source changes.
In17597 the125051232-byte ELF came locally from tracked gzip/SHA; no download
was established. Custody must bind those exact generated identities separately
from unexpected additions. This port does not claim to save125MB or weaken
strict committed-compiler provenance.

Package-release/registry Python commands, the original Python library and its
broad legacy driver remain separate, unqualified routes. They are not invoked
by these Node controls or planned bounded check/compile/run/project validation.
This is not a claim that every repository workflow is Python-free.

## Parser custody and compatibility

The repository-locked smol-toml1.6.1 archive passed its SHA512 integrity check
before extraction. BSD-3-Clause notices remain under `vendor/smol-toml`.
`vendor/PROVENANCE.json` records upstream/patched hashes; `vendor/FILES.sha256`
covers every included parser/license/provenance file. A narrow date patch rejects
impossible calendar rollovers and retains temporal lexemes; first-party code
rejects invalid UTF8 and unpaired surrogate strings. No runtime install/network.

This pinned parser accepts TOML1.1 grammar. Universal syntax parity with the old
TOML1.0 parser, OS-specific diagnostic text and arbitrary container repr are not
asserted. Valid schema values and receipt canonical identities have focused
controls. Invalid container fields become UNKNOWN. Metadata temporal coercion
retains offsets and six-digit microseconds as text; this is not scientific
numerical computation. Receipt numeric fields are safe integer counters.

## Executed operational validation

Core34, closure67, policy/claims30 and full CLI28 controls passed:159 controls.
They cover Unicode/BOM/UTF8, quoted TSV/TOML, physical escapes, raw-AST
contradictions, policy/evidence refusals, source/compiler/ELF mutation, rehashed
forged fields, wrong finalization source, deterministic receipts and project
precedence/malformed tables. Fixture compiler/artifact bytes never execute.

Independent review retained failing-before/passing-after metadata repros for
missing finalization source binding (also a legacy defect), malformed evidence,
container-context collisions, scalar lib refusal and formatting edges. No full
legacy Python-suite parity or historical178-assertion denominator is claimed.

## Native execution checkpoint

Slurm **17600 completed0:0 in13s**, using exact source checkpoint
`b81b5bcf447e441e9cb45806d7451c6b104882ec`, raw a292 and Node22.22.2.
All159 controls and six strict source/project check/compile/run routes passed.
The direct candidate was UNKNOWN21; the wrapper refused1 before dispatching a
build and emitted neither an ELF nor a requested final receipt. Two controlled
Python-deny probes returned125 in a separate225-byte log; the runtime guard log
was empty. This covers the guarded names and observed wrapper routes, not a
whole-system execve/language audit.

Body/final-integrity/aggregate statuses were0. The2109 tracked source files,
six fixtures, Node/raw binaries, seven Node runtime dependencies and45 wrapper
tools/libraries plus two helpers passed before/after checks. Only the exact two
expected materialized prebuilt artifacts were admitted separately. Terminal
workspace/catalog comparison matched cleanb81. Requested1CPU/4GiB; allocated
two logical CPUs/4GiB by scheduler granularity on cpuops-t560-proxmox.

The two retained compile ELFs have SHA256
`eed608a000135801a8dbaf3a8607c2934af4e34eac91b2765023d5fe29594531`.
Run receipts verified against that identical retained content; temporary run
paths were removed by the unchanged wrapper and are not claimed retained.

`native17600-custody.tar.gz`:947001 bytes, SHA256
`86abba40ffdb4ffd93393f4855c34ebd94e793ad5c5a9d07566aef5685dbbe77`.
The153-entry text packet excludes source bulk and ELF payloads. Fresh extraction
passed FILES verification and offline reproduction. The coordinator independently
rehashed all2109 files against cleanGitb81 (SHA256/blob/mode), recomputed seven
canonical receipt identities, checked hello source binding and actual raw routes,
and verified the negative never dispatched build. Its separate receipt is
`ROOT_NATIVE_VERIFICATION.json`. The first independent verifier incorrectly
expected a final receipt on strict refusal; that verifier assumption was corrected,
with no production change and no rerun.

This qualifies bounded operational native wire only. It does not replace the
incomplete17597 measurement or qualify a full scientific/E2E corpus. New full
measurements require their own composed source and complete fresh census.

Draft for human review; no main merge, auto-merge or queue.
