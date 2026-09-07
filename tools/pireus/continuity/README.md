# Pireus continuity execution

Canonical plan: docs/roadmap/PIREUS_CONTINUITY_PLAN.md. Current acceptance state:
status.json. This directory preserves source lineage, review output and
executable validation receipts separately from planned milestones.

The current-source Madaros repair for Seq<T> struct-field ownership is in
commit 193670aa6b on the parent integration branch. Three unchanged ontology
queries and six scoped Pireus gates pass; formal V13/V14 remain OPEN.

## External proposal admission

admission.sio is the Sounio semantic boundary; cycle.py transports data and
persists immutable dependencies, raw requests/responses, proposals and receipts.
Its commands cover prepare, generate, validate, materialize, benchmark, report
and resume. The integrated benchmark coordinator awaits live verification;
new operators and the GRPO corpus remain pending.
Do not mistake this implemented admission stage for completed M3–M6.

Compile admission.sio through bin/souc with the rebuilt engine on the R770.
Run test_admission.py and test_cycle.py against that actual executable.
The committed gate transcript records adversarial refusal and an eight-plan
deterministic custody regression. No test fixture is a real LLM response.

prepare accepts --context-engine RESEARCH_CONTEXT.elf or a supplied context,
provenance evidence and the admission executable SHA256. The Sounio producer
queries TripleStore/SPARQL with declared research-local primitive facts.
These are not observations of running hardware. The semantic contract is
docs/internal/concepts/pireus-external-proposal-admission.md.

```sh
python3 tools/pireus/continuity/cycle.py prepare --run RUN \
  --context CONTEXT.json --evidence QUERY_RECEIPT.txt \
  --engine-sha256 ADMISSION_SHA256 --condition deterministic --budget 8
python3 tools/pireus/continuity/cycle.py generate --run RUN
python3 tools/pireus/continuity/cycle.py validate --run RUN --engine ADMISSION.elf
python3 tools/pireus/continuity/cycle.py resume --run RUN
```

Inkling conditions additionally require the internal batch endpoint and its
actual served model ID. Generation preserves the original response without
repairing malformed JSON. An interrupted request with no persisted response
is ambiguous and will not be silently issued twice. resume verifies custody
and reports remaining stages; it does not mutate the frozen research context.

The production pilot still requires the founder's fixed three conditions,
three rounds, 32 proposals per condition, 30 interleaved measurement blocks
per node, and the existing promotion criteria. No performance result is claimed.


## Material evidence and measurement contract

Sounio emits admitted PTX through materialize_ptx.sio and fixtures through
numeric_fixtures.sio. Python only loads PTX and records GPU output bits.
material_parity.sio decides exact non-NaN bits / NaN class agreement.
The 320 vectors cover all 256 basis pairs, 32 dense inputs and 32 edge inputs.
Job 11859 passed all eight plans on both Sparks (5120 exact bits each).
Job 11860 ran an intentionally poisoned sign mask outside admission: both
nodes refused with 42 mismatches. These finite tests are not a general FP proof.

benchmark_decision.sio consumes four sets of 30 paired blocks: direct and
shuffle controls on each node. It computes median gain in ppm and a seeded
4000-resample percentile bootstrap. All four medians must reach 50000 ppm
and all four lower 95% bounds must be positive. This is an exploratory
per-comparison interval, not familywise coverage. The measured scope is
resident-layout kernels; layout conversion is excluded. Each CUDA event
brackets 32 captured kernel launches over 16384 vectors. Partial trials are
retained and refused on retry; they are not silently overwritten.

Materialize with cycle.py materialize --run RUN --engine ADMISSION.elf.
Run cycle.py benchmark inside workspace tmux, after generation teardown,
with --run, --engine, --fixture-engine, --parity-engine and --gain-engine.

The observer/launcher checks both host grants in addition to the Kubernetes
lease. It refuses FENCED or unknown observations and leaves recovery to the
frozen Spark Pair Arbiter. Checkpoint hashes passed on both nodes in job 11864;
serving remains unverified because the host fence interrupted weight loading.
The initial trigger is unresolved. Recovery also exposed a race between
worker recreation and proving the fenced cgroup set empty.
