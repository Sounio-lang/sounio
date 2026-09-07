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
and resume. The integrated benchmark coordinator completed the live deterministic baseline:
eight distinct plans, exact pair parity,30 paired blocks per node, eight
NO_GAIN decisions and zero promotion-eligible plans. New operators and the
GRPO corpus remain pending.
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

HTTP Inkling transport requires the internal endpoint and its actual model ID.
The separate offline transport uses an owned Slurm batch and paired token-ID
receipts, without an HTTP endpoint. Generation preserves the original response without
repairing malformed JSON. An interrupted request with no persisted response
is ambiguous and will not be silently issued twice. resume verifies custody
and reports remaining stages; it does not mutate the frozen research context.

The production pilot still requires the founder's fixed three conditions,
three rounds, 32 proposals per condition, 30 interleaved measurement blocks
per node, and the existing promotion criteria. Only the deterministic baseline
has a measured result; no Inkling performance or completed pilot is claimed.


## Owned offline Inkling batch

The current transport is sglang-offline-token-ids. It preserves the pinned
checkpoint/SIF, original mixed NVFP4 and BF16 weights, configured context16384,
one request at a time and at most4096 output tokens. Four hash-bound source
overlays reduce unused Marlin allocation and bound expert/vocabulary copies.
Production kernels are unchanged. Actual stock/candidate GPU controls and
real checkpoint-tensor loader controls are in validation/.

The owned CPU staging slabs are at most4MiB. This matters on the Sparks:
direct CUDA reads from an mmap-backed packed tensor reduced host availability
by about1.1GB per rank, while the staged real-tensor controls preserved exact
TP slices with MB-scale overhead. Checkpoint files and the original SIF remain
unchanged. The rank script fixes its own NCCL2-channel/256KiB profile and skips
the resident tokenizer, HTTP processes and linearized shared-expert copy.

Full initialization, all ten checkpoint files, postprocessing and canonical
hybrid caches completed on both ranks in11918. The first proposal then hit
the33GiB early guard. The native host floor remains32GiB. Releasing unused
allocator pages in11919 was insufficient. These results do not constitute
completed generation or general HTTP-serving acceptance.

For the frozen eight-request canary, the offline cache capacity is 6144 tokens. Its SWA/full ratio is 0.15: the prior 0.1 ratio
produced only 512 SWA tokens and was refused by the runtime admission floor
(511 sliding-window tokens plus one 128-token page) in job11922.
Each encoded prompt is344 tokens plus an unchanged4096-token output allowance.
Oversized requests refuse before loading, and actual pool capacity is checked
again before each request. The model context setting remains16384, but this
batch profile does not accept arbitrary16384-token requests. Cache budget is
part of the hashed runtime lock and a changed profile requires a newly prepared
run; do not rewrite a previous manifest.

Use cycle.py prepare with --transport sglang-offline-token-ids, --budget8 and
the intended native context/admission dependencies. Then use tokenized_cycle.py
pack-encode, the owned launch_pair.py tokenize job, and accept-encode with both
rank receipts. pack-offline creates the frozen inference input bundle.
launch_pair.py offline-generate --input-bundle RUN/offline-bundle.json runs the
actual pinned SGLang worker/ModelRunner/sampler path. It must run inside remote
tmux. Check the Slurm job's final state before collecting both complete receipts
and all16 per-rank response files for accept-offline.

After inference teardown, pack-decode and another owned tokenize job produce
paired decoder receipts. finalize preserves the decoded proposal text exactly.
Continue with native validate, materialize, applicable pair benchmark and report.
No malformed model text is repaired, and failed or ambiguous attempts remain
evidence rather than being silently replayed. Current outcome: status.json.

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
that historical load failed under the host fence. Canonical recovery at epoch15
and production database relocation subsequently passed. Full initialization
now passes, but first-request memory acceptance remains pending as described
above. The historical recovery also exposed a race between
worker recreation and proving the fenced cgroup set empty.


Offline completion receipts and every token response explicitly carry execution_profile:
configured context16384, requested/actual full cache6144, actual SWA capacity,
SWA/full ratio0.15, page128, TP2, concurrency1, output4096, host floor32GiB,
guard33GiB, and existing-pynccl collective backend. Admission refuses an absent
or incompatible profile, including an HTTP or general16K acceptance claim.

The eager offline path enables the already initialized PyNCCL communicator for
forward/decode and rank-zero token broadcast. Its final barrier uses the existing
CPU group. Diagnostic11927 reproduced a lazy torch.distributed NCCL communicator
failing against the worker's64MiB /dev/shm at the embedding all-reduce.
Diagnostic11928 passes exact collective controls and two synthetic embedding/norm
passes on both ranks using PyNCCL. Those controls load no checkpoint tensors and
stop before the first transformer layer; they are not real generation evidence.
