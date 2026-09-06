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
Its commands currently cover prepare, generate, validate, report and resume.
Materialization/benchmark, new operators and the GRPO corpus remain pending.
Do not mistake this implemented admission stage for completed M3–M6.

Compile admission.sio through bin/souc with the rebuilt engine on the R770.
Run test_admission.py and test_cycle.py against that actual executable.
The committed gate transcript records adversarial refusal and an eight-plan
deterministic custody regression. No test fixture is a real LLM response.

prepare requires an already prepared context, its provenance evidence and the
SHA256 of the admission executable. Context extraction from ontology queries
is not yet automated. The semantic contract is
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
