---
name: sounio-erdos
description: Work on Erdős problems (Erdős-Straus, Erdős90, chi>=6) in the Sounio repository
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# Sounio Erdős Problems

Use this skill when the user asks about or wants to work on Erdős-related problems in Sounio:
- Erdős-Straus GPU sieve (`examples/erdos_straus_gpu_*.sio`)
- Erdős90 unit-distance / 5-squares problem (`stdlib/research/erdos90_*.sio`, `formal/lean4/SounioErdos90*.lean`)
- chi>=6 / chromatic / cube-cover campaigns (`examples/erdos/chi6_*.py`)

## Canonical branch

The consolidated lane lives in:

- Branch: `research/erdos-canonical`
- Worktree: `/workspace/sounio-erdos-canonical`

Do **not** work on Erdős problems from `docs/i18n-zh-hk` or other unrelated branches.

## Resolver

Use the canonical compiler resolver:

```bash
source /workspace/sounio/scripts/lib/resolve_souc.sh
$SOUC --version
```

## Quick checks

```bash
# Erdős90 search kernel
cd /workspace/sounio-erdos-canonical
$SOUC check stdlib/research/erdos90_search.sio

# Erdős-Straus examples (may need syntax updates for current Madaros)
$SOUC check examples/erdos_straus_gpu_sieve.sio
$SOUC check examples/erdos_straus_gpu_lean.sio
```

## Running Erdős90 witness gates

```bash
bash scripts/gates/erdos90_grid144_witness_gate.sh
bash scripts/gates/erdos90_subset144_witness_gate.sh
bash scripts/gates/erdos90_unified_witness_gate.sh
```

## Heavy validation

For cluster sweeps, Slurm arrays, or GPU sieve stress tests, use the Sounio Compiler Foundry / Slurm path. Never run heavy stress in `/workspace/sounio-erdos-canonical`.

See `docs/ops/foundry_slurm_handoff.md` for submission templates.

## Offload requirements

Any new Lean theorem statement or numeric claim about Erdős bounds requires `bin/llm-offload -t math-review -p xai` before commit. Log the review in `.claude/llm_offload_log.md`.

## File map

| Problem | Source | Formal | Scripts |
|---|---|---|---|
| Erdős-Straus | `examples/erdos_straus_gpu_*.sio` | `tests/formal/gpu_thread_rewrite_proof.lean` | `scripts/run_erdos_lean_pipeline.sh` |
| Erdős90 grid | `stdlib/research/erdos90_search.sio` | `formal/lean4/SounioErdos90Grid*Witness.lean` | `scripts/gates/erdos90_grid*_witness_gate.sh` |
| Erdős90 subset | `stdlib/research/erdos90_subset.sio` | `formal/lean4/SounioErdos90Subset*Witness.lean` | `scripts/gates/erdos90_subset*_witness_gate.sh` |
| Erdős90 unified | `stdlib/research/erdos90_optimize.sio` | `formal/lean4/SounioErdos90UnifiedQsqrt3Witness.lean` | `scripts/gates/erdos90_unified_witness_gate.sh` |
| chi>=6 | `examples/erdos/chi6_*.py` | `formal/lean4/SounioChi6*.lean` | `examples/erdos/test_chi6_*.sh` |
