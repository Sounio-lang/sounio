# v13ac Promotion Packet

Date: 2026-05-25
Rung: repair_v13_octo_ossm_hybrid
Candidate: v13ac anchor-balanced from v13y
Status: promoted baseline for v13 compiler/runtime plus structure-slice evidence

## Artifact

Run root:
`/orangefs/training/sounio-ai/repair-v13-octo-ossm-hybrid-20260524T180926-1364094/run-v13ac-anchor-balanced-from-v13y-20260525T065500`

Adapter:
`/orangefs/training/sounio-ai/repair-v13-octo-ossm-hybrid-20260524T180926-1364094/run-v13ac-anchor-balanced-from-v13y-20260525T065500/adapter`

Training manifest:
`train-manifest.json`

Heldout exclusion: 297 records excluded by prompt id/source path.

## Training Shape

Dataset mode: `substance_anchor_balanced`

Sampled records: 800

Key sampled dataset counts:
- `datasets/sounio-ai-octonion-repair/octonion_repair.v1.jsonl`: 482
- `datasets/sounio-ai-science-repair/science_repair.v1.jsonl`: 169
- `datasets/sounio-ai-repair/repair.v2-heldout.jsonl`: 92 clean-policy records after heldout filtering
- `datasets/sounio-ai-structure-hybrid/structure_hybrid.v1.jsonl`: 11

This rung was trained from v13y and was designed to keep the Cayley/octonion anchor while restoring non-minimal scientific completions.

## Slice Gate

Slice decision:
`slice-decision.json`

```json
{
  "algebra_compile_at_5": 1.0,
  "algebra_stdlib_check_pass_at_5": 3,
  "algebra_stdlib_run_pass": 3,
  "heldout_excluded": 297,
  "ossm_compile_at_5": 1.0,
  "run_full_v2": true,
  "scientific_avg_completion_chars": 271.17021276595744,
  "scientific_compile_at_5": 0.9787234042553191
}
```

### Algebra / Octonion

Summary:
`eval/algebra_octonion/results/20260525T101319Z.summary.json`

- prompts: 7
- compile@1: 100%
- compile@3: 100%
- compile@5: 100%
- runtime selector: 100%
- stdlib octonion: 3/3 check, 3/3 run
- output match: 100% over 1 checked output

### Scientific

Summary:
`eval/scientific/results/20260525T101746Z.summary.json`

- prompts: 47
- compile@1: 91.49%
- compile@3: 97.87%
- compile@5: 97.87%
- runtime selector: 100%
- output match: 100% over 6 checked outputs
- scientific avg completion chars: 271.17

### O-SSM

Summary:
`eval/ossm/results/20260525T101955Z.summary.json`

- prompts: 2
- compile@1: 100%
- compile@3: 100%
- compile@5: 100%
- runtime selector: 100%

## Full v2 Gate

Job: 1839 `v13-anchor-recover`
State: COMPLETED 0:0
Node: gpuorangefs-r770-proxmox
Elapsed: 00:27:16

Summary:
`eval/full-v2/results/20260525T102659Z.summary.json`

- prompts: 150
- samples per prompt: 5
- compile@1: 97.33%
- compile@3: 98.67%
- compile@5: 100%
- runtime selector: 100%
- raw syntax purity: 99.33%
- output match: 92.31% over 13 checked outputs
- sample compile rate: 95.33%

## Non-Minimality Audit

Scientific completions:
- count: 235
- avg chars: 271.17
- median chars: 139
- <=180 chars: 173
- <=140 chars: 120
- >=260 chars: 50
- >=500 chars: 33

Full-v2 completions:
- count: 750
- avg chars: 188.95
- median chars: 127
- <=180 chars: 644
- <=140 chars: 476
- >=260 chars: 73
- >=500 chars: 32

Interpretation: v13ac passes the scientific-slice anti-collapse gate and improves over v13y-style minimal witness collapse on the targeted scientific slice. Full-v2 still contains many intentionally short/basic witnesses, so do not claim that every full-v2 prompt became semantically rich.

## Comparison Notes

- v13y was clean and very strong on compiler/runtime, but degenerate/minimalist in completion length.
- v13z and v13aa recovered longer scientific bodies but broke the Cayley/octonion invariant gate.
- v13ab preserved algebra/scientific/O-SSM but failed the anti-collapse threshold with scientific avg completion chars below 220.
- v13ac is the first clean candidate in this ladder to satisfy octonion, O-SSM, scientific, anti-collapse, and full-v2 compiler/runtime gates together.

## Claim Discipline

Supported claim:

v13ac is the promoted clean v13 baseline for compiler-in-the-loop Sounio LoRA evaluation: it preserves octonion/Cayley, O-SSM, and scientific slice gates, passes the scientific non-minimality threshold, and reaches full-v2 compile@5/runtime selector 100% with raw syntax 99.33%.

Do not claim:

- broad mathematical correctness beyond the available stdout/invariant oracles;
- that every full-v2 prompt produces a large/substantive program;
- that Mandelbrot-d2 effects are included in v13.

Mandelbrot-d2 belongs to v14 and should compare against this v13ac baseline.
