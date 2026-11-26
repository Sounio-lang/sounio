# N-10 — Granite-3.1-8B cross-backend parity receipt (2026-11-26)

**Status at 13:08 UTC: CPU controls PASS; staged checkpoint verified; GPU
prediction NOT TESTED.** This is an interim accounting receipt, not a claim
of cross-GPU or Metal identity. The queued dependency chain is fail-closed.
No 61,440-token artifact is available yet; no drift or mismatch count can be
reported as zero. The registered prediction remains unchanged.

## Source and contract

- Base: `575d1eb57` (`origin/feat/llm-backend-v2` at worktree creation).
- Step-zero specification: [`../granite_step0.md`](../granite_step0.md), verified
  against Transformers v4.48.3 Granite modeling source and the IBM checkpoint
  config. The checkpoint index has four shards and deliberately omits
  `lm_head.weight`; output uses the unscaled embedding weight with tied head.
- BF16 architecture: 40 layers, hidden 4096, intermediate 12800, 32 Q/8 KV
  heads of 128, vocab 49155. No Q/K normalization. Input embeddings multiply
  by `0x41400000`; attention dots multiply by `0x3C000000` before softmax;
  both projected residual branches multiply by `0x3E6147AE` before addition;
  every logit scales by `0x3D800000` (division by 16) before top-two reduction.
  CUDA, Metal and CPU source paths share the existing kernels/contract ops.
  Unsupported Granite quantized lanes are refused rather than silently running
  Qwen3 normalization slots.
- The RoPE generator already imports `pow`, `sin`, `cos` from `math::pure`,
  using in-tree arithmetic, not host libm. No replacement was warranted by
  step 3. Extracted source of `qwen3_fill_rope_table` has identical SHA256 at
  base and current head:
  `b534afbe6f3ad7e3fc2cf984ce05c4bf409e4f37ffba0f1baa0b8e2a5b24127f`.
  This is **source identity**, not a measured macOS/Linux table-byte equality.
- Granite pretokenization uses ByteLevel GPT-2-style splitting and existing
  vocabulary/merge parsing. The frozen audit chat date is November 26, 2026
  to avoid time-dependent system messages. The existing in-tree Unicode
  category classifier is bounded; the 480-prompt independent oracle is the
  required acceptance gate, not the four smoke cases.
- Audit source commit pinned for both scheduled GPU jobs:
  `ad4a505c1931d48552397089ca11d7cfaeabd988`. Registered prediction remains in
  `granite_parity.sbatch`: token, runner-up, top1, top2, margin bit mismatches
  and drift all zero, `CROSS_BACKEND_IDENTICAL`.

## Stage — complete

CPU-only job **29242**, `dl380-dl380-proxmox`, `COMPLETED 0:0`, elapsed
`00:15:05`, batch MaxRSS `4207124K`. Working path is the Ceph-mounted shared
training directory:

```
/orangefs/training/llm-granite-n10/granite-3.1-8b-instruct
```

3.6.1 against `huggingface.co/ibm-granite/granite-3.1-8b-instruct`, ran
`git lfs fsck`, wrote `MODEL_REVISION` and `SHA256SUMS`, and ran
`sha256sum -c SHA256SUMS` successfully, under `set -euo pipefail`.
It downloaded the weights once. No model weights were committed to GitHub.
The full manifest is at
`/orangefs/training/llm-granite-n10/granite-3.1-8b-instruct/SHA256SUMS`.
Manifest readback job 29268 is queued; the actual revision and per-shard
hashes will be included by the terminal receipt collector. Direct visibility
from r740/r770 has not yet been measured (GPU slots unavailable); the r740
ops job checks the shared manifest before running.

The first CPU-only staging attempt (29219) failed because its `/orangefs`
was an unmounted container overlay. On `dl380-dl380`, 29236 confirmed an
actual writable Ceph mount. Git LFS was missing from both CPU worker images;
the existing t560 Linux x86-64 executable was transported through five small
CPU-only Slurm jobs, SHA-checked and reconstructed in the task-owned shared
`tooling` directory, without modifying node packages or services. Binary SHA256:
`95795bacd7040d12e1d510309ea8abfe8d3007ff60b7d996af0df19298096fb8`;
compressed SHA256:
`2d3445a9420c15b9ebe44595b83af257fcbee741e3454ff7bab194478d973db7`.

## Executed CPU evidence

Source-built Madaros SHA256:
`5df8c456cdb5db06bf48485b4ebfdbae518e39be4d50efa798b5d00ebdbd3659`.

Job **29243**, source `faa2f6fd5`, `dl380-dl380-proxmox`, completed
`00:02:27`, MaxRSS `3517528K`, exit `0:0`. Individual check/build/run steps:

| Control | Result | SHA256 of executable |
|---|---|---|
| Raw memory unaligned read/write and neighbor preservation | PASS | `5d38c6d206f66a9809a89fb902773590ccea9e8cb683797e4f15edbce5e4cc14` |
| POSIX arena/mapped-shard contract | PASS | `ecdef3e3011efc69ccbcc72b9c91180aa3e3f6d617843ddbb7c0b9a59bf5f237` |
| Shared CPU argmax/top-two, width 49155, three tie/tail cases | PASS | `9d1acacb1ad663238147202c8544459a39c7fc8d840ae772c64ab5c42026a4e0` |
| Full adversarial FTZ CPU suite, including Granite tail cases | PASS | `0e138f42098b38ed85c87bd9f51241b108ed5a514777f5bf7f2ea8af9503c8ae` |
| Audit 61440 | check/build PASS, **not executed** | `28d603fdba2aacc4135a45aa9e26e1c951cde3e5241792730832c3d82a4ac8ab` |

The adversarial suite printed `GRANITE_ARGMAX_TAIL_CPU_PASS`,
`TOTAL_OPS=11 TOTAL_ELEMENTS=13312 TOTAL_SUBNORMAL_OUTPUTS=0
TOTAL_FMA_MIN_NORM_HITS=0`, and `LLM_V2_ADVERSARIAL_FTZ_PASS backend=cpu`.
The three Granite-specific cases are asserted separately from the original
11 serialized operation records. Tail cases cover indices 49152–49154,
cross-lane ties as well as a unique final-element winner, and exact value bits.
This does not establish CUDA execution.

Job **29284**, source `66e64b558` (runtime argument built-ins),
`dl380-dl380-proxmox`, completed `00:01:49`, MaxRSS `1915092K`, exit `0:0`.
`GRANITE_TOKENIZER_SMOKE_PASS` against actual staged tokenizer JSON:

- Tokenizer SHA256:
  `ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654`.
- Smoke ELF SHA256:
  `f712234ba964a8019e53268c4a17a2e48c7ae4fbf038fbb363261732c741e4b5`.
- Expected IDs from the pinned vocabulary: `hello` → 7656; ` hello!` →
  17964, 19; `123` → 35, 36, 37; role delimiters/end-of-text/tool-call →
  49152, 49153, 0, 49154. These four controls passed, not the 480-prompt gate.
- The current audit's `arg_count()` / `get_arg()` revision is scheduled for
  full recheck/rebuild and CLI refusal controls in job 29285. A previous
  native `main(argc,argv)` smoke build returned 1 at the argc guard (29266);
  the runtime-argument built-in smoke then passed (29284).

## Localized failures and repairs before subsequent runs

The source-built compiler at the base lacks the out-of-tree runtime names
`ptr_read_u8`, `ptr_write_u8`, `ptr_read_i64_le`, `ptr_write_i64_le` and
related mapping primitives. Job 29224 reproduced E137 on a standalone
16-byte allocation/write/read witness and on the **unmodified** base
`575d1eb57` audit. Source-level byte access and POSIX dynamic mapping replace
those assumptions on the Linux lane. The typed-byte probe 29226 ran cleanly.

A second discriminating failure was an unaligned `*mut u16` store: after
writing `0xabcd` at byte offset 1 of `0x112233442a667788`, expected
`1234605615719632264` (`0x112233442aabcd88`), actual `11259272`
(`0x0000000000abcd88`) in job 29231. A typed u32 write also overwrote neighbors.
The source helper now splits u16/u32 stores into byte stores, and its regression
contract passed on both CPU nodes (29233 and 29232) and in the complete
29243 suite. The stack-address mmap symbol probe died (139, 29227); using
heap-backed NUL-terminated symbol storage and the existing POSIX indirect
call-through ABI passed (29228), followed by the actual mmap contract.
Compiler-requested `Mut`/`Alloc` effects, a raw-write argument cast, and
linear tokenizer construction were repaired before the next source check.
No numerical GPU divergence was observed or diagnosed because no GPU run
has occurred. No speculative parity fix has been applied to a nonexistent
comparison.

## Slurm accounting (batch MaxRSS)

A diagnostic job may be `COMPLETED` while its recorded child check/build/run
returns nonzero. The observations column records that distinction. Units are
exactly those reported by `sacct`; `—` denotes no allocation or no final
measurement. Node abbreviations: **P**=`dl380p-dl380p-proxmox`,
**D**=`dl380-dl380-proxmox`, **R740**=`gpuorangefs-multi-r740-proxmox`.

| Job | Node | State / exit | Elapsed | MaxRSS | Observation |
|---|---|---|---|---|---|
| 29219 | P | FAILED 1:0 | 00:00:00 | 1628K | Staging path not writable; no download |
| 29220 | — | CANCELLED | 00:00:00 | — | Superseded staging submission |
| 29221 | — | CANCELLED | 00:00:00 | — | Superseded RTX control submission |
| 29222 | P | FAILED 1:0 | 00:01:20 | 2235644K | Rebuilt compiler, missing raw pointer primitives |
| 29223 | P | COMPLETED 0:0 | 00:00:02 | 7980K | Retrieved checker and path diagnostics |
| 29224 | P | COMPLETED 0:0 | 00:00:49 | 2182924K | Minimal and base audit checker failures distinguished |
| 29225 | P | COMPLETED 0:0 | 00:00:01 | 188904K | Byte probe checked/built; executable bit not set, no run |
| 29226 | P | COMPLETED 0:0 | 00:00:01 | 188220K | Typed byte probe run PASS |
| 29227 | P | COMPLETED 0:0 | 00:00:02 | 191192K | Stack mmap probe run rc139 |
| 29228 | P | COMPLETED 0:0 | 00:00:01 | 192020K | Heap symbol mmap probe run PASS |
| 29229 | P | COMPLETED 0:0 | 00:00:24 | 1888472K | Raw memory contract rc4, missing import, c++ rc127 |
| 29230 | — | CANCELLED | 00:00:00 | — | Superseded oversized diagnostic allocation |
| 29231 | P | COMPLETED 0:0 | 00:00:01 | 188660K | u16/u32 neighboring byte overwrite localized |
| 29232 | D | COMPLETED 0:0 | 00:01:34 | 2186444K | Raw contract PASS; missing effect/linear checker diagnostics |
| 29233 | P | COMPLETED 0:0 | 00:00:21 | 1046348K | Raw contract PASS; mmap missing Mut |
| 29234 | D | COMPLETED 0:0 | 00:01:35 | 2173472K | Raw/mmap/CPU tail PASS; hal cast and tokenizer ownership diagnostics |
| 29235 | P | COMPLETED 0:0 | 00:00:01 | 7816K | Overlay, no writable Ceph |
| 29236 | D | COMPLETED 0:0 | 00:00:27 | 10472K | Writable Ceph confirmed; c++/jq/git-lfs/curl absent |
| 29237 | D | COMPLETED 0:0 | 00:00:01 | 11796K | Git LFS tool chunk 00 SHA verified |
| 29238 | D | COMPLETED 0:0 | 00:00:01 | 11556K | Tool chunk 01 SHA verified |
| 29239 | D | COMPLETED 0:0 | 00:00:00 | 11552K | Tool chunk 02 SHA verified |
| 29240 | D | COMPLETED 0:0 | 00:00:00 | 11556K | Tool chunk 03 SHA verified |
| 29241 | D | COMPLETED 0:0 | 00:00:00 | 3396K | Tool chunk 04 SHA verified |
| 29242 | D | COMPLETED 0:0 | 00:15:05 | 4207124K | Stage, LFS fsck, SHA256SUMS verified |
| 29243 | D | COMPLETED 0:0 | 00:02:27 | 3517528K | Full CPU adversarial PASS and audit check/build PASS; C++ tools rc127 |
| 29244 | — | CANCELLED | 00:00:00 | — | Superseded RTX job; no GPU used |
| 29245 | D | COMPLETED 0:0 | 00:03:44 | 34636K | LFS progress, no original prompt corpus found in scoped Ceph search |
| 29262 | D | COMPLETED 0:0 | 00:00:02 | 9644K | LFS bytes 16341738616, staged tokenizer 3475806 bytes |
| 29263 | D | FAILED 1:0 | 00:00:00 | 652K | Previous job's `/tmp` executable not visible; no CLI run |
| 29264 | D | FAILED 1:0 | 00:01:41 | 1918296K | Initial smoke check failed before executable; subsequent single-disposal check passed |
| 29265 | — | CANCELLED | 00:00:00 | — | Superseded pending runtime check |
| 29266 | D | FAILED 1:0 | 00:01:45 | 1850704K | Smoke check/build passed, argc guard rc1 |
| 29284 | D | COMPLETED 0:0 | 00:01:49 | 1915092K | Actual staged Granite tokenizer smoke PASS |

## Scheduled continuation — no fabricated comparison

At the snapshot, R740 was **ALLOCATED** (both `gpu:quadro_rtx_8000:1` and
`gpu:a5000:1`); R770 was **DRAINING** (`gpu:1`, L4); both Sparks were
DRAINED. No node configuration was modified. The original R740 control was
superseded by the artifact-preserving job below. Slurm reported the earliest
start estimate for 29269 as **2026-11-27 16:40** (scheduler estimate, not a
reservation guarantee). R740 is used for both CUDA legs; R770 is untouched.

| Submitted job | Requested lane | State at snapshot | Purpose / gate |
|---|---|---|---|
| 29268 | D, CPU only | PENDING | Read back staged manifest/revision after completed stage |
| 29269 | R740 RTX 8000 sm_75 | PENDING | Rebuild and run actual CPU/CUDA adversarial test; check shared model visibility; recover exact `/tmp/v2_480_prompts.txt` if visible |
| 29285 | D, CPU only | PENDING | Current audit argument built-ins check/build/refusal controls and durable CPU witnesses |
| 29286 | D, CPU only | PENDING | Transfer SHA-checked tool kit for missing C++ dependencies |
| 29288 | D, CPU only | PENDING (afterok 29286) | Compile exact C++23 tools, run cmp selftest and independent tokenizer smoke; no Python/Rust/Julia |
| 29297 | D, CPU only | PENDING (afterany 29269, afterok 29288) | Generate reference token IDs for the original 480 prompts only |
| 29298 | R740 RTX 8000 sm_75 | PENDING (afterok 29269 and 29297) | First 480 × 128 BF16 CUDA **reference**, no preclaimed comparison |
| 29299 | R740 A5000 sm_86 | PENDING (afterok 29298) | Same commit, same weights and SHA-verified prompt oracle; `cmp_61440.cpp` against 29298 |
| 29304 | D, CPU only | PENDING (afterany terminal predecessors) | Collect accounting, full artifact manifests, SHA checks, comparator verdict and reference TSV path |

The original 480-prompt file was not present in the scoped t560 worktree
search or the Ceph search. No replacement corpus was invented. The queued
RTX control will recover the existing file only if it is visible on that
worker, preserve its bytes, and record SHA256. The CPU oracle job refuses
missing input or a count other than exactly 480; dependent GPU audits are
cancelled on invalid dependency rather than consuming GPUs without inputs.
The C++ compiler and jq were absent on both CPU images, so tool bootstrap
is isolated under the task shared directory with the release archive SHA
pinned. The C++23 comparator and oracle are not yet built/executed; their
pending job will report actual results. The two GPU audit scripts resolve
expected input hashes from the completed predecessor manifests and verify the
candidate bytes before inference/comparison.

The intended Mac-lane reference is
`/orangefs/training/llm-granite-n10/audit-29298/cuda.tsv` with its sibling
`SHA256SUMS`, **only if job 29298 succeeds**. It does not yet exist as a
completed reference. The terminal collector will publish its accounting
artifact in the shared stage directory at
`/orangefs/training/llm-granite-n10/cluster-receipt-29304.txt` plus
`.sha256`. The comparator's first divergent tuple, if any, is retained;
it is not by itself a causal diagnosis and no automatic fix is applied.

**Prediction:** untested, neither held nor refuted. **Compared-artifact
SHA256:** unavailable because no cross-GPU pair has run. The above hashes
are actually executed CPU/scoped tokenizer witnesses, not surrogate GPU
parity. The final N-10 receipt and PR review gate remain pending terminal
GPU results; branch is pushed and no merge has been performed.
