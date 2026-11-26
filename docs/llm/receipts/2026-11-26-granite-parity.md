# N-10 — Granite-3.1-8B cross-backend parity receipt (2026-11-26)

**Status at 13:08 UTC: CPU controls PASS; staged checkpoint verified; GPU
prediction NOT TESTED.** This is an interim accounting receipt, not a claim
of cross-GPU or Metal identity. The original dependency chain became dead when kit job 29286 failed; it has been superseded by the replacement chain recorded below.
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
Manifest readback job 29268 completed (00:00:03, MaxRSS 9512K, exit 0:0).
Revision: `4009206d5fc95d2e65a7b7633e159d6e97e25d35`; manifest SHA256:
`10edfe81703521bea67e7a285388590c6b740a639665411c76157d82a83609e6`.
The four shard SHA256 values, in order:

- `191c4e9c6263d9cf591104f2d16ab2c39dcc43c1ad0680cc5a34d5c86d61ee41`
- `c7c38b0d5a436775b09d764465ed6e6eb7a8c4e302d05e301e151c96e3076f22`
- `f02784b72391fa04e9b986313c1a1720ce88f0eb40f7ae81fa0daadc93049457`
- `9d86d201ff8e73d8a46e92b543c9dd44f133e60b35ccada4a76439af62f22212`

Direct visibility
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
- The current audit's `arg_count()` / `get_arg()` revision passed full
  recheck/rebuild and both CLI refusal controls in job 29285 (00:02:33,
  MaxRSS 3489640K, exit 0:0). Audit ELF SHA256:
  `08ac3a474a4a3c9a9c19e766874e2184386c6954a390cba6c51bd67c93613905`. A previous
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

## Original continuation — failed and superseded

At the snapshot, R740 was **ALLOCATED** (both `gpu:quadro_rtx_8000:1` and
`gpu:a5000:1`); R770 was **DRAINING** (`gpu:1`, L4); both Sparks were
DRAINED. No node configuration was modified. The original R740 control was
superseded by the artifact-preserving job below. Slurm reported the earliest
start estimate for 29269 as **2026-11-27 16:40** (scheduler estimate, not a
reservation guarantee). R740 is used for both CUDA legs; R770 is untouched.

| Submitted job | Requested lane | State at snapshot | Purpose / gate |
|---|---|---|---|
| 29268 | D, CPU only | COMPLETED 0:0 | Read back staged manifest/revision after completed stage |
| 29269 | R740 RTX 8000 sm_75 | PENDING | Rebuild and run actual CPU/CUDA adversarial test; check shared model visibility; recover exact `/tmp/v2_480_prompts.txt` if visible |
| 29285 | D, CPU only | COMPLETED 0:0 | Current audit argument built-ins check/build/refusal controls and durable CPU witnesses |
| 29286 | D, CPU only | FAILED 1:0 | Transfer SHA-checked tool kit for missing C++ dependencies |
| 29288 | D, CPU only | DependencyNeverSatisfied; subsequently CANCELLED | Compile exact C++23 tools, run cmp selftest and independent tokenizer smoke; no Python/Rust/Julia |
| 29297 | D, CPU only | Dead dependency; subsequently CANCELLED | Generate reference token IDs for the original 480 prompts only |
| 29298 | R740 RTX 8000 sm_75 | Dead dependency; subsequently CANCELLED | First 480 × 128 BF16 CUDA **reference**, no preclaimed comparison |
| 29299 | R740 A5000 sm_86 | Dead dependency; subsequently CANCELLED | Same commit, same weights and SHA-verified prompt oracle; `cmp_61440.cpp` against 29298 |
| 29304 | D, CPU only | Dead dependency; subsequently CANCELLED | Collect accounting, full artifact manifests, SHA checks, comparator verdict and reference TSV path |

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

The former intended reference (audit-29298/cuda.tsv) and collector
(cluster-receipt-29304.txt) are superseded and will not be produced by
those canceled jobs. Replacement paths are recorded in the recovery addendum. The comparator's first divergent tuple, if any, is retained;
it is not by itself a causal diagnosis and no automatic fix is applied.

**Prediction:** untested, neither held nor refuted. **Compared-artifact
SHA256:** unavailable because no cross-GPU pair has run. The above hashes
are actually executed CPU/scoped tokenizer witnesses, not surrogate GPU
parity. The final N-10 receipt and PR review gate remain pending terminal
GPU results; branch is pushed and no merge has been performed.

### Final source-only cleanup (13:12 UTC)

Commit `c2fbca468899f25227b6163b565736f3ec794ae3` removes one trailing
blank line from `top2.sio`; the numerical source is unchanged. Job **29308**
(`dl380p-dl380p-proxmox`, CPU-only) checked that library via the public
compiler wrapper: `COMPLETED 0:0`, elapsed `00:00:23`, batch MaxRSS `1022M`.
`git diff --check 575d1eb57..HEAD` is clean. Both GPU jobs remain pinned to
`ad4a505c1931d48552397089ca11d7cfaeabd988` for a matched-source comparison.

## Coordinator correction and dependency recovery (2026-11-26 14:33 UTC)

**Correction:** the original chain was dead, not merely queued. Kit job
29286 finished `FAILED 1:0` (elapsed 00:00:00, MaxRSS 9800K, D), and
29288 became `DependencyNeverSatisfied`; 29297/29298/29299/29304 consequently
could not run. Those five task-owned dead jobs were canceled; no GPU ran.

The accounting path for the original log was `/tmp/granite-cpp-kit-%j.log`.
Diagnostic job **29506** (D, 00:00:01, MaxRSS 10640K, exit 0:0) attempted
readback of `/tmp/granite-cpp-kit-29286.log`, but that prior job-local log
was no longer visible. It reproduced the exact checksum failure against the
staged kit and recorded the check output in `sacct` steps: `./SHA256SUMS:
FAILED`, all seven payload files `OK`, and `1 computed checksum did NOT
match`. The local original payload manifest confirms the cause: generating
`find . -type f ... | sha256sum > SHA256SUMS` created the output file before
enumeration, thereby hashing its own empty contents (`e3b0c442...`). The
incorrect manifest SHA256 was
`0917d6bd7d281c51dd2e921a6292dd6ba08b4207bebe6465c071a61f6a35d07f`.
This is a reproduced diagnosis, not a claim of having retrieved the absent
original stdout.

Repair job **29507** (D, 00:00:02, MaxRSS 8596K) independently verified all
seven payload hashes, preserved `SHA256SUMS.failed-29286`, and atomically
removed only the erroneous self-entry. Corrected manifest SHA256:
`0ea6574786cb3292aee211dd06e3dce0cf4458e8a2fb75d06497d4843ca24ff9`.
The subsequent `curl --version` preflight exposed a host/worker shared-library
symbol lookup error, so the job deliberately exited 127 rather than mark
the tool kit operational.

Job **29508** (D, 00:00:00, MaxRSS 15544K, exit 0:0) staged a statically
linked Debian `busybox-static` package (package SHA256
`598a3fd92bdafc34cd81196b2952ad36e910e47a4ecf162f8ce5e8e262598e53`,
executable SHA256
`7ae95c260622412ceba6eb9f45d68f616cdc091a1d71d6c0a08b5875bb2c31dd`)
into the existing private kit directory. It checked the `wget`/`xzcat`
applets and ran `jq --version` (1.7), recording
`KIT_STATIC_WGET_JQ_PASS`. All payload bytes were reverified. The new kit
manifest SHA256 is
`015380bcc3e907bd9d524a6d03f7c24c075a5ef16c581a753a9bd273c29330d1`.
The C++ tool bootstrap now uses this static download/decompression path
and still checks the pinned publisher SHA256 of the compiler archive before
extraction; it no longer calls the incompatible copied curl. No package was
installed on a node.

### Replacement chain (controller readback)

Both numerical audits and the C++ tool source checkout remain pinned to
`ad4a505c1931d48552397089ca11d7cfaeabd988`. The prediction and 480 × 128
shape are unchanged. Completed stage 29242 has aged out of the live
controller, so the new reference job checks its actual staged SHA256SUMS
rather than requiring a stale live dependency ID. At the readback:

| Job | Node/lane | State | Dependency |
|---|---|---|---|
| 29509 | D, CPU only | RUNNING | Corrected kit 29508 passed; compiler download/build in progress |
| 29269 | R740, RTX 8000 control | PENDING | Preserved existing control; scheduler estimate 2026-11-27 19:40 UTC |
| 29511 | D, CPU only | PENDING (Dependency) | afterany 29269; afterok 29509 |
| 29512 | R740, RTX 8000 sm_75 | PENDING (Dependency) | afterok 29269 and 29511 |
| 29513 | R740, A5000 sm_86 | PENDING (Dependency) | afterok 29512 |
| 29514 | D, CPU only | PENDING (Dependency) | afterany 29269, 29509, 29511, 29512, 29513 |

These are real replacement submissions with live dependencies, not completed
results. The C++ bootstrap is running; eventual build/test success is not yet
established. The intended replacement reference, **only if 29512 succeeds**,
is `/orangefs/training/llm-granite-n10/audit-29512/cuda.tsv`. The replacement
terminal collector path is
`/orangefs/training/llm-granite-n10/cluster-receipt-29514.txt` plus `.sha256`.
The original missing-prompt gate remains in force.

The coordinator restricted future work to D for CPU-only allocations and
R740 for the named GPU allocations. No new jobs were submitted to dl380p or
5860; prior job 29308 is historical. Coordinator job 29217 was not canceled
or modified and was read back RUNNING. No node state was changed. R770 is
untouched. Prediction remains **NOT TESTED**, and the final parity receipt
and PR gate are still outstanding.

## R770 L4-first reroute (2026-11-26, coordinator-authorized)

The preceding replacement table is historical: 29509 ultimately **FAILED
1:0**, not an active bootstrap. Task-owned jobs 29511, 29512, 29513 and 29514
were canceled in favor of the L4-first chain. No GPU audit output was produced
by that chain. Job 29269 remains queued; 29217 is the other lane's running
compiler job and was not canceled or modified.

`29509` ran 00:18:20 on D with batch MaxRSS 1778520K. Its comparator compiled
and passed `CMP_61440_SELFTEST_PASS`, SHA256
`f74a1e17ed38075f0934bfbb55097beb1c3aac92823aa9a913b414052f80acb3`.
The first independent tokenizer binary SHA256 was
`c016a340da84b10f48a216799ee72d042aecba08b8a0a434949323ca3664a891`;
its smoke test failed with `missing vocabulary entry`. Diagnostic job
**29517** (D, COMPLETED 0:0, 00:00:01, MaxRSS 28088K) found no missing merge
operands or products and located 14 missing single-byte alphabet symbols
(`À Á ñ õ ö ÷ ø ù ú û ü ý þ ÿ`) in the actual staged tokenizer vocabulary.
The failure is in `Tokenizer::load` (C++ independent oracle), which had
required every symbol in the 256-entry byte alphabet. The reference loader
now permits sparse byte tables and rejects an actual unrepresentable input
byte, rather than silently dropping it. Its regression smoke tests the
14 absent entries, rejection of an absent input byte, and original fixed
IDs. Oracle repair source: `baa3f359735f65e9d84455ae7f1366a5cc81724e`;
verification is pending (29518). The comparator and every numerical audit
remain at the original pinned source
`ad4a505c1931d48552397089ca11d7cfaeabd988`.

The controller reported R770 `IDLE+DYNAMIC_NORM`, `RealMemory=61440`,
`Gres=gpu:1` before submission. **29515**, CPU-only on
`gpuorangefs-r770-proxmox`, completed 0:0 (00:01:11, batch MaxRSS 2094284K).
It read the shared model directory, matched its pinned revision and exact
manifest SHA256 `10edfe81703521bea67e7a285388590c6b740a639665411c76157d82a83609e6`,
and verified every payload hash (`L4_WEIGHT_SHA256SUMS_PASS`). It also recorded
`L4_ORIGINAL_PROMPTS_NOT_FOUND`; the required original registered 480-line
corpus is still absent from shared staging and R770's searched locations.
The coordinator was notified. No substitute prompts were created. The
reference generation job deliberately fails if the original corpus is
still unavailable, so the first audit cannot run on a different dataset.

### L4-first chain: controller readback

| Job | Node / resource | State at readback | Dependency |
|---|---|---|---|
| 29515 | R770 CPU-only, 2G | COMPLETED 0:0 | Weight SHA256 preflight passed |
| 29516 | R770 L4 sm_89, 12G | RUNNING | afterok 29515 |
| 29517 | D CPU-only, 2G | COMPLETED 0:0 | Oracle diagnosis |
| 29518 | D CPU-only, 6G | RUNNING | Corrected oracle smoke and pinned comparator self-test |
| 29519 | D CPU-only, 4G | PENDING (Dependency) | afterok 29518; original prompt file independently required |
| 29520 | R770 L4 sm_89, 48G | PENDING (Dependency) | afterok 29516, 29519; first unpaired audit |
| 29269 | R740 RTX 8000 control | PENDING (ReqNodeNotAvail) | Existing control retained |
| 29521 | R740 RTX 8000 sm_75, 48G | PENDING (Dependency) | afterok 29269, 29520 |
| 29522 | R740 A5000 sm_86, 48G | PENDING (Dependency) | afterok 29520 |
| 29523 | D CPU-only, 512M | PENDING (Dependency) | afterany 29269, 29516, 29518, 29519, 29520, 29521, 29522 |

Every new job and retained 29269 carries `lane:llm-granite`. All R770 requests
are within the 56G ceiling. All CPU-only allocations other than the explicit
R770 preflight use D; no new jobs target dl380p or 5860. R740 is still busy
with 29217; the L4 audit has **no R740 dependency**. Both subsequent GPUs
compare directly with the exact L4 TSV and its completed artifact manifest.
Conditional Mac reference path (not yet produced):
`/orangefs/training/llm-granite-n10/audit-29520/cuda.tsv`. Collector output:
`/orangefs/training/llm-granite-n10/cluster-receipt-29523.txt` and `.sha256`.
No node state has been changed. Prediction remains **NOT TESTED**;
no divergence claim, final comparison hashes, or PR yet.
