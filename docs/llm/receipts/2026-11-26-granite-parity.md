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

## Coordinator-approved corpus registration (before any Granite GPU audit)

**Protocol anomaly:** the original Qwen3 61,440 audit corpus
(`/tmp/v2_480_prompts.txt`) was never committed and was not preserved; the
reference JSON retains output token IDs only. The coordinator confirmed it
is irrecoverable and authorized a new, deterministic Granite corpus. This
changes the input corpus, not the registered within-Granite cross-backend
prediction or its 480 × 128 shape. It is not a reconstruction of the Qwen3
corpus. Corpus recovery is closed.

- Repository corpus: `tests/llm/data/granite_480_prompts.txt`.
- Generator: `tests/llm/data/generate_granite_480.cpp` (C++23, 20 subjects ×
  24 fixed templates, subject-major order; ASCII, LF termination, no RNG,
  clock, external input, or platform-dependent arithmetic).
- Corpus SHA256: `e14ac28d67b08146f5640c08a820620ceffcf0734527e1699d158245a58f85bf`.
- 480 nonblank **distinct** records, 26,568 bytes. The generator ran on D
  in CPU-only Slurm job **29549**, COMPLETED 0:0, 00:01:18, batch MaxRSS
  1226.50M. Its exact compressed output was read back from 40 numbered
  accounting steps, decoded and SHA256-verified on t560. That output, not
  a separately constructed approximation, was committed.
- Staged corpus: `/orangefs/training/llm-granite-n10/granite_480_prompts.txt`;
  sidecar `.sha256` written by the generator job. The reference job verifies
  the registered digest and compares the staged bytes against the pinned
  Git blob. Both the reference and audit job headers carry the digest;
  every numerical audit additionally checks the oracle manifest and actual
  staged corpus bytes against it before build/inference.
- Registration commit `094ec9e472b7f0313efbf13049c9b5ba9ef90888` was pushed
  **before** the new GPU audit submissions. Original numerical audit source
  stays `ad4a505c1931d48552397089ca11d7cfaeabd988`.

### Superseded chain terminal evidence

- **29518** D: COMPLETED 0:0, 00:01:31, batch MaxRSS 1536384K. The repaired
  oracle smoke and comparator self-test both passed. Comparator SHA256:
  `4072e34b5bb725bfcf6c5f09cf95a3783ecb208bd52fc59f0cb4cb2d7afecb8a`;
  oracle SHA256:
  `491f4baf489289a3003529fbc63a621842d22ea07b28b08d3b9d61f85a2cda31`.
- **29519** D: FAILED 1:0, 00:01:16, batch MaxRSS 4192264K; the original
  prompt file was not staged. 29520–29522 were canceled by dependency;
  task-owned collector 29523 was canceled. No numerical audit ran.
- **29516** R770: TIMEOUT, 01:00:17, batch MaxRSS 3462420K. CPU control
  recorded `granite-tail-cpu-pass`. Recovery job **29550**, R770 L4,
  COMPLETED 0:0, 00:00:01, batch MaxRSS 25948K, recovered the shared copy of
  the CUDA log from 29516 and recorded `CUDA_TARGET sm=89 driver=13020`,
  `GRANITE_ARGMAX_TAIL_CUDA_PASS`, and
  `LLM_V2_ADVERSARIAL_FTZ_PASS backend=cuda fma_hits=1112 subnormals=0`.
  All 11 operations were reported with zero subnormal outputs. This is
  affirmative numerical control evidence, not a successful batch exit;
  the cause of the post-PASS timeout has not been established. Probe GPU:
  NVIDIA L4, compute capability 8.9, 23034 MiB, driver 595.71.05. The audit
  does not wait for a redundant L4 control. RTX control remains required.

### Registered-corpus replacement chain

| Job | Node/resource | State at controller readback | Dependency |
|---|---|---|---|
| 29549 | D CPU-only | COMPLETED 0:0 | Corpus generated and staged |
| 29550 | R770 L4, 2G | COMPLETED 0:0 | Prior L4 control log recovered |
| 29551 | D CPU-only, 6G | RUNNING | Replaces failed 29519; uses verified tools 29518 and registered corpus |
| 29552 | R740 RTX 8000, 12G | PENDING (ReqNodeNotAvail) | Replaces canceled 29269; obsolete corpus recovery removed |
| 29553 | R770 L4 sm_89, 48G | PENDING (Dependency) | afterok 29551; first audit |
| 29554 | R740 RTX 8000 sm_75, 48G | PENDING (Dependency) | afterok 29552, 29553 |
| 29555 | R740 A5000 sm_86, 48G | PENDING (Dependency) | afterok 29553 |
| 29556 | D CPU-only, 512M | PENDING (Dependency) | afterany 29551, 29552, 29553, 29554, 29555 |

Both R740 comparisons and the later Mac lane use the same registered corpus
and conditional reference `/orangefs/training/llm-granite-n10/audit-29553/cuda.tsv`.
Collector: `/orangefs/training/llm-granite-n10/cluster-receipt-29556.txt`.
All new jobs are tagged `lane:llm-granite`, R770 requests stay under 56G,
no new jobs on dl380p or 5860, no node state changes, and coordinator-owned
29217 remains RUNNING and untouched. The **prediction remains NOT TESTED**;
no final compared artifact hashes/PR yet.

**Reference gate completed:** 29551 finished `COMPLETED 0:0`, elapsed
00:02:01, batch MaxRSS 6289988K, recording `registered-corpus-sha-` with
exact registered digest and `GRANITE_REFERENCE_480_PASS`. The offline 480-ID
reference is `/orangefs/training/llm-granite-n10/granite_480_prompt_ids.tsv`,
SHA256 `739880da2d40ff74b9a409f8a76d22d22ebc146b1afdb14007ae095512b8c22e`.
At the next controller readback 29553 was **RUNNING** on
`gpuorangefs-r770-proxmox`. This releases the L4 leg, not a completed audit
or comparison. Both R740 audit jobs still wait for its successful result.

## Resume after t560 reboot: completed L4 reference, process-exit hang

**OBSERVED (coordinator):** the L4 leg 29553 completed all 480 prompts;
`run.log` ended `AUDIT_61440_DONE backend=cuda new_prompts=480 TOTAL_CUDA_FMA_MIN_NORM_HITS=0`.
The process did not exit: thread-group leader Z, live `cuda00001800007`
thread blocked in poll, GPU utilization 0%; `srun` waited indefinitely.
The coordinator preserved the outputs before cancellation in
`/orangefs/training/llm-granite-n10/l4-ref-29553/`.

**OBSERVED (independent Slurm readback):** job 29565 on D, CPU-only,
COMPLETED 0:0, 00:00:02, batch MaxRSS 13092K, verified every preserved
manifest payload, exact reference digest, 481 lines (one header, 480
indexed prompt records of 128 tokens), and the exact completion marker.
Preserved hashes:

| Artifact | SHA256 |
|---|---|
| cuda.tsv | `74b22db3f82221c5550593f2ce73f9b8430e8723ca219e45ce73b62b6c7c5fc5` |
| run.log | `cee2bfcb33d786b04072f18f2062fae751dbe65c74c0a6d0f6ff8da957b21eef` |
| inputs.sha256 | `d39c72d31dcff7fbe25ba53cf1509fa0c0835c55adde41aa92268f9c062ecb85` |
| build-audit.log | `0d7d30c472a768d3ef7882fa8a23be8c5c463637890d96ab2d8dc9590951bbb0` |

The run header reports binary SHA256
`8c74378a189919e4733847a100a72e976120f4b48b261ccdffcbe25e08bbd04c`
(the binary itself is not listed in the preserved manifest). Job 29553 was
canceled after preservation: elapsed 02:26:23, batch MaxRSS 15042524K,
`granite-run` step elapsed 02:23:50, MaxRSS 13666696K. This is complete
numerical reference evidence, **not** a successful Slurm termination.

**OBSERVED (source inspection):** at the pinned audit commit, the entry
trampoline returns from main using Linux x86-64 syscall 60 (`exit`, only
calling thread). `nc_emit_exit_code`, alternate trampoline and heap-reserve
failure paths likewise emit 60; the imported `native::encode::emit_exit`
helper also emits 60. This explains why a driver-owned thread can survive
main's return. **NOT ESTABLISHED:** whether the earlier 29516 timeout had
exactly the same thread-group state (numerical PASS was recovered, but no
live-thread observation from that job).

### A. Watchdog reroute, unchanged numerical pin

Canceled only owned jobs 29552–29556, by ID, including stuck 29553. New
runner commit `8f96c1bc4` leaves the numerical audit pin
`ad4a505c1931d48552397089ca11d7cfaeabd988` unchanged. It launches the pinned
binary directly in the Slurm batch allocation, retains the exact PID, polls
for the done marker, tokenizer PASS and validated output record count/shape,
then allows 10 seconds for normal teardown. Only if the process remains
alive and the output is still complete does it signal that exact child,
record process status, require the resulting SIGKILL status and revalidate
the file, and report `r-exit-hang-workaround`. The CUDA tail control similarly
requires the tail and full FTZ-suite PASS markers plus a complete AFTZ binary
(header/record count/expected byte length). Genuine failures remain failures.
The comparator runs unchanged against the independently pinned L4 TSV.

| Job | Resource | State at submission/readback | Dependency |
|---|---|---|---|
| 29559 | R740 RTX 8000 control | PENDING (ReqNodeNotAvail) | None |
| 29560 | R740 RTX 8000 audit, 48G | PENDING (Dependency) | afterok 29559 |
| 29561 | R740 A5000 audit, 48G | PENDING (ReqNodeNotAvail) | None |
| 29562 | D CPU collector | PENDING (Dependency) | afterany 29559, 29560, 29561 |
| 29565 | D CPU reference verification | COMPLETED 0:0 | None |

Neither numerical comparison depends on 29553's job exit. The preserved
reference, also for the Mac lane, is
`/orangefs/training/llm-granite-n10/l4-ref-29553/cuda.tsv`. Pending collector
`/orangefs/training/llm-granite-n10/cluster-receipt-29562.txt` verifies that
reference directory and both candidate manifests. R740 is still unavailable
for new allocations at readback; no node state changes, no allocations on
dl380p or 5860. Job 29217 completed independently; it was never modified.
**NOT ESTABLISHED:** cross-GPU identity until candidate audits and comparator
outputs are collected.

### B. Separate compiler correction

Branch `fix/linux-x86-exit-group`, based on origin/main
`ad5beda241dc61f2abfc5ba09355c8d74401fa13`. Test-only source commit
`648d25bfdb959fe7407e81fb99b7d0f75bdd3789`; patch changes all direct process
exit paths in `codegen_x86_linux.sio` and the shared x86 `emit_exit` helper in
`encode.sio` to syscall 231, with corresponding encoder byte expectation.
Both post-fork execve-failure child-only `_exit(127)` calls retain syscall
60. Regression loads a small DSO whose pthread signals readiness via a pipe
then blocks on a second pipe; Sounio main returns 37. Gate requires exit 37
within 5 seconds. Before job 29563 and after job 29564 (D CPU-only) both build
Madaros from their respective source checkout; after depends on the
reproduced before failure. **NOT ESTABLISHED:** before/after execution
results until those jobs finish. No compiler changes on the Granite branch.

### B. Compiler regression terminal results (OBSERVED)

The initial baseline allocation **29563** built Madaros successfully but
failed its harness with status 126 (generated probe lacked executable bit;
00:02:00, batch MaxRSS 2209536K). Its dependent 29564 was canceled. This
was not a reproduced thread hang. The harness was corrected to `chmod +x`
and rerun with the same unpatched compiler source and identical regression
source; the corrected harness alone was overlaid at
`46f3c22e84c42a210c237c344446213e641605d9`.

| Job | Node | Source | State | Wall time | Batch MaxRSS | Program result |
|---|---|---|---|---|---|---|
| 29566 | D | `648d25bfdb959fe7407e81fb99b7d0f75bdd3789` | COMPLETED 0:0 | 00:02:17 | 2475884K | readiness marker; timeout 124, expected exit 37 |
| 29567 | D | `46f3c22e84c42a210c237c344446213e641605d9` | COMPLETED 0:0 | 00:01:48 | 2194760K | readiness marker; exit 37 within 5 seconds, PASS |

Both used source-built Madaros, the same Sounio and C++ probe sources,
CPU-only jobs tagged `lane:llm-granite`. The before batch succeeded because
it asserted the expected regression failure. The after batch recorded
`EXIT_GROUP_THREAD_RETURN_PASS` and `EXIT_GROUP_AFTER_PASS`.

| SHA256 artifact | Before (29566) | After (29567) |
|---|---|---|
| Madaros | `82e085c09464e7446d6345a14324802ea68bf8c66a3988d4d3f23bda9da1e1d1` | `7b4a7e53da0e13fc2c0459eb086faaad33343d4268e9d4f335fbdd06f1ff0628` |
| probe executable | `6919e16b848cad4e4a397db95f30dad4e5fb64fab01021eed5fe27f050f56f3d` | `46992b58b4f926f19b527d711b845ac27eb0c3d028d73f054530dacde2ce3804` |
| helper DSO | `88028ef48dd4b065c996b1ecd2621cc64e67340db1e2851dbc901d925373428d` | `6bc6f17188bc25a90f9bb891af22a40776300bdbb22c3599cb5b68b308d16ace` |

The DSO build digests differ, but both builds used the same helper source
and compiler and emitted the readiness marker. Logs and manifests:
`/orangefs/training/llm-granite-n10/exit-group-29566/` and `exit-group-29567/`.
The compiler change and receipt were pushed to `fix/linux-x86-exit-group`
(head `d1d7dfb61dda75fca9e560c824c8ff089b6adcde`); PR
[#3752](https://github.com/Sounio-lang/sounio/pull/3752) targets `main` and
was not merged. **OBSERVED:** the blocked-thread process-return regression
fails before and passes after, with the patched Madaros built from source.
**NOT ESTABLISHED:** actual patched-compiler CUDA inference, the exact
cause of job 29516's timeout, and the cross-GPU bit-identity prediction
while candidate jobs remain pending. The original audit pin is unchanged.

## 2026-11-27: R740 numerical campaign failure (coordinator audit)

**OBSERVED (coordinator):** both R740 legs 29560 (Quadro RTX 8000, sm_75)
and 29561 (RTX A5000, sm_86) printed `GRANITE_TOKEN_IDS_480_PASS`, the
header, `[audit61440] resume_skip=0 max_tokens=128 max_seq=512 max_prompts=480`,
then `AUDIT_61440_FAIL`. Each TSV contained only a header, no prompt
records. Both audit processes were Zl with GPU utilization 0%. Unlike
29553's completed L4 reference, this is a numerical-run failure, not a
completed result. The existing exit watchdog was incomplete: it recognized
only successful `AUDIT_61440_DONE`, leaving failure-marker zombies alive.

**OBSERVED (Slurm readback):** 29559 RTX CPU/CUDA 49,155-tail control
COMPLETED 0:0, 00:02:22, batch MaxRSS 3441356K; records `granite-tail-cpu-pass`,
`r-exit-hang-workaround`, and `granite-tail-cuda-pass`. Both audit legs were
RUNNING for 03:04:05 before cancellation; 29562 was PENDING. Canceled only
29560, 29561, 29562 by ID. Batch MaxRSS was unavailable in the terminal
accounting readback for the two canceled legs. The controller subsequently
reported R740 `IDLE+DRAIN+DYNAMIC_NORM` with reason `Kill task failed
(JobId=29561 StepId=batch) [root@2026-11-27T04:16:35]`. No node state was
changed by this lane; coordinator notified. No new heavy job on dl380p or
5860.

**NOT ESTABLISHED:** root cause of `run_61440_cuda` returning false. Source
inspection shows the absent `backend=cuda_audit` line precedes first-prompt
processing and is conditional on both `qwen3_cuda_runner_is_ok(&rt)` and
`page_arena_is_ok(&work_arena)` in `run_61440_cuda`; failure during runner
initialization is a hypothesis, not a proven cause. Prior successful Qwen3
RTX 8000 audit and Granite L4 audit are not evidence of the failed branch.
No tuple comparison was possible, so prediction is **NOT TESTED**.

### Binary identity anomaly

The pinned source is still `ad4a505c1931d48552397089ca11d7cfaeabd988`.
Accounting recorded audit binary digest prefixes `8c74378a1899` (L4),
`5909057062c7` (RTX 8000), `d65d3dbf04cb` (A5000). All three registered the
same corpus hash. **NOT ESTABLISHED:** whether those executable differences
are path metadata, compiler nondeterminism, or actually different emitted
instructions. The original L4 executable and Madaros binary were not
retained by the earlier preservation (29639 confirmed unavailable); its
verified build log, `inputs.sha256` and numerical TSV remain. CPU recovery
job 29641 on D could not find the two failed R740 binaries, logs or TSVs in
the shared stage; the R740-local recovery job 29638 is queued on the drained
node. The coordinator's original observations above are retained separately
from artifacts actually recovered by this lane. New failure traps preserve
full audit and compiler files, inputs, logs and partial TSVs with a
failure-specific SHA256 manifest, never treating partial TSVs as reference.

### Diagnostic-only chain (no audit re-pin)

The watchdog now detects `AUDIT_61440_FAIL`, `AUDIT_61440_INIT_FAIL` and
other uppercase `*_FAIL` markers, kills only the owning child PID if
necessary, records `r-audit-fail`, and exits nonzero. The tail runner has
the corresponding failure-marker guard.

Branch `diag/granite-cuda-init` at
`a625788c03f20f687b5670e7f8d1cce327f88b51`, based on the pinned audit source,
instruments session open, allocation and copy return codes, runner resources,
and the precise failure branches in `run_61440_cuda`. It retains the original
480-prompt tokenizer oracle check but runs at most one prompt for diagnosis;
its distinct success marker is `GRANITE_DIAG_FIRST_PROMPT_PASS`, never the
registered audit DONE marker. Build once on D, then use the same SHA-verified
CPU-built diagnostic ELF on all three GPUs, to avoid confounding devices and
compilers. No changed arithmetic is admitted to the audited source yet.

| Job | Node / resource | Role |
|---|---|---|
| 29638 | R740 CPU-only, oversubscribe | Original artifact recovery, pending on drained node |
| 29639 | R770 CPU-only | L4 artifact recovery, COMPLETED 0:0 |
| 29640 | D CPU-only | Source-built Madaros and diagnostic ELF; checks |
| 29641 | D CPU-only | Failed R740 shared-stage artifact recovery; COMPLETED 0:0, missing artifacts |
| 29642 | R740 sm_75, 24G, oversubscribe | Diagnostic, afterok 29640 |
| 29643 | R740 sm_86, 24G, oversubscribe | Diagnostic, afterok 29640 |
| 29644 | R770 sm_89, 24G, oversubscribe | Same diagnostic ELF positive control, afterok 29640 |
| 29645 | D CPU-only | Same-source same/path-different-build binary comparison |
| 29646 | R770 CPU-only, 12G, oversubscribe | Independent binary reproducibility experiment |

All jobs tagged `lane:llm-granite`; no source re-pin, no parity assertion
from these diagnostics. If an audited-source fix is required, preregister
it and rerun L4, RTX 8000, and A5000 using the identical source revision
and preferably a single SHA-verified Linux ELF.

### 2026-11-27 follow-up: binary provenance established; R740 execution blocked

**OBSERVED:** diagnostic build 29640 COMPLETED 0:0 on D, 00:01:59,
batch MaxRSS 3532956K, with source-built Madaros, all three Sounio checks,
and diagnostic ELF SHA256
`532ff03113dad862ba271def036242211f5afc7e3cf71f08501333b1cb4afafe`.
L4 job 29644 ran that exact ELF (R770, 00:02:11, MaxRSS 16088868K,
COMPLETED 0:0), selecting sm_89 and 22563 MiB free at entry. It verified
the 480-tokenizer-ID oracle, reported session/module readiness, and all four
shard allocations/copies with return code 0 (4974661520, 1405169256,
4991447808, 4970460032 bytes). Work (3801088), K/V caches (83886080 each)
and RoPE (262144) allocations also returned 0. Runner resource checks were
all 1, context sync 0, work arena 1, and
`GRANITE_DIAG_FIRST_PROMPT_PASS backend=cuda new_prompts=1 TOTAL_CUDA_FMA_MIN_NORM_HITS=0`.
The known pinned-exit bug required child SIGKILL (137) after that diagnostic
marker; it is not a full numerical audit or an exit-group-fixed run.

**OBSERVED:** the binary anomaly is reproducible without CUDA or a GPU
allocation. Independent CPU jobs 29645 (D, COMPLETED 0:0, 00:04:28,
MaxRSS 4676804K) and 29646 (R770, COMPLETED 0:0, 00:02:22,
MaxRSS 4627008K) used exact pinned source. Both built source-derived
Madaros twice in independently located checkouts. All four compiler
executables had the same SHA256:
`5df8c456cdb5db06bf48485b4ebfdbae518e39be4d50efa798b5d00ebdbd3659`.
Repeated audit builds at one checkout path were bit-identical, including
when output filenames changed. Moving only the source checkout from A to B
changed the executable.

| CPU build node | A1 = A2 SHA256 | B1 = B2 SHA256 |
|---|---|---|
| D | `85143950484b178283d5e95e714be119ca403b7bcd559ccb9d77fd1f94c299b6` | `cfe4ee4b1e850a05baa87ab7f79ff0b9758a6e8be2b82fa03e9d7d40dfe9b644` |
| R770 | `dae790d39b35473fb75a71fd0028c6ce94bbeae3f21b347e4da0d80ada3a8e19` | `0b19f11898e4fa9eaeb03bfdb68e0532c059fe4717710e239119f292609517ee` |

CPU verification job 29647 (D, COMPLETED 0:0, 00:00:03, MaxRSS 20024K)
counted 1280 absolute checkout-path occurrences in each audit ELF. Replacing
only the same-length absolute checkout path made the entire ELF identical
within each node and across D/R770 (`PATH_ONLY_BYTE_DIFF-29645`,
`PATH_ONLY_BYTE_DIFF-29646`, `CROSS_NODE_PATH_ONLY_BYTE_DIFF`,
`CROSS_NODE_MADAROS_IDENTICAL`). First differing byte offsets were 4519,
4966, 5413, 5742, 6278 (octal 101→102, ASCII A→B). Thus the different
hashes arise from emitted absolute source paths, not GPU detection or
nondeterministic Madaros binaries in these controlled builds.

CPU reconstruction job 29648 (D, COMPLETED 0:0, 00:02:15, MaxRSS 6398424K)
used that exact compiler and the original absolute checkout paths
`/tmp/llm-granite-parity-<job>/src`, with the original source and options.
It reconstructed the historical L4 full digest and both R740 observed
48-bit prefixes. Full **reconstructed** digests:

| Historical leg | Reconstructed SHA256 | Agreement with archived observation |
|---|---|---|
| 29553 L4 | `8c74378a189919e4733847a100a72e976120f4b48b261ccdffcbe25e08bbd04c` | Exact full SHA256 |
| 29560 RTX 8000 | `5909057062c7fb2fe8ab3258f4abe5d98ed8e97d2e4af98a15a3d6a413506550` | Exact recorded prefix |
| 29561 A5000 | `d65d3dbf04cb6e1b8e950764249d66bc0f724ab07af55141f32c3195918108e2` | Exact recorded prefix |

Replacing only the historical source path in the reconstructed L4 ELF
makes it byte-identical to each reconstructed R740 ELF. The original R740
full hashes and raw binaries remain unavailable, so this is a successful
reconstruction against recorded prefixes, not a claim of actual full
R740 file readback. Job 29651 (D, 00:00:01, MaxRSS 6708K, COMPLETED 0:0)
verified the original L4 `inputs.sha256` against staged actual bytes; the
reconstructed L4 `build-audit.log` is byte-identical to the archived log,
SHA256 `0d7d30c472a768d3ef7882fa8a23be8c5c463637890d96ab2d8dc9590951bbb0`.
Evidence directories: `binary-repro-29645/`, `binary-repro-29646/`,
`binary-compare-29647/`, and `binary-reconstruct-29648/` under the shared
stage, with raw ELF hashes and log artifacts. The reconstructed L4 ELF
can serve as a single byte-identical executable for future pinned CUDA
legs without rebuilding on the GPU host, provided the failing branch
requires no audited-source change.

**NOT ESTABLISHED:** why the two historical R740 initializations failed.
The instrumented diagnostic is proven executable on L4, but R740 remains
`IDLE+DRAIN+DYNAMIC_NORM` (`Kill task failed (JobId=29561 StepId=batch)`).
No node state intervention is authorized. The two queued diagnostic jobs
29642/29643 were canceled and replaced with a bounded child-kill version
(10-second reap check rather than indefinite `wait`). Their replacements
use same verified diagnostic artifact, are 15-minute jobs, with
`OverSubscribe=YES`, exact named GRES and `lane:llm-granite`:

| Job | Node/GRES | State at readback |
|---|---|---|
| 29638 | R740 CPU binary recovery | PENDING (ReqNodeNotAvail) |
| 29649 | R740 RTX 8000 diagnostic | PENDING (ReqNodeNotAvail) |
| 29650 | R740 A5000 diagnostic | PENDING (ReqNodeNotAvail) |

The campaign runner and tail runner also bound child reap time and record
`r-child-kill-timeout` as a failure rather than waiting indefinitely.
**No source fix, new numerical pin, or audit rerun is authorized by the
current evidence.** The failure-branch diagnostic must execute on R740
first; if a source fix is required, all three legs will be rerun from a
new preregistered pin. Cross-GPU prediction remains **NOT TESTED**.

## 2026-11-27 05:55 UTC: VRAM-contention prediction (registered before diagnostic submission)

**OBSERVED (coordinator):** R740 is double-booked between Slurm and
Kubernetes GPU services outside Slurm. The coordinator identified pods
`beagle/llamacpp-coder-a5000` holding the RTX A5000 and
`vllm-r1-rtx8000/llamacpp-prover` holding the Quadro RTX 8000. The Slurm
worker's `nvidia-smi` reported 20549 MiB used on the A5000 and 37185 MiB
used on the RTX 8000 with no living Slurm audit process. Corresponding
free VRAM was approximately 3.5 GiB and 11 GiB, respectively. The
coordinator killed the remaining live CUDA threads of 29560/29561
(558836/558844); this lane did not perform that cleanup. R740 remains
`IDLE+DRAIN` with `Kill task failed (JobId=29561 StepId=batch)`. Resuming the
node or changing the Kubernetes services is the founder's decision, not
an action for this lane.

**PREDICTION (preregistered):** conditional on the same contention at
execution, the first failed device allocation on **both** sm_75 and sm_86
will return CUDA_ERROR_OUT_OF_MEMORY (`CUresult=2` from `cuMemAlloc_v2`),
leading to runner initialization failure and zero prompt records. The
four verified Granite BF16 shard sizes sum to 16,341,738,616 bytes
(15.2194 GiB), before the work/KV/RoPE allocations or module/context memory.
The A5000 is predicted to fail during shard loading, plausibly the first
4,974,661,520-byte shard; the RTX 8000 cannot hold all four at the observed
free capacity, so a later shard allocation is a likely failure. Exact
allocation index is not preregistered because service occupancy can vary.
The prior L4 control is consistent with this explanation but does not
prove the R740 cause.

Diagnostic-only source `ae078834a4e8a3204200509dfd179aab9b2d0124` adds
64-bit `cuMemGetInfo_v2` free/total bytes and query CUresult after module
initialization, before the first shard and before every explicit allocation.
The diagnostic allocator already prints the exact requested bytes and
`cuMemAlloc_v2` return code. Both replacement R740 submissions carry this
prediction in the job header; the numerical audit source pin and corpus
remain unchanged. The old pending 29649/29650 were canceled to replace
their immutable scripts before either ran. No Kubernetes service or node
state was modified.

**NOT ESTABLISHED:** the actual R740 failing operation or causal diagnosis.
Comparison criteria: the target context must be the requested sm, the
`cuMemGetInfo_v2` query must succeed under comparable insufficient free
memory, the first failed allocation must return 2, and the terminal audit
failure must leave only the header. A different first failure code under
these conditions refutes the preregistered allocation mechanism for that
run. If occupancy has changed or the measurement/target context is absent,
report the precondition as not established, not a historical refutation.
No full numerical comparison may be inferred from a diagnostic.
