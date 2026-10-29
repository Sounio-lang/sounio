# ROOT5 historical/current replay — prepared only

No compiler, census binary, test, array or bootstrap has been executed by this
harness. Source and recipe are prepared for root review; they are not qualified.
The actual raw hash remains null in INPUTS.json until a real ROOT5 RAW_READY
marker is admitted. No hash is inherited from an older compiler.

The12originals are copied byte-for-byte from whole-integrated01/source.
The same physical copy under root5-replay01/input/original is passed in both SDK
arms. Original SDK files and these copies are independently hash-checked.
The list contains the11compiler139cases from RESULT19392 plus the unchanged
PBPK original (historical compile code1). The latter is byte-identical to the
original used by root-pbpk-compile04, SHA5f230b6f...f10b1d0.

Each case has four independent arms: historical/current SDK times loop mode0/1.
Every arm captures check and compile separately, yielding96 phases. Compiler
errors and timeouts are data and do not abort the remaining arms. The array
recipe reserves1CPU/16GiB per case,3h, with concurrency2 (root maximum4).
The compiler deadline is1200s; timeout sends TERM then KILL after a30s grace.
No generated scientific program is executed.

## Admission

Run bind only after root explicitly authorizes this exact raw following:

1. Main build terminal COMPLETED Exit0, verified against the supplied Slurm job.
2. ROOT5/results/BUILD_STATUS exactly0.
3. All final build guards complete; SOURCE/COMPILER/SEED/CAPACITY are mandatory.
4. Root's seed smoke and loader admission.
5. Actual RAW_READY hash/path matching ROOT5/results/madaros.

bind additionally checks the private producer hash, final lean source hash,
SOURCE_HEAD equality (the caller cannot supply an arbitrary different head),
and captures BUILD_STATUS, SOURCE_HEAD, GIT_SOURCE_TREE, terminal accounting and
all final guards. The authorization environment value must equal the actual raw
SHA. Authorization is a separate root decision; file existence cannot grant it.

The SDK guards cover all regular file bytes and a NUL-delimited inventory of all
paths, types, modes and symlink targets before/after. This catches additions,
deletions, chmod and symlink changes that a SHA list alone would miss.

Prepared scratch input:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-replay01/input

The tools profile is explicitly tools-env-v3.sh with qualified GNUtime. Build
manifest path/cwd, actual SOURCE_HEAD, seed path and main build job id must be
provided after the root build. The main job currently announced by root is19789;
this preparation does not assert its terminal state.

## Commands after authorization

Use the repository script, not a relocated standalone copy, for bind:

    SOUNIO_ROOT5_REPLAY_AUTHORIZATION=ACTUAL_RAW_SHA \
      bash scripts/compiler/root5_regression_replay.sh bind \
      D ROOT5/RAW_READY HISTORICAL_SDK ROOT5/source \
      BUILD_MANIFEST BUILD_MANIFEST_CWD SEED TOOLS_ROOT ACTUAL_SOURCE_HEAD MAIN_JOB

bind performs metadata checks/copies only and does not submit or run a test.

Root may then submit the prepared array, with concurrency at most4:

    sbatch --array=0-11%2 --output=D/slurm-%A_%a.out \
      --error=D/slurm-%A_%a.err D/harness/launch.array.sbatch D

After all array tasks are terminal, collect in a Slurm allocation:

    bash D/harness/root5_regression_replay.sh collect D

The C++23 census checks all96 command/SDK/mode/source/output bindings and retains
exact byte offsets/excerpts for recognized error diagnostics. A zero compiler
exit with error bytes is not counted as a clean compiler outcome. It preserves
outer exit codes and reports negative signals only when GNUtime explicitly
reports them; no old PC or ABI tracer is used.

Collector status is structural/provenance status. Historical and current compiler
statuses are separate files. Successful collection does not imply all compiler
phases succeeded and never implies numerical validation. A historical library
refusal after a crash disappears (for example Graph typed len) remains visible
alongside the separate current-SDK result.

Static review is pending root acceptance. C++ compilation and protocol
qualification must occur only after authorization; no execution is implied here.
