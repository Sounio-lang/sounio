<!-- docs:meta
topic_id: repo.docs.spec.2026-10.15-bounded-conformance-receipt
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.15-bounded-conformance-receipt
-->

# 15. Bounded Conformance Execution Receipt

## Result and Claim Boundary

**Executed, not queued-only:** on 2026-10-16, Slurm job `14573` completed
with scheduler exit `0:0`. The existing
`scripts/ci/serious_language_conformance_gate.sh` reported
`total=16 passed=16 failed=0`, and its process exit was `0`.

The manifest, fixtures, standard library, launchers, and compressed compiler
artifact were taken from Git commit
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. The executed compiler was that
commit's **committed Linux x86-64 prebuilt**, materialized and hash-checked,
not a compiler freshly built from that commit's source. This is therefore a
bounded prebuilt-artifact receipt against exact-baseline test inputs, not proof
that the baseline compiler source reproduces this behavior.

This report's metadata deliberately remains `unvalidated`: the execution
does not validate the specification prose, all language contracts, or every
chapter evidence row. Chapter [06](06-execution-memory-and-targets.md) remains
a source-research chapter; its unrun witnesses are not globally promoted by this
receipt. No new language requirement IDs are introduced here.

## Identity, Isolation, and Scheduling

| Field | Observed value |
|---|---|
| Control host / SSH destination | `t560-proxmox`, SSH user `devsounio` |
| Source object repository on control host | `/home/devsounio/sounio` |
| Its checked-out branch / HEAD | `feat/dialogue-tapestry-phase-a` / `9feae917a090c0236bf2d171a722265259fe637d` |
| Selected archive commit, independently verified from Git archive PAX metadata | `f5b2c5850be9f3d3995719ffcade44a6d851658a` |
| Scheduler control surface | `/home/devsounio/beagle/k8s/hpc-sota/ops/lab-ops.sh`, `lab_slurm_exec` |
| Login deployment | `slurm-pilot/deploy/slurm-pilot-login-slinky` |
| Job | `14573`, `spec-bounded-f5b2`, user/group `sounio` / UID:GID `1000:1000` |
| Allocation | Partition `all`; one node, one task, two CPUs, `8G` requested memory, no GPU |
| Walltime / inner gate bound | `00:12:00` / `timeout --signal=TERM --kill-after=10s 600` |
| Actual node and worker hostname | `cpuops-t560-proxmox` |
| Worker kernel / architecture | Linux `7.0.2-5-pve`, `x86_64` |
| Scheduler start / end / elapsed | `2026-10-16T00:38:04` / `2026-10-16T00:39:51` / `00:01:47` (scheduler timestamps; worker identity separately records UTC) |
| Worker identity UTC timestamp | `2026-10-16T00:38:04Z` |
| Harness generated timestamp | `20261016T003810Z` |
| Unique compute-local scratch | `/tmp/spec-bounded-14573.a63sZp` |
| Extracted source directory | `/tmp/spec-bounded-14573.a63sZp/source` |
| Artifact root during execution | `/tmp/spec-bounded-14573.a63sZp/conformance` |

The active development workspace was inspected **read-only**, not used as the
execution surface. It was on `fix/if-call-condition` at
`9b5dba8b3f09a952802cebf68f7ab700f3ab88b4`, with dirty compiler and fixture
files. Neither that branch nor the control host's checked-out branch was
assumed to be main. No branch switch, source edit, compiler rebuild, commit,
push, or test modification was performed.

Preflight read the cluster operational guidance, the actual Foundry CLI/runner,
and repository Slurm scripts. The Foundry profile allowlist did not expose a
bounded-conformance profile or arbitrary custom-command profile. Rather than
repurpose a named profile, execution used the existing documented generic Slurm
route: `lab_slurm_exec` into the login pod, invoking the baseline's
`scripts/dev/slurm_srun_minimal.sh`. This is a Slurm execution receipt, not a
claim that a new Foundry profile exists.

Read-only gates included `lab_status`, `lab_slurm_status`,
`lab_orangefs_status`, controller ping/queue inspection, storage inspection,
and Kubernetes worker readiness/events/node-pressure checks. The actual CPUops
worker `slurm-pilot-worker-cpuops-hnllp` was `2/2 Running` with zero restarts
and no events; its node was Ready with no DiskPressure, MemoryPressure, or
PIDPressure. Worker-local scratch reported `83G` available. Other queued or
running jobs were neither cancelled nor modified. There was no infrastructure
repair, admission change, queue bypass, or local execution fallback.

An attempted environment hint
`SLURM_NODELIST=gpuorangefs-r770-proxmox` did **not** pin the allocation:
`scontrol` showed `ReqNodeList=(null)`, and the scheduler selected CPUops.
The minimal wrapper has no nodelist option. The actual allocation, not the
intended hint, is the identity used in this receipt. Both relevant workers were
checked; no GPU capacity is claimed or consumed.

## Artifact Custody and Hashes

Host-visible run directory:

`/var/lib/cephfs-slurm/training/sounio-foundry/runs/spec-bounded-f5b2-20261016.QA1hSb`

The same shared storage is visible to login/compute at:

`/orangefs/training/sounio-foundry/runs/spec-bounded-f5b2-20261016.QA1hSb`

The legacy mount spelling does not imply OrangeFS was the measured filesystem;
the live shared mount was CephFS. Source was copied to compute-local scratch,
and the terminal receipt bundle was published back to this unique run directory.

| Object | SHA-256 |
|---|---|
| `source.tar.gz`, selected baseline files, not a full checkout | `3e22b0088233efce9caf8f28794f98c4f94439632180a708b51b7ac22f2830a1` |
| `tests/conformance/manifest.v1.tsv` | `ff7199b63fb4b292ac8a94e770fabff8c0ec13d952c4e440aabf03175973b1cf` |
| `scripts/ci/serious_language_conformance_gate.sh` | `24ee7d64e91356479e5989f613e628ebae9ec5e7ac738ec4da74d926596ac992` |
| `bin/madaros-linux-x86_64.gz` | `feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a` |
| `bin/madaros-linux-x86_64.sha256`, checksum document | `d231ac0230c05d4c87429d61fe3429935e5fb294134ed7cc68ca0aa8f98b573b` |
| Executed `bin/madaros-linux-x86_64` | `125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f` |
| Published `receipt.tar.gz` | `4b3159a574541639cb75b462c4a7afda8e8e05e37210dbbc2eccc174e35f8def` |
| Control-plane `ops/lab-ops.sh` | `4c62af9986112e4c53df6e78905fdde5744f6432905ff8e4470db76abfb4561f` |
| Inspected `sounio-forge` | `d8df6768dbace582ea0f08ebe4929bd901932f252cd1fd6aa221bd965b8a7201` |
| Inspected `sounio-foundry-runner.sh` | `3ceda210766437c8b13722c98d91ac08921cb6879d6ec7e6ece95f197cbdb794` |

Compiler stdout identifies `Madaros v0.80.0 -- the Sounio self-hosted compiler`.
The raw executable header was inspected:

```text
7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
02 00 3e 00
```

The snapshot had no `.git` directory and did not contain the compiler source
tree. Consequently the harness JSON correctly records empty `branch` and
`commit` fields. Baseline identity is established by the external archive
commit check, archive hash, and file hashes, not invented harness Git metadata.
The compiler's own provenance output warns `tree=no-git` and explicitly says
the ELF is the committed binary, not built from the displayed tree. Its
`self-hosted/ MODIFIED` wording is not evidence that the archived fixture
inputs were modified; that tree was absent. No built-from-source provenance is
asserted.

The published bundle contains `identity.txt`, `worker-command.txt`,
`source-files.sha256`, `input-identities.txt`, `snapshot-verify.txt`,
`compiler-verify.txt`, `compiler-header.txt`, materialization/version logs,
`gate-result.txt`, `job-result.txt`, gate stdout/stderr, and
`conformance/RESULTS.md`, `summary.v1.tsv`, `summary.v1.json`, and
per-case command/stdout/stderr logs. The shared run directory separately retains
the source archive, source member listing, source hash, preparation metadata,
transport log, and receipt hash. The receipt omits extracted source and scratch
temporaries; the source archive is retained separately.

## Expected and Observed Cases

These are the existing manifest's acceptance conditions, not newly invented
numeric tolerances or a complete semantic oracle. A dash means no substring
condition. Every case has `status=pass` and `reason=ok`. All manifest
`stderr_contains` cells are `-`: this is not an assertion that stderr must
be empty. Negative diagnostics below were observed on stdout.

| Case / mode | Repository-relative source | Expected exit / stdout substring | Observed exit / result |
|---|---|---|---|
| `core-hello-check` / check | `examples/hello.sio` | `0` / — | `0` / pass |
| `core-hello-run` / run | `examples/hello.sio` | `0` / — | `0` / pass |
| `core-struct-run` / run | `examples/native/struct_basic.sio` | `0` / `7` | `0` / pass |
| `effects-superset-run` / run | `tests/run-pass/effect_superset_ok.sio` | `0` / `effect superset: PASS` | `0` / pass |
| `effects-missing-diagnostic` / check-fail | `tests/compile-fail/effect_missing.sio` | nonzero / `effect not declared` | `1` / pass; E035 |
| `observe-io-boundary` / check-fail | `tests/compile-fail/observe_io_boundary.sio` | nonzero / `cannot print Unobserved<T>` | `1` / pass; E036 |
| `modules-import-check` / check | `tests/conformance/import_public.sio` | `0` / — | `0` / pass |
| `generics-struct-run` / run | `tests/run-pass/generic_struct_basic.sio` | `0` / `generic struct parse ok` | `0` / pass |
| `generics-multi-run` / run | `tests/run-pass/generics_multi_param.sio` | `0` / `first_of_two: PASS` | `0` / pass |
| `ownership-release-check` / check | `tests/run-pass/borrow_call_explicit_release.sio` | `0` / — | `0` / pass |
| `ownership-conflict-diagnostic` / check-fail | `tests/compile-fail/borrow_call_conflict_explicit.sio` | nonzero / `cannot borrow exclusively while other borrows are active` | `1` / pass; E038 |
| `gum-compliance-run` / run | `tests/run-pass/gum_compliance.sio` | `0` / `All 7 tests match GUM` | `0` / pass |
| `gum-iso-budget-run` / run | `tests/run-pass/gum_iso_budget.sio` | `0` / `PASS` | `0` / pass |
| `epistemic-bmi-run` / run | `tests/run-pass/epistemic_bmi.sio` | `0` / `BMI uncertainty` | `0` / pass |
| `knowledge-boundary-diagnostic` / check-fail | `tests/compile-fail/knowledge_no_silent_unwrap.sio` | nonzero / `found Knowledge<f64>` | `1` / pass; E001 |
| `epistemic-effect-diagnostic` / check-fail | `tests/compile-fail/epistemic_epsilon_mismatch.sio` | nonzero / `requires with Epistemic` (the manifest includes backticks around `with Epistemic`) | `1` / pass; E170, E001, E009, E036 |

The last fixture emits several diagnostics; the manifest matches only its
required effect substring. This pass must not be promoted into a dedicated
proof of every epsilon rule. Similarly, the GUM fixture's printed claim is its
existing acceptance marker, not an independent recalculation or certification
of all GUM mathematics. The `7` and `PASS` substring conditions are weaker
than exact-output equality. Cases marked `check` do not establish execution.
The gate's `check-fail` mode invokes `bin/souc check`, not native-code
emission; the named diagnostics plus observed exit `1` distinguish these
observations from a generic nonzero crash.

## Commands and Reproduction Boundary

Commands were issued from the Mac through
`ssh -o BatchMode=yes -o ConnectTimeout=8 devsounio@t560-proxmox`.
The following archive selection is the executed baseline file set; it is not a
claim to have cloned the complete source tree. The run root was created uniquely
using `mktemp -d` beneath the shared Foundry runs directory.

```bash
baseline=f5b2c5850be9f3d3995719ffcade44a6d851658a
source_repo=/home/devsounio/sounio
run_root=/var/lib/cephfs-slurm/training/sounio-foundry/runs/spec-bounded-f5b2-20261016.QA1hSb
git -C "$source_repo" archive --format=tar "$baseline" -- \
  AGENTS.md FOUNDER_INTENT.md CLAUDE.md science-rings.tsv schemas tools/science_boundary \
  bin/souc bin/madaros bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 \
  scripts/lib scripts/dev/slurm_srun_minimal.sh scripts/ci/serious_language_conformance_gate.sh \
  stdlib tests/conformance examples/hello.sio examples/native/struct_basic.sio \
  tests/run-pass/effect_superset_ok.sio tests/compile-fail/effect_missing.sio \
  tests/compile-fail/observe_io_boundary.sio tests/run-pass/generic_struct_basic.sio \
  tests/run-pass/generics_multi_param.sio tests/run-pass/borrow_call_explicit_release.sio \
  tests/compile-fail/borrow_call_conflict_explicit.sio tests/run-pass/gum_compliance.sio \
  tests/run-pass/gum_iso_budget.sio tests/run-pass/epistemic_bmi.sio \
  tests/compile-fail/knowledge_no_silent_unwrap.sio tests/compile-fail/epistemic_epsilon_mismatch.sio \
  | gzip -n > "$run_root/source.tar.gz"
archive_commit=$(gzip -dc "$run_root/source.tar.gz" | { git get-tar-commit-id; cat >/dev/null; })
test "$archive_commit" = "$baseline"
```

The worker command below is the actual execution payload, also retained
verbatim as `worker-command.txt` in the receipt bundle:

```bash
set -euo pipefail
shared=/orangefs/training/sounio-foundry/runs/spec-bounded-f5b2-20261016.QA1hSb
test "$(/usr/bin/id -u)" = 1000
test "$(/usr/bin/id -g)" = 1000
test -n "${SLURM_JOB_ID:-}"
scratch=$(mktemp -d "/tmp/spec-bounded-${SLURM_JOB_ID}.XXXXXX")
mkdir "$scratch/source" "$scratch/tmp"
finish() {
  job_rc=$?
  set +e
  printf 'job_exit=%s\nscratch=%s\njob_id=%s\n' "$job_rc" "$scratch" "$SLURM_JOB_ID" > "$scratch/job-result.txt"
  tar -czf "$scratch/receipt.tar.gz" -C "$scratch" --exclude=receipt.tar.gz --exclude=source --exclude=tmp .
  cp "$scratch/receipt.tar.gz" "$shared/receipt.tar.gz"
  sha256sum "$shared/receipt.tar.gz" > "$shared/receipt.sha256"
  cp "$scratch/job-result.txt" "$shared/job-result.txt"
  printf 'BOUNDED_JOB_EXIT=%s JOB_ID=%s SCRATCH=%s\n' "$job_rc" "$SLURM_JOB_ID" "$scratch"
}
trap finish EXIT
{
  date -u +%FT%TZ
  hostname
  uname -a
  id
  printf 'job_id=%s\njob_nodelist=%s\nscratch=%s\nbaseline=f5b2c5850be9f3d3995719ffcade44a6d851658a\ncompiler_kind=baseline_committed_prebuilt_not_source_rebuilt\n' "$SLURM_JOB_ID" "${SLURM_JOB_NODELIST:-}" "$scratch"
  df -h "$scratch"
} > "$scratch/identity.txt"
printf '%s\n' "$BASH_EXECUTION_STRING" > "$scratch/worker-command.txt"
printf '%s\n' '3e22b0088233efce9caf8f28794f98c4f94439632180a708b51b7ac22f2830a1  source.tar.gz' > "$scratch/expected-snapshot.sha256"
cp "$shared/source.tar.gz" "$scratch/source.tar.gz"
(cd "$scratch" && sha256sum -c expected-snapshot.sha256) > "$scratch/snapshot-verify.txt"
tar -xzf "$scratch/source.tar.gz" -C "$scratch/source"
rm "$scratch/source.tar.gz"
cd "$scratch/source"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$scratch/source-files.sha256"
sha256sum tests/conformance/manifest.v1.tsv scripts/ci/serious_language_conformance_gate.sh bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 > "$scratch/input-identities.txt"
bash scripts/lib/materialize_madaros_prebuilt.sh --verify > "$scratch/materialize.log" 2>&1
sha256sum -c bin/madaros-linux-x86_64.sha256 > "$scratch/compiler-verify.txt"
sha256sum bin/madaros-linux-x86_64 >> "$scratch/input-identities.txt"
od -An -tx1 -N20 bin/madaros-linux-x86_64 > "$scratch/compiler-header.txt"
test "$(sha256sum bin/madaros-linux-x86_64 | cut -d' ' -f1)" = 125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f
export MADAROS_RAW_BIN="$scratch/source/bin/madaros-linux-x86_64"
export SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_REQUIRE_COMMITTED_MADAROS=1
export SOUNIO_STDLIB_PATH="$scratch/source/stdlib"
export SOUNIO_SERIOUS_CONFORMANCE_SOUC_BIN="$scratch/source/bin/souc"
export SOUNIO_SERIOUS_CONFORMANCE_STDLIB_PATH="$scratch/source/stdlib"
export SOUNIO_SERIOUS_CONFORMANCE_MANIFEST="$scratch/source/tests/conformance/manifest.v1.tsv"
export SOUNIO_SERIOUS_CONFORMANCE_ARTIFACT_ROOT="$scratch/conformance"
export TMPDIR="$scratch/tmp"
timeout 30 bin/souc --version > "$scratch/compiler-version.stdout" 2> "$scratch/compiler-version.stderr"
set +e
timeout --signal=TERM --kill-after=10s 600 bash scripts/ci/serious_language_conformance_gate.sh > "$scratch/conformance-gate.stdout" 2> "$scratch/conformance-gate.stderr"
gate_rc=$?
set -e
printf 'gate_exit=%s\n' "$gate_rc" > "$scratch/gate-result.txt"
cat "$scratch/identity.txt" "$scratch/input-identities.txt" "$scratch/compiler-version.stdout" "$scratch/compiler-version.stderr" "$scratch/gate-result.txt" "$scratch/conformance-gate.stdout"
if test -f "$scratch/conformance/summary.v1.tsv"; then cat "$scratch/conformance/summary.v1.tsv"; fi
exit "$gate_rc"
```

Submission used the baseline minimal wrapper extracted from the archive into
the unique shared run directory. For readability, this transcript factors the
shell-quoted worker payload into `worker_command`; the command executed by
Slurm is the verbatim payload above. The original submission embedded it in
the quoted command argument. No additional scheduler profile was installed.

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
# lab_slurm_exec /bin/bash -c was passed the following login-side command.
job_root=/orangefs/training/sounio-foundry/runs/spec-bounded-f5b2-20261016.QA1hSb
tar -xzf "$job_root/source.tar.gz" --strip-components=2 \
  -C "$job_root" scripts/dev/slurm_srun_minimal.sh
# worker_command is exactly the payload printed above.
# shell_quote denotes ordinary POSIX single-quote escaping of that payload.
nohup /usr/sbin/runuser --user sounio -- /usr/bin/env \
  SLURM_CONF=/run/slurm/conf/slurm.conf \
  SLURM_CPUS_PER_TASK=2 SLURM_MEM_PER_NODE=8192 \
  SLURM_NODELIST=gpuorangefs-r770-proxmox SLURM_JOB_NAME=spec-bounded-f5b2 \
  /bin/bash "$job_root/slurm_srun_minimal.sh" \
  --partition=all --time=00:12:00 -- \
  "/bin/bash --noprofile --norc -c $(shell_quote "$worker_command")" \
  > "$job_root/logs/transport.log" 2>&1 < /dev/null &
```

The helper supplies `srun --nodes=1 --ntasks=1 --chdir=/tmp` and an
explicit exported base environment. Login `SLURM_CONF` was set to the live
configless cache `/run/slurm/conf/slurm.conf`; the stale default
`/tmp/slurm-direct.conf` was not used. The job asserted UID/GID and Slurm
job identity before executing. The explicit `MADAROS_RAW_BIN` and
`SOUNIO_REQUIRE_COMMITTED_MADAROS=1` selected the verified snapshot-local
prebuilt through the public wrappers, not an ambient active-workspace binary
or a seed. The public engine remained Madaros; no legacy-engine fallback was
requested.

Postflight used the following read-only commands, with host-visible paths for
archive reads and `lab_slurm_exec` for scheduler queries:

```bash
run_root=/var/lib/cephfs-slurm/training/sounio-foundry/runs/spec-bounded-f5b2-20261016.QA1hSb
cd "$run_root"
sha256sum receipt.tar.gz
gzip -t receipt.tar.gz
tar -tzf receipt.tar.gz | sed -n '1,35p'
tar -xOzf receipt.tar.gz \
  ./identity.txt ./input-identities.txt ./compiler-header.txt \
  ./compiler-verify.txt ./snapshot-verify.txt \
  ./compiler-version.stdout ./compiler-version.stderr ./gate-result.txt \
  ./conformance-gate.stderr ./conformance/summary.v1.tsv
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_slurm_exec sacct -X -j 14573 \
  --format=JobID,JobName,User,State,ExitCode,Start,End,Elapsed,AllocCPUS,ReqMem,NodeList -P
lab_slurm_exec squeue -h -j 14573 -o '%i %j %T %R'
```

Observed scheduler receipt:

```text
JobID|JobName|User|State|ExitCode|Start|End|Elapsed|AllocCPUS|ReqMem|NodeList
14573|spec-bounded-f5b2|sounio|COMPLETED|0:0|2026-10-16T00:38:04|2026-10-16T00:39:51|00:01:47|2|8G|cpuops-t560-proxmox
```

The terminal `squeue` query returned no job row. The JSON, gate stdout,
per-case negative diagnostic logs, and a representative recorded run command
were separately read back from the archive. Both
`snapshot-verify.txt` (`source.tar.gz: OK`) and
`compiler-verify.txt` (`bin/madaros-linux-x86_64: OK`) were read back.

## Incidents, Limitations, and Remaining Checks

- The initial metadata-only archive-commit extraction pipeline did not drain
  gzip output and exited `141` under pipefail. The already-created archive
  was retained; the draining pipeline shown above verified its commit before
  submission. This was an orchestration issue, not a compiler failure.
- Terminal packing emitted `tar: .: file changed as we read it` because the
  excluded output archive was created inside the archived scratch directory.
  The EXIT trap preserved the gate exit independently and did not treat tar
  status as a semantic result. Postflight `gzip -t`, archive listing,
  content readback, and the published archive SHA-256 succeeded. This receipt
  does not claim a warning-free packaging process.
- The first host-side `sha256sum -c receipt.sha256` could not open the stored
  absolute `/orangefs/...` pathname: the host uses `/var/lib/cephfs-slurm/...`.
  Direct hashing of the host-visible archive produced the identical digest
  recorded in the job-visible checksum file. The pathname failure was not
  a hash mismatch.
- Python was used only by the existing repository metadata/summary and
  provenance machinery. No Python/Rust numerical oracle, new numeric
  calculation, or recalculation was introduced. Existing Sounio fixtures
  exercised their own embedded checks; no independent numerical
  verification is claimed.
- No fresh source build, self-hosting fixed point, full corpus, stress run,
  sanitizer campaign, complete ownership/memory-safety proof, optimizer
  equivalence proof, Lean theorem build, or cross-target ABI audit ran.
  Darwin ARM64, Windows, WASM, PTX, Metal, and Pireus were not executed.
- The archive intentionally lacks compiler source and Git checkout metadata.
  A future current-source claim needs an independently identified built
  compiler and the same bounded manifest on that build, preserving the raw
  logs and negative diagnostic checks.
- No execution blocker remains for this bounded prebuilt run. Editorial
  finalization need not wait on another queue or Foundry profile. Registry
  registration and central review remain the parent coordinator's work;
  this receipt does not edit or assert their completion.

The execution receipt was checked against the raw scheduler, archive, and
per-case outputs by the authoring agent. No separate orthogonal LLM review
of this new receipt was invoked; earlier chapter-06 consultations were
editorial assistance, not execution witnesses. All semantic limits above
remain in force regardless of that editorial review.
