#!/usr/bin/env bash
# Build Madaros (and optionally run gates) on an IDLE SLURM COMPUTE NODE instead
# of on the workspace pod.
#
# Why this exists
# ---------------
# The workspace pod has 12 CPUs. The cluster has ~204 idle ones
# (cpuops-t560: 64, gpuorangefs-r770: 128, gpuorangefs-5860: 12). Every Madaros
# build ran on the pod, serialized behind a global lock, because the k8s
# liveness probe recycles the pod under CPU saturation -- on 2026-05-29 a
# concurrent-build stampede pushed 15-min load to ~153 and evicted it twice
# (CLAUDE.md section 4). The lock protects the pod from itself. It is the right
# answer to the wrong problem: the build should not be on the pod.
#
# Compute nodes do NOT mount /workspace, cannot reach github, and have no
# gcc/make. None of that matters: the Madaros build is self-contained -- a
# Sounio ELF compiling .sio sources -- and the payload it needs is 7.9 MB.
# So the repo subset is shipped through srun's stdin as a tarball.
#
# Measured 2026-07-26 on cpuops-t560-proxmox with 16 CPUs:
#   payload 7.9 MB -> unpacked 70 MB -> build rc=0 in 220s -> smoke test OK
# against roughly 10 minutes on the pod, while using zero pod CPU.
#
# Usage
#   scripts/dev/souc-build-remote.sh                       # build only
#   scripts/dev/souc-build-remote.sh --gate full           # + madaros_full_gate.sh
#   scripts/dev/souc-build-remote.sh --gate corpus         # + corpus regression gate
#   scripts/dev/souc-build-remote.sh --gate check          # + gen1 typechecks main.sio
#   scripts/dev/souc-build-remote.sh --gate check-stack-matrix # diagnostic stack sweep
#   scripts/dev/souc-build-remote.sh --gate hello          # + compile and run hello with gen1
#   scripts/dev/souc-build-remote.sh --gate gen2-progress  # + self-build into_acc_done floor
#   scripts/dev/souc-build-remote.sh --gate gen2-measure   # + direct single-process self-build metric
#   scripts/dev/souc-build-remote.sh --gate stack-floor    # + <=32 MiB compiler stack gate
#   SOUNIO_WITNESS_GLOB='tests/compiler/foo/*.sio' \
#     scripts/dev/souc-build-remote.sh --gate witness      # + task witness gate
#   scripts/dev/souc-build-remote.sh --gate full --gate corpus
#   SOUNIO_REMOTE_PARTITION=cpu-ops SOUNIO_REMOTE_CPUS=32 ...
#
# The built ELF stays on the node. Only text comes back. That is deliberate:
# shipping a 101 MB ELF over srun's stdout is slower than rebuilding, and what
# you almost always want is the gate verdict, not the binary. If you need the
# binary locally, build locally.
#
# Falls back to a local build when SLURM is unavailable, so callers do not have
# to branch.

set -uo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT_DIR="${SOUNIO_REMOTE_SOURCE_ROOT:-$SCRIPT_ROOT}"
cd "$ROOT_DIR"

# The workspace pod carries a 16 GiB soft RLIMIT_AS even though its hard limit
# is unlimited. Slurm propagates that soft limit to compute jobs, where the
# source-generated seed then SIGSEGVs while compiling main.sio. Lift it before
# srun so the remote build can use the memory granted by the selected node.
ulimit -v unlimited 2>/dev/null || true

# Target policy comes from the cluster-ops tier document
# (~devsounio/beagle/k8s/hpc-sota/ops/cluster-cpu-tiers.md, 2026-06-27):
#
#   r770  128 cores  128G  HPC-first     Slurm cap 120   <- batch belongs here
#   r740   80 cores  515G  Balanced      cap 72          (NOT in sinfo today)
#   t560   64 cores  192G  Service-first cap 32 FIXED    <- control-plane, etcd+apiserver
#   5860   12 cores  128G  GPU-probe     cap 12
#
# The document is explicit: "NAO usar cpu-ops (t560) p/ jobs grandes -- e
# control-plane, cap 32 EXCLUSIVE." Defaulting here to r770, which the same
# document rates HPC-first at 0% CPU utilisation.
PARTITION="${SOUNIO_REMOTE_PARTITION:-all}"
NODE="${SOUNIO_REMOTE_NODE:-gpuorangefs-r770-proxmox}"
CPUS="${SOUNIO_REMOTE_CPUS:-32}"
TIMELIMIT="${SOUNIO_REMOTE_TIME:-00:40:00}"
GATES=""
GEN2_MIN="${SOUNIO_REMOTE_GEN2_MIN_INTO_ACC_DONE:-40}"
FIXED_SEED_SHA256="${SOUNIO_REMOTE_FIXED_SEED_SHA256:-$(sha256sum bin/souc-lean-single-x86_64 | awk '{print $1}')}"
MADAROS_SEED_PATH="${SOUNIO_REMOTE_MADAROS_SEED_PATH:-}"
MADAROS_SEED_SHA256="${SOUNIO_REMOTE_MADAROS_SEED_SHA256:-}"
EXPORT_MADAROS_SEED_PATH="${SOUNIO_REMOTE_EXPORT_MADAROS_SEED_PATH:-}"
CLEAN_ENV="${SOUNIO_REMOTE_CLEAN_ENV:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --gate) GATES="$GATES $2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v srun >/dev/null 2>&1; then
  echo "[remote-build] srun not available -- falling back to a local build" >&2
  exec bash scripts/dev/souc-build-lock.sh \
    bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
fi

if ! sinfo -p "$PARTITION" -h -o "%t" 2>/dev/null | grep -q "idle\|mix"; then
  echo "[remote-build] no idle node in partition $PARTITION -- falling back to a local build" >&2
  exec bash scripts/dev/souc-build-lock.sh \
    bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
fi

case "$NODE" in
  *t560*)
    echo "[remote-build] REFUSING to target $NODE." >&2
    echo "[remote-build] t560 is the single control-plane node (etcd + apiserver)." >&2
    echo "[remote-build] cluster-cpu-tiers.md: do not send large jobs there." >&2
    echo "[remote-build] Override with SOUNIO_REMOTE_ALLOW_T560=1 if you mean it." >&2
    [ "${SOUNIO_REMOTE_ALLOW_T560:-0}" = "1" ] || exit 2
    ;;
esac

echo "[remote-build] partition=$PARTITION node=$NODE cpus=$CPUS gates=${GATES:-none}"

REMOTE_SCRIPT=$(cat <<REMOTE
set -uo pipefail
W=/tmp/sounio-remote-\$\$
mkdir -p "\$W" && cd "\$W" || exit 1
tar xzf - 2>/dev/null || { echo "REMOTE: untar failed"; exit 1; }
echo "REMOTE: host=\$(hostname) nproc=\$(nproc) unpacked=\$(du -sh . | cut -f1)"
export SOUNIO_STDLIB_PATH="\$W/stdlib"
# A private lock path: the pod's global build lock is meaningless here, and
# reusing it would serialize independent nodes against each other.
export SOUNIO_BUILD_LOCK=/tmp/remote-build-\$\$.lock
payload_bootstrap="\$W/bin/souc-linux-x86_64"
payload_lean_src="\$W/self-hosted/compiler/lean_single.sio"
fixed_seed="\$W/bin/souc-lean-single-x86_64"
bootstrap_sha=\$(sha256sum "\$payload_bootstrap" | awk '{print \$1}')
lean_source_sha=\$(sha256sum "\$payload_lean_src" | awk '{print \$1}')
seed_sha=\$(sha256sum "\$fixed_seed" | awk '{print \$1}')
echo "REMOTE: bootstrap sha256=\$bootstrap_sha"
echo "REMOTE: lean_source sha256=\$lean_source_sha"
echo "REMOTE: seed sha256=\$seed_sha"
t0=\$SECONDS
if [ -n "$MADAROS_SEED_PATH" ]; then
  cp "$MADAROS_SEED_PATH" "\$W/madaros.elf"
  rc=\$?
  echo "REMOTE: madaros source=fixed_seed path=$MADAROS_SEED_PATH copy_rc=\$rc"
  if [ \$rc -ne 0 ] || [ ! -s "\$W/madaros.elf" ]; then rm -rf "\$W"; exit 1; fi
  actual_madaros_seed_sha=\$(sha256sum "\$W/madaros.elf" | awk '{print \$1}')
  if [ -n "$MADAROS_SEED_SHA256" ] && [ "\$actual_madaros_seed_sha" != "$MADAROS_SEED_SHA256" ]; then
    echo "REMOTE: Madaros seed hash mismatch expected=$MADAROS_SEED_SHA256 actual=\$actual_madaros_seed_sha"
    rm -rf "\$W"
    exit 1
  fi
else
  if [ "\$seed_sha" != "$FIXED_SEED_SHA256" ]; then
    echo "REMOTE: fixed seed hash mismatch expected=$FIXED_SEED_SHA256 actual=\$seed_sha"
    rm -rf "\$W"
    exit 1
  fi
  chmod +x "\$fixed_seed"
  unset SOUNIO_SOUC_BIN
  SOUC_BIN="\$fixed_seed" bash scripts/ci/build_modular_madaros.sh "\$W/madaros.elf" \
    > "\$W/build.log" 2>&1
  rc=\$?
  echo "REMOTE: madaros source=current_snapshot build_rc=\$rc elapsed=\$((SECONDS-t0))s"
  echo "REMOTE: build rc=\$rc elapsed=\$((SECONDS-t0))s"
  if [ \$rc -ne 0 ]; then tail -20 "\$W/build.log"; rm -rf "\$W"; exit \$rc; fi
fi
if [ -n "$EXPORT_MADAROS_SEED_PATH" ]; then
  mkdir -p "\$(dirname "$EXPORT_MADAROS_SEED_PATH")"
  cp "\$W/madaros.elf" "$EXPORT_MADAROS_SEED_PATH"
  export_rc=\$?
  echo "REMOTE: madaros seed export path=$EXPORT_MADAROS_SEED_PATH rc=\$export_rc"
  if [ \$export_rc -ne 0 ]; then rm -rf "\$W"; exit \$export_rc; fi
fi
ls -la "\$W/madaros.elf" | awk '{print "REMOTE: elf bytes="\$5}'
sha256sum "\$W/madaros.elf" | awk '{print "REMOTE: elf sha256="\$1}'
for g in $GATES; do
  case "\$g" in
    full)
      echo "REMOTE: --- madaros_full_gate ---"
      MADAROS_RAW_BIN="\$W/madaros.elf" bash scripts/ci/madaros_full_gate.sh 2>&1 | tail -20
      echo "REMOTE: full_gate rc=\$?"
      ;;
    corpus)
      echo "REMOTE: --- corpus regression gate ---"
      SOUNIO_MADAROS_CORPUS_BIN="\$W/madaros.elf" SOUNIO_TEST_JOBS=\$(nproc) \\
        bash scripts/ci/madaros_corpus_regression_gate.sh 2>&1 | tail -25
      echo "REMOTE: corpus_gate rc=\$?"
      ;;
    check)
      echo "REMOTE: --- gen1 check self-hosted/compiler/main.sio ---"
      ulimit -s 524288 2>/dev/null || true
      "\$W/madaros.elf" check self-hosted/compiler/main.sio > "\$W/fpcheck.log" 2>&1
      fp_rc=\$?
      fp_err=\$(grep -cE 'error\\[E[0-9]+\\]' "\$W/fpcheck.log" || true)
      echo "REMOTE: fpcheck rc=\$fp_rc errors=\$fp_err"
      grep 'specializer: generic_fns=' "\$W/fpcheck.log" | tail -1 | sed 's/^/REMOTE: /'
      grep -oE 'error\\[E[0-9]+\\]' "\$W/fpcheck.log" | sort | uniq -c | sort -rn | head -8 | sed 's/^/REMOTE: /'
      if [ "\${fp_err:-0}" -gt 0 ]; then
        grep '^error\\[' "\$W/fpcheck.log" | head -20 | sed 's/^/REMOTE: /'
      fi
      tail -3 "\$W/fpcheck.log" | sed 's/^/REMOTE: /'
      if [ \$fp_rc -ne 0 ] || [ "\${fp_err:-0}" -gt 0 ]; then exit 1; fi
      ;;
    check-stack-matrix)
      echo "REMOTE: --- diagnostic check stack matrix ---"
      for stack_kib in 524288 1048576 2097152 unlimited; do
        (
          ulimit -s "\$stack_kib" 2>/dev/null || true
          "\$W/madaros.elf" check self-hosted/compiler/main.sio
        ) > "\$W/check-\$stack_kib.log" 2>&1
        matrix_rc=\$?
        matrix_err=\$(grep -cE 'error\\[E[0-9]+\\]' "\$W/check-\$stack_kib.log" || true)
        echo "REMOTE: check_stack stack_kib=\$stack_kib rc=\$matrix_rc errors=\$matrix_err"
        if [ \$matrix_rc -ne 0 ] || [ "\${matrix_err:-0}" -gt 0 ]; then
          tail -3 "\$W/check-\$stack_kib.log" | sed 's/^/REMOTE: /'
        fi
      done
      ;;
    hello)
      echo "REMOTE: --- gen1 compile/run examples/hello.sio ---"
      # Gate cases share this shell; lowering its stack here poisoned a later
      # gen2-progress check even though hello itself passed.
      "\$W/madaros.elf" compile examples/hello.sio -o "\$W/hello.elf" \
        > "\$W/hello.compile.log" 2>&1
      hello_compile_rc=\$?
      if [ \$hello_compile_rc -ne 0 ]; then
        tail -20 "\$W/hello.compile.log" | sed 's/^/REMOTE: /'
        echo "REMOTE: hello compile_rc=\$hello_compile_rc"
        exit \$hello_compile_rc
      fi
      chmod +x "\$W/hello.elf"
      hello_output=\$("\$W/hello.elf")
      hello_run_rc=\$?
      echo "REMOTE: hello compile_rc=0 run_rc=\$hello_run_rc output=\$hello_output"
      if [ \$hello_run_rc -ne 0 ] || [ "\$hello_output" != "Hello, Sounio" ]; then exit 1; fi
      ;;
    gen2-progress)
      echo "REMOTE: --- Madaros self-build progress ---"
      ulimit -s 524288 2>/dev/null || true
      MADAROS_BIN="\$W/madaros.elf" \
        SOUNIO_MADAROS_FP_DIR="\$W/fixed-point" \
        SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE="$GEN2_MIN" \
        bash scripts/ci/madaros_fixed_point_gate.sh
      gen2_rc=\$?
      echo "REMOTE: gen2_progress rc=\$gen2_rc minimum=$GEN2_MIN"
      if [ \$gen2_rc -ne 0 ]; then exit \$gen2_rc; fi
      ;;
    gen2-measure)
      echo "REMOTE: --- direct Madaros self-build measurement ---"
      ulimit -s 524288 2>/dev/null || true
      "\$W/madaros.elf" compile self-hosted/compiler/main.sio -o "\$W/gen2.elf" \
        > "\$W/gen2-measure.log" 2>&1
      gen2_measure_rc=\$?
      gen2_measure_into=\$(grep -oE 'into_acc_done[[:space:]]+[0-9]+' "\$W/gen2-measure.log" | grep -oE '[0-9]+' | tail -1)
      gen2_measure_into=\${gen2_measure_into:-0}
      gen2_measure_first=\$(grep -m1 -E 'println-poison|IR lowering failed|ir_[a-z_]+_failed|error\\[E[0-9]+\\]|Error: native code buffer overflow|Failed to write native binary|multimodule native thin-link compilation failed' "\$W/gen2-measure.log" || true)
      echo "REMOTE: gen2_measure rc=\$gen2_measure_rc into_acc_done=\$gen2_measure_into minimum=$GEN2_MIN"
      if [ -n "\$gen2_measure_first" ]; then
        echo "REMOTE: gen2_measure first_failure=\$gen2_measure_first"
      fi
      grep -E 'first flagged in preseed stage|unresolved identifiers|lowering-error record|lowering errors:|raised at lower\\.sio lines:|cause:' "\$W/gen2-measure.log" \
        | head -8 | sed 's/^/REMOTE: gen2_measure context=/'
      tail -3 "\$W/gen2-measure.log" | sed 's/^/REMOTE: gen2_measure tail=/'
      if [ "\$gen2_measure_into" -lt "$GEN2_MIN" ]; then exit 1; fi
      ;;
    stack-floor)
      echo "REMOTE: --- Madaros stack-floor gate ---"
      MADAROS_RAW_BIN="\$W/madaros.elf" \
        bash scripts/dev/measure_madaros_stack_floor.sh --reps 10 \
        > "\$W/stack-floor.csv" 2>&1
      stack_rc=\$?
      sed 's/^/REMOTE: /' "\$W/stack-floor.csv"
      if [ \$stack_rc -ne 0 ]; then
        echo "REMOTE: stack_floor rc=\$stack_rc"
        exit \$stack_rc
      fi
      for expected in 'minimal,32,0,10' 'helper,32,0,10'; do
        if ! grep -Fxq "\$expected" "\$W/stack-floor.csv"; then
          echo "REMOTE: stack_floor FAIL missing=\$expected"
          exit 1
        fi
      done
      echo "REMOTE: stack_floor PASS stack_mib=32 reps=10/10 programs=2/2"
      ;;
    silent)
      echo "REMOTE: --- silent verdict measurement ---"
      export SOUNIO_STDLIB_PATH="\$W/stdlib"
      ulimit -s 524288 2>/dev/null || true
      SOUNIO_SILENT_VERDICT_MADAROS="\$W/madaros.elf" \\
        bash scripts/dev/measure_silent_verdicts.sh 2>&1 | tail -40
      echo "REMOTE: silent rc=\$?"
      ;;
    sabotage)
      echo "REMOTE: --- witness declares its sabotage ---"
      export SOUNIO_STDLIB_PATH="\$W/stdlib"
      ulimit -s 524288 2>/dev/null || true
      SOUNIO_WITNESS_SABOTAGE_MADAROS="\$W/madaros.elf" \\
        bash scripts/ci/witness_declares_its_sabotage_gate.sh 2>&1 | tail -30
      echo "REMOTE: sabotage_gate rc=\$?"
      ;;
    witness)
      echo "REMOTE: --- witness ---"
      export SOUNIO_STDLIB_PATH="\$W/stdlib"
      ulimit -s 524288 2>/dev/null || true
      WITNESS_GLOB="${SOUNIO_WITNESS_GLOB:-tests/run-pass/r1_i*_lorenz_peak.sio}"
      if [ "\$WITNESS_GLOB" = 'tests/compiler/epistemic_payload_gate/*.sio' ]; then
        MADAROS_RAW_BIN="\$W/madaros.elf" \
          SOUNIO_WITNESS_GLOB="\$WITNESS_GLOB" \
          bash scripts/ci/madaros_epistemic_payload_gate.sh 2>&1
        wit_rc=\$?
      else
        wit_rc=0
        for src in \$W/\$WITNESS_GLOB; do
          echo "REMOTE: witness src=\$src"
          # Positive control: sabotaged mul MUST fail the fixture.
          out_bad=/tmp/witness-bad-\$\$.elf
          SOUNIO_WIDE_MUL_SABOTAGE=1 "\$W/madaros.elf" build "\$src" "\$out_bad"
          if [ \$? -eq 0 ]; then
            chmod +x "\$out_bad"
            "\$out_bad"
            bad_rc=\$?
            echo "REMOTE: sabotage run rc=\$bad_rc (must be non-zero)"
            if [ \$bad_rc -eq 0 ]; then
              echo "REMOTE: CONTROL_FAIL witness passed under sabotaged mul"
              wit_rc=2
              continue
            fi
            echo "REMOTE: CONTROL_PASS"
          else
            echo "REMOTE: sabotage build failed; treating as control pass (compile refused)"
          fi
          out=/tmp/witness-\$\$.elf
          unset SOUNIO_WIDE_MUL_SABOTAGE
          "\$W/madaros.elf" build "\$src" "\$out"
          b_rc=\$?
          echo "REMOTE: witness build rc=\$b_rc"
          if [ \$b_rc -ne 0 ]; then wit_rc=\$b_rc; continue; fi
          chmod +x "\$out"
          "\$out"
          r_rc=\$?
          echo "REMOTE: witness run rc=\$r_rc"
          if [ \$r_rc -ne 0 ]; then wit_rc=\$r_rc; fi
        done
      fi
      echo "REMOTE: witness_gate rc=\$wit_rc"
      if [ \$wit_rc -ne 0 ]; then exit \$wit_rc; fi
      ;;
    *) echo "REMOTE: unknown gate \$g" ;;
  esac
done
rm -rf "\$W"
REMOTE
)

# tests/ is included only when a gate needs it -- it is the bulk of the payload.
PAYLOAD="self-hosted stdlib bin/souc bin/souc-linux-x86_64 bin/souc-lean-single-x86_64 scripts"
case "$GATES" in *full*|*corpus*|*witness*|*sabotage*|*silent*) PAYLOAD="$PAYLOAD tests bin/madaros bin/madaros-linux-x86_64" ;; esac
case "$GATES" in *hello*) PAYLOAD="$PAYLOAD examples" ;; esac

SRUN_ENV_ARGS=()
if [ "$CLEAN_ENV" = "1" ]; then
  SRUN_ENV_ARGS=(--export=NONE,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
fi

# Keep the measurement harness fixed when ROOT_DIR points at a historical
# source snapshot. The final tar entry replaces that snapshot's gate on unpack.
tar czf - $PAYLOAD -C "$SCRIPT_ROOT" scripts/ci/madaros_fixed_point_gate.sh 2>/dev/null \
  | srun --partition="$PARTITION" ${NODE:+--nodelist="$NODE"} --ntasks=1 \
     "${SRUN_ENV_ARGS[@]}" --job-name="${SOUNIO_REMOTE_JOBNAME:-souc-${GATES:-build}-$$}" \
         --cpus-per-task="$CPUS" --time="$TIMELIMIT" /usr/bin/bash -c "$REMOTE_SCRIPT" 2>&1 \
  | grep -vE "^srun: (job|Job)|couldn't chdir"
