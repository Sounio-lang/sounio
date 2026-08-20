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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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
t0=\$SECONDS
bash scripts/ci/build_modular_madaros.sh "\$W/madaros.elf" > "\$W/build.log" 2>&1
rc=\$?
echo "REMOTE: build rc=\$rc elapsed=\$((SECONDS-t0))s"
if [ \$rc -ne 0 ]; then tail -20 "\$W/build.log"; rm -rf "\$W"; exit \$rc; fi
ls -la "\$W/madaros.elf" | awk '{print "REMOTE: elf bytes="\$5}'
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
      grep -oE 'error\\[E[0-9]+\\]' "\$W/fpcheck.log" | sort | uniq -c | sort -rn | head -8 | sed 's/^/REMOTE: /'
      if [ "\${fp_err:-0}" -gt 0 ]; then
        grep '^error\\[' "\$W/fpcheck.log" | head -20 | sed 's/^/REMOTE: /'
      fi
      tail -3 "\$W/fpcheck.log" | sed 's/^/REMOTE: /'
      if [ \$fp_rc -ne 0 ] || [ "\${fp_err:-0}" -gt 0 ]; then exit 1; fi
      ;;
    witness)
      echo "REMOTE: --- witness ---"
      export SOUNIO_STDLIB_PATH="\$W/stdlib"
      ulimit -s 524288 2>/dev/null || true
      WITNESS_GLOB="${SOUNIO_WITNESS_GLOB:-tests/run-pass/r1_i*_lorenz_peak.sio}"
      wit_rc=0
      for src in \$W/\$WITNESS_GLOB; do
        echo "REMOTE: witness src=\$src"
        # The harness annotations decide what this witness IS. Reading them is
        # not optional: without them a compile-fail witness is scored as a
        # broken build, and its refusal -- the entire point -- reads as failure.
        want_fail=0
        grep -qE '^//@[[:space:]]*compile-fail' "\$src" && want_fail=1
        pat=\$(sed -n 's|^//@[[:space:]]*error-pattern:[[:space:]]*||p' "\$src" | head -1)
        # Each witness names the knob that must break it. The default perturbs
        # WIDE multiply, which reaches nothing outside 256/512-bit arithmetic.
        # Applying it to any other witness and calling the pass CONTROL_FAIL
        # blames the witness for the sabotage being inapplicable.
        has_knob=0
        grep -qE '^//@[[:space:]]*sabotage:' "\$src" && has_knob=1
        knob=\$(sed -n 's|^//@[[:space:]]*sabotage:[[:space:]]*||p' "\$src" | head -1)
        [ -z "\$knob" ] && knob=SOUNIO_WIDE_MUL_SABOTAGE
        kind=run-pass; [ \$want_fail -eq 1 ] && kind=compile-fail
        echo "REMOTE: witness kind=\$kind sabotage=\$knob"

        out=/tmp/witness-\$\$.elf
        log=/tmp/witness-\$\$.log
        env -u SOUNIO_WIDE_MUL_SABOTAGE "\$W/madaros.elf" build "\$src" "\$out" > "\$log" 2>&1
        b_rc=\$?
        echo "REMOTE: witness build rc=\$b_rc"
        grep -E 'error\[E[0-9]+\]' "\$log" | head -4 | sed 's/^/REMOTE:   /'

        if [ \$want_fail -eq 1 ]; then
          if [ \$b_rc -eq 0 ]; then
            echo "REMOTE: WITNESS_FAIL compile-fail witness built cleanly"
            wit_rc=1; continue
          fi
          if [ -n "\$pat" ] && ! grep -qF "\$pat" "\$log"; then
            echo "REMOTE: WITNESS_FAIL refused, but not for the declared reason: \$pat"
            wit_rc=1; continue
          fi
          echo "REMOTE: WITNESS_PASS refused with \$pat"
          # A refusal cannot be controlled by "it also failed under sabotage" --
          # it fails either way, so that control is satisfied automatically and
          # proves nothing. Only a knob that makes the refusal DISAPPEAR shows
          # the rule under test is what refused.
          if [ \$has_knob -eq 0 ]; then
            echo "REMOTE: CONTROL_ABSENT no //@ sabotage: knob -- refusal unproven, not proven"
          else
            env "\$knob=1" "\$W/madaros.elf" build "\$src" "\$out.sab" > "\$log.sab" 2>&1
            if [ \$? -eq 0 ]; then echo "REMOTE: CONTROL_PASS refusal disappears under \$knob"
            else echo "REMOTE: CONTROL_INAPPLICABLE \$knob does not reach this refusal"; fi
          fi
          continue
        fi

        if [ \$b_rc -ne 0 ]; then wit_rc=\$b_rc; continue; fi
        chmod +x "\$out"
        "\$out"
        r_rc=\$?
        echo "REMOTE: witness run rc=\$r_rc"
        if [ \$r_rc -ne 0 ]; then wit_rc=\$r_rc; continue; fi

        # If the sabotaged build is byte-identical to the clean one, the knob
        # never reached this code. That is inapplicable, not a vacuous witness,
        # and saying which is the difference between a real alarm and one every
        # future lane learns to ignore.
        out_bad=/tmp/witness-bad-\$\$.elf
        env "\$knob=1" "\$W/madaros.elf" build "\$src" "\$out_bad" > /dev/null 2>&1
        if [ \$? -ne 0 ]; then
          echo "REMOTE: CONTROL_PASS sabotaged build refused"
        elif cmp -s "\$out" "\$out_bad"; then
          # A DECLARED knob that does not bite is a false claim of control, and
          # that is red. The DEFAULT knob not biting is merely no control, which
          # is the same false alarm under a new name if it fails the gate.
          if [ \$has_knob -eq 1 ]; then
            echo "REMOTE: CONTROL_FAIL declared knob \$knob left the ELF byte-identical"
            wit_rc=3
          else
            echo "REMOTE: CONTROL_ABSENT default \$knob does not reach this witness -- declare //@ sabotage:"
          fi
        else
          chmod +x "\$out_bad"; "\$out_bad"; bad_rc=\$?
          echo "REMOTE: sabotage run rc=\$bad_rc (must be non-zero)"
          if [ \$bad_rc -eq 0 ]; then
            echo "REMOTE: CONTROL_FAIL witness passed under sabotaged \$knob"
            wit_rc=2
          else
            echo "REMOTE: CONTROL_PASS"
          fi
        fi
      done
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
PAYLOAD="self-hosted stdlib bin/souc-linux-x86_64 scripts"
case "$GATES" in *full*|*corpus*|*witness*) PAYLOAD="$PAYLOAD tests bin/madaros bin/madaros-linux-x86_64" ;; esac

tar czf - $PAYLOAD 2>/dev/null \
  | srun --partition="$PARTITION" ${NODE:+--nodelist="$NODE"} --ntasks=1 \
         --cpus-per-task="$CPUS" --time="$TIMELIMIT" bash -c "$REMOTE_SCRIPT" 2>&1 \
  | grep -vE "^srun: (job|Job)|couldn't chdir"
