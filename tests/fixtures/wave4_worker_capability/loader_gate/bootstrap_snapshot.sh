#!/bin/bash
# Proposed build phase. Execute only after root approves a sealed snapshot.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Slurm required' >&2;exit 2; }
[[ $# == 4 ]] || { echo 'usage: bootstrap_snapshot.sh SEALED_SOURCE SOURCE_MANIFEST LEGACY_SEED FRESH_OUTPUT' >&2;exit 2; }
SOURCE=$(realpath -e "$1")
MANIFEST=$(realpath -e "$2")
SEED=$(realpath -e "$3")
OUT=$(realpath -m "$4")
[[ -d "$SOURCE/self-hosted" && -s "$MANIFEST" && -x "$SEED" && ! -e "$OUT" ]] ||exit 2
[[ $(od -An -tx1 -N4 "$SEED"|tr -d ' \n') == 7f454c46 ]] ||exit 2
# The snapshot owner, not this recipe, injects opt-in and capacity overlays.
grep -q 'worker_lower_backend_opt_in()' "$SOURCE/self-hosted/compiler/main.sio" || { echo SNAPSHOT_OPT_IN_MISSING >&2;exit 2; }
mkdir -p "$OUT"
trap 'r=$?;echo "$r" > "$OUT/BUILD_STATUS";exit "$r"' EXIT
ulimit -s 524288
ulimit -c 0
export PATH=/usr/bin:/bin
export SOUC_BIN="$SEED" SOUNIO_STDLIB_PATH="$SOURCE/self-hosted"
export SOUNIO_BUILD_LOCK="$OUT/build.lock" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
sha256sum "$SEED" "$MANIFEST" > "$OUT/INPUTS.sha256"
cd "$SOURCE"
sha256sum -c "$MANIFEST" > "$OUT/SOURCE.before.verify"
bash scripts/ci/build_modular_madaros.sh "$OUT/madaros" > "$OUT/build.stdout" 2> "$OUT/build.stderr"
[[ -s "$OUT/madaros" && $(od -An -tx1 -N4 "$OUT/madaros"|tr -d ' \n') == 7f454c46 ]]
! grep -Eq '^error:|^error\[|Error:|field initializer mismatch|would have dropped|cannot compile an assignment' "$OUT/build.stdout" "$OUT/build.stderr"
sha256sum -c "$MANIFEST" > "$OUT/SOURCE.after.verify"
sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.after.verify"
chmod +x "$OUT/madaros"
sha256sum "$OUT/madaros" > "$OUT/ELF.sha256"
cut -d ' ' -f1 "$OUT/ELF.sha256" > "$OUT/RAW_READY.tmp"
mv "$OUT/RAW_READY.tmp" "$OUT/RAW_READY"
echo WORKER_CANDIDATE_READY_BFS_AND_RUNTIME_GATES_PENDING
