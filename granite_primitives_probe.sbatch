#!/usr/bin/env bash
#SBATCH --job-name=granite-prims-probe
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-prims-probe-%j.log
set -uo pipefail
w=/tmp/granite-prims-probe-$SLURM_JOB_ID
mkdir -p "$w"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
compiler=/tmp/granite-tail-29222/madaros
export SOUNIO_STDLIB_PATH=/tmp/granite-tail-29222/src/stdlib
cd "$w"
cat > probe.sio <<'SIO'
fn main() -> i64 with IO, Alloc, Mut, Panic {
    let a = heap_alloc(32) as i64
    let p = a as *mut i64
    (*p) = 1234605616436508552
    let b = (a + 3) as *mut u8
    let got = (*b) as i64
    if got != 85 { print_int(got); return 1 }
    (*b) = 42 as u8
    if (*p) != 1234605615715088264 { print_int(*p); return 2 }
    heap_free(a as *mut u8)
    0
}
SIO
"$compiler" check probe.sio > check.log 2>&1
rec "typed-byte-check-rc-$?"
"$compiler" build probe.sio -o probe > build.log 2>&1
rec "typed-byte-build-rc-$?"
if [[ -f probe ]]; then chmod +x probe; ./probe > run.log 2>&1; rec "typed-byte-run-rc-$?"; cat run.log | while IFS= read -r l; do rec "$l"; done; fi
grep -E 'error\[|= name|failed' check.log build.log | head -10 | while IFS= read -r l; do rec "$l"; done
exit 0
__CA_BUNDLE__
