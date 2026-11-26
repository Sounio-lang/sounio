#!/usr/bin/env bash
#SBATCH --job-name=granite-check
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-check-%j.log
set -uo pipefail
work=/tmp/granite-check-$SLURM_JOB_ID
mkdir -p "$work"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
ulimit -s unlimited
cd "$work"
set -e
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
cd src
compiler=/tmp/granite-tail-29222/madaros
if [[ ! -x "$compiler" ]]; then compiler="$work/madaros"; bash scripts/ci/build_modular_madaros.sh "$compiler" > "$work/build.log" 2>&1; fi
rec "compiler-sha-$(sha256sum "$compiler" | cut -d' ' -f1)"
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_MADAROS_BIN="$compiler"
set +e
cat > "$work/missing-prims.sio" <<'SIO'
fn main() -> i64 with Alloc, Mut, IO {
    let p = heap_alloc(16) as i64
    let _w = ptr_write_u8(p, 0, 42)
    let v = ptr_read_u8(p, 0)
    heap_free(p as *mut u8)
    if v == 42 { return 0 }
    1
}
SIO
"$compiler" check "$work/missing-prims.sio" > "$work/missing-prims.log" 2>&1
rec "minimal-raw-ptr-check-rc-$?"
grep -E 'error\[|= name' "$work/missing-prims.log" | head -8 | while IFS= read -r l; do rec "$l"; done
for f in stdlib/llm/qwen3.sio stdlib/llm/qwen3_tokenizer.sio stdlib/llm/qwen3_kernels.sio stdlib/llm/ptx_kernels_sm80.sio tests/llm/llm_v2_adversarial_ftz.sio tests/llm/llm_v2_61440_audit.sio; do
  name=$(basename "$f")
  ./bin/souc check "$f" > "$work/check-$name.log" 2>&1
  rec "check-$name-rc-$?"
  grep -E 'error\[|= name' "$work/check-$name.log" | head -5 | while IFS= read -r l; do rec "$l"; done
done
# Check the unmodified base with exactly the same compiler.
git fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git 575d1eb57
if [[ $? != 0 ]]; then git fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git refs/heads/feat/llm-backend-v2; fi
if git checkout -q --detach 575d1eb57; then
  "$compiler" check tests/llm/llm_v2_61440_audit.sio > "$work/check-base-audit.log" 2>&1
  rec "base-audit-check-rc-$?"
  grep -E 'error\[|= name' "$work/check-base-audit.log" | head -8 | while IFS= read -r l; do rec "$l"; done
fi
for p in /orangefs /orangefs/training /orangefs/*; do
    [[ -d "$p" ]] || continue
    rec "path-$p-mode-$(stat -c '%a-%u-%g' "$p")-writable-$([[ -w "$p" ]] && echo yes || echo no)"
done
exit 0
__CA_BUNDLE__
