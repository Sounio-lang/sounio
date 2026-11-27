#!/usr/bin/env bash
#SBATCH --job-name=granite-binary-repro
#SBATCH --partition=all
#SBATCH --exclude=dl380p-dl380p-proxmox,gpuorangefs-5860-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-binary-repro-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
art=$stage/binary-repro-$SLURM_JOB_ID
work=/tmp/granite-binary-repro-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$art/" 2>/dev/null || true; rec "binary-repro-exit-$rc"' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt" HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
mkdir -p "$HOME"
ulimit -s unlimited
cd "$work"
git init -q source-A
git -C source-A fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C source-A checkout -q --detach FETCH_HEAD
[[ $(git -C source-A rev-parse HEAD) == "$1" ]]
# A and B differ only in absolute checkout path. Same compiler for A1/A2/B1.
git clone -q --shared source-A source-B
[[ $(git -C source-B rev-parse HEAD) == "$1" ]]
(cd source-A && bash scripts/ci/build_modular_madaros.sh "$work/madaros-A" > "$work/build-madaros-A.log" 2>&1)
rec "madaros-A-sha-$(sha256sum madaros-A | cut -d' ' -f1)"
build(){ local src=$1 compiler=$2 label=$3; (cd "$src" && SOUNIO_STDLIB_PATH="$PWD/stdlib" "$compiler" build tests/llm/llm_v2_61440_audit.sio -o "$work/$label" > "$work/build-$label.log" 2>&1); rec "$label-sha-$(sha256sum "$work/$label" | cut -d' ' -f1)"; }
build source-A "$work/madaros-A" A1
build source-A "$work/madaros-A" A2
build source-B "$work/madaros-A" B1
# Independently derive the compiler from the same source in the other checkout.
(cd source-B && bash scripts/ci/build_modular_madaros.sh "$work/madaros-B" > "$work/build-madaros-B.log" 2>&1)
rec "madaros-B-sha-$(sha256sum madaros-B | cut -d' ' -f1)"
build source-B "$work/madaros-B" B2
cmp_report(){
  local x=$1 y=$2
  if cmp -s "$x" "$y"; then rec "IDENTICAL-$x-$y"; else
    rec "DIFFERENT-$x-$y"
    { cmp -l "$x" "$y" | head -20; } > "$work/diff-$x-$y.log" 2>&1 || true
    head -5 "$work/diff-$x-$y.log" | while IFS= read -r l; do rec "diff-$x-$y $l"; done
    if command -v readelf >/dev/null; then
      readelf -W -S "$x" > "$work/sections-$x.log"
      readelf -W -S "$y" > "$work/sections-$y.log"
    fi
    if command -v strings >/dev/null; then
      strings -a "$x" | grep -F "$work" > "$work/paths-$x.log" || true
      strings -a "$y" | grep -F "$work" > "$work/paths-$y.log" || true
      [[ ! -s "$work/paths-$x.log" ]] || rec "embedded-checkout-path-$x"
      [[ ! -s "$work/paths-$y.log" ]] || rec "embedded-checkout-path-$y"
    fi
  fi
}
cmp_report A1 A2
cmp_report A1 B1
cmp_report B1 B2
cmp_report madaros-A madaros-B
# Preserve full executables and build logs, not only their hash prefixes.
cp A1 A2 B1 B2 madaros-A madaros-B "$art/"
(cd "$art" && sha256sum A1 A2 B1 B2 madaros-A madaros-B > SHA256SUMS)
for j in 29553 29560 29561; do
  p="$stage/provenance-$j"
  if [[ -f "$p/audit" ]]; then
    rec "old-$j-vs-A1-$(if cmp -s "$p/audit" A1; then echo identical; else echo different; fi)"
  fi
  if [[ -f "$p/build-audit.log" ]]; then
    sed -E 's@/tmp/llm-granite-parity-[0-9]+@WORK@g; s@/tmp/granite-binary-repro-[0-9]+@WORK@g' "$p/build-audit.log" > "$work/old-$j-build-normalized.log"
    sed -E 's@/tmp/granite-binary-repro-[0-9]+@WORK@g' "$work/build-A1.log" > "$work/A1-build-normalized.log"
    diff -u "$work/old-$j-build-normalized.log" "$work/A1-build-normalized.log" > "$work/old-$j-build-diff.log" || true
  fi
done
rec GRANITE_BINARY_REPRO_CAPTURED
exit 0
__CA_BUNDLE__
