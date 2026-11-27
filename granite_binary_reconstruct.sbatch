#!/usr/bin/env bash
#SBATCH --job-name=granite-binary-reconstruct
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=none
#SBATCH --time=00:20:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-binary-reconstruct-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
repro=$stage/binary-repro-${1:?repro job id}
head_c=ad4a505c1931d48552397089ca11d7cfaeabd988
art=$stage/binary-reconstruct-$SLURM_JOB_ID
work=/tmp/granite-binary-reconstruct-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$art/" 2>/dev/null || true; rec "reconstruct-exit-$rc"' EXIT
(cd "$repro" && sha256sum -c SHA256SUMS)
[[ $(sha256sum "$repro/madaros-A" | cut -d' ' -f1) == 5df8c456cdb5db06bf48485b4ebfdbae518e39be4d50efa798b5d00ebdbd3659 ]]
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
ulimit -s unlimited
cd "$work"
git init -q source
git -C source fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$head_c"
git -C source checkout -q --detach FETCH_HEAD
for j in 29553 29560 29561; do
  # Reconstruct in the historical job-local source path on D. Refuse collision.
  historical=/tmp/llm-granite-parity-$j
  [[ ! -e "$historical" ]] || { rec "historical-path-existing-$j"; exit 1; }
  mkdir -p "$historical/home"
  git clone -q --shared source "$historical/src"
  [[ $(git -C "$historical/src" rev-parse HEAD) == "$head_c" ]]
  (cd "$historical/src" && HOME="$historical/home" XDG_CACHE_HOME="$historical/home/.cache" SOUNIO_STDLIB_PATH="$PWD/stdlib" "$repro/madaros-A" build tests/llm/llm_v2_61440_audit.sio -o "$historical/audit" > "$work/build-$j.log" 2>&1)
  cp "$historical/audit" "$art/audit-$j"
  hash=$(sha256sum "$art/audit-$j" | cut -d' ' -f1)
  rec "reconstructed-$j-$hash"
  case "$j" in
    29553) expected=8c74378a189919e4733847a100a72e976120f4b48b261ccdffcbe25e08bbd04c ;;
    29560) expected=5909057062c7 ;;
    29561) expected=d65d3dbf04cb ;;
  esac
  if [[ "$hash" == "$expected"* ]]; then rec "historical-hash-match-$j"; else rec "historical-hash-mismatch-$j"; fi
  if [[ -f "$stage/provenance-$j/build-audit.log" ]]; then
    diff -u "$stage/provenance-$j/build-audit.log" "$work/build-$j.log" > "$work/old-$j-build.diff.log" || true
  fi
done
for j in 29560 29561; do
  LC_ALL=C sed -b "s@/tmp/llm-granite-parity-29553/src@/tmp/llm-granite-parity-$j/src@g" "$art/audit-29553" > "$work/normalized-$j"
  if cmp -s "$work/normalized-$j" "$art/audit-$j"; then rec "HISTORICAL_PATH_ONLY_DIFF-$j"; else rec "HISTORICAL_REMAINING_DIFF-$j"; cmp -l "$work/normalized-$j" "$art/audit-$j" | head -20 > "$work/diff-$j.log" || true; fi
done
(cd "$art" && sha256sum audit-* > SHA256SUMS)
rec GRANITE_HISTORICAL_RECONSTRUCTION_CAPTURED
exit 0
__CA_BUNDLE__
