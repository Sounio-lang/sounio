#!/usr/bin/env bash
#SBATCH --job-name=madaros-exit-group
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=none
#SBATCH --time=00:45:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/madaros-exit-group-%j.log
set -euo pipefail
commit=${1:?source commit}
mode=${2:?before or after}
work=/tmp/madaros-exit-group-$SLURM_JOB_ID
stage=/orangefs/training/llm-granite-n10
art="$stage/exit-group-$SLURM_JOB_ID"
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$art/" 2>/dev/null || true; if [[ -d "$work/probe-out" ]]; then cp "$work/probe-out/"*.log "$work/probe-out/SHA256SUMS" "$art/" 2>/dev/null || true; fi; rec "exit-group-job-rc-$rc"; if ((rc)); then for f in "$work/build-madaros.log" "$work/gate.log" "$work/probe-out/check.log" "$work/probe-out/build.log"; do if [[ -s "$f" ]]; then rec "failure-$(basename "$f")"; tail -5 "$f" | while IFS= read -r l; do rec "$l"; done; fi; done; fi' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt" ZIG_GLOBAL_CACHE_DIR="$work/cache" ZIG_LOCAL_CACHE_DIR="$work/local-cache"
export HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
mkdir -p "$HOME"
ulimit -s unlimited
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$commit"
git -C src checkout -q --detach FETCH_HEAD
[[ $(git -C src rev-parse HEAD) == "$commit" ]]
rec "source-$commit"
cd src
bash scripts/ci/build_modular_madaros.sh "$work/madaros" > "$work/build-madaros.log" 2>&1
rec "madaros-sha-$(sha256sum "$work/madaros" | cut -d' ' -f1)"
cat > "$work/cxx" <<WRAP
#!/usr/bin/env bash
exec "$stage/tooling/zig-x86_64-linux-0.15.2/zig" c++ "\$@"
WRAP
chmod +x "$work/cxx"
rc=0
CXX="$work/cxx" bash scripts/ci/madaros_exit_group_thread_gate.sh "$work/madaros" "$work/probe-out" > "$work/gate.log" 2>&1 || rc=$?
cat "$work/gate.log"
while IFS= read -r l; do rec "$l"; done < "$work/gate.log"
[[ -s "$work/probe-out/SHA256SUMS" ]] || exit 1
rec "probe-sha-$(sha256sum "$work/probe-out/probe" | cut -d' ' -f1)"
rec "dso-sha-$(sha256sum "$work/probe-out/libexit_group_probe.so" | cut -d' ' -f1)"
if [[ "$mode" == before ]]; then
  [[ "$rc" -ne 0 ]]
  grep -qx THREAD_READY_RETURN_37 "$work/gate.log"
  grep -qx 'EXIT_GROUP_PROBE_RC=124 EXPECTED=37' "$work/gate.log"
  rec EXIT_GROUP_BEFORE_TIMEOUT_REPRODUCED
else
  [[ "$mode" == after && "$rc" == 0 ]]
  grep -qx EXIT_GROUP_THREAD_RETURN_PASS "$work/gate.log"
  rec EXIT_GROUP_AFTER_PASS
fi
exit 0
__CA_BUNDLE__
