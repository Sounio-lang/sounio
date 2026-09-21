#!/usr/bin/env bash
# Slurm forensic: written Knowledge field names vs read names.
# /workspace is invisible on the node. Does not edit self-hosted/ or the dissertation file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PARTITION="${SOUNIO_REMOTE_PARTITION:-cpu-ops}"
CPUS="${SOUNIO_REMOTE_CPUS:-4}"
TIMELIMIT="${SOUNIO_REMOTE_TIME:-00:15:00}"
OUT="${EPSILON_SLURM_OUT:-/tmp/epsilon_inerte_slurm.out}"
export SLURM_CONF="${SLURM_CONF:-/tmp/slurm-direct.conf}"

REMOTE="$ROOT/_epsilon_remote.sh"
cat >"$REMOTE" <<'REMOTE'
#!/usr/bin/env bash
set -uo pipefail
ulimit -s 1048576 || true
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
unset SOUC_BIN SOUNIO_SOUC_BIN || true
ROOT="$(pwd)"
SOUC="$ROOT/bin/souc"
chmod +x "$ROOT/bin/souc" "$ROOT/bin/madaros" \
  "$ROOT/bin/madaros-linux-x86_64" "$ROOT/bin/souc-lean-single-x86_64" 2>/dev/null || true
echo "HOST=$(hostname)"
echo "PWD=$ROOT"
echo "workspace_visible=$(test -d /workspace && echo yes || echo no)"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== MADAROS_VERSION ==="
env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE "$SOUC" --version 2>&1 | head -3
echo "RESULT_HEADER	case	engine	verb	rc	diag	note"

run_one() {
  local case="$1" engine="$2" verb="$3" src="$4"
  local log="/tmp/eps_${case}_${engine}_${verb}.log"
  local rc=0
  echo ""
  echo "=== CASE case=$case engine=$engine verb=$verb src=$src ==="
  set +e
  if [[ "$engine" == "madaros" ]]; then
    env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE \
      "$SOUC" "$verb" "$src" >"$log" 2>&1
    rc=$?
  else
    env -u SOUC_BIN -u SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE=lean_single \
      "$SOUC" "$verb" "$src" >"$log" 2>&1
    rc=$?
  fi
  set -e
  local diag="none"
  if grep -qoE 'error\[E[0-9]+\]' "$log"; then
    diag="$(grep -oE 'error\[E[0-9]+\]' "$log" | head -1)"
  elif grep -qoE 'E[0-9]{3}' "$log"; then
    diag="$(grep -oE 'E[0-9]{3}' "$log" | head -1)"
  elif grep -q 'unknown field' "$log"; then
    diag="unknown_field_warning"
  fi
  echo "rc=$rc diag=$diag"
  echo "----- log -----"
  if [[ "$verb" == "run" ]]; then
    # keep numbers; drop banner
    grep -E '^[0-9]|warning:|error|check:|unknown field' "$log" || tail -20 "$log"
  else
    tail -25 "$log"
  fi
  echo "RESULT	$case	$engine	$verb	$rc	$diag	ok"
}

for f in pos_value neg_epsilom sentinel_epsilon epsilom_third write_confidence; do
  src="docs/audit/repro/epsilon/${f}.sio"
  run_one "$f" madaros check "$src"
  run_one "$f" lean_single check "$src"
  run_one "$f" madaros run "$src"
  run_one "$f" lean_single run "$src"
done
echo "=== DONE ==="
REMOTE
chmod +x "$REMOTE"

echo "[epsilon] packing + srun partition=$PARTITION"
tar czf - \
  _epsilon_remote.sh \
  bin/souc bin/madaros bin/madaros-linux-x86_64 bin/souc-lean-single-x86_64 \
  docs/audit/repro/epsilon \
  | srun -p "$PARTITION" -N1 -n1 -c "$CPUS" --mem=8G --time="$TIMELIMIT" \
      --job-name=epsilon-inerte --chdir=/tmp \
      --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp,HOME=/tmp \
      /bin/bash -lc 'set -e; rm -rf /tmp/eps-in; mkdir -p /tmp/eps-in; cd /tmp/eps-in; tar xzf -; bash _epsilon_remote.sh' \
      >"$OUT" 2>&1
rc=$?
echo "[epsilon] srun rc=$rc log=$OUT bytes=$(wc -c <"$OUT")"
rm -f "$REMOTE"
exit $rc
