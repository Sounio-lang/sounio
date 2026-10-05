#!/usr/bin/env bash
# Pack a disposable Slurm job that type-checks the dissertation Knowledge
# constructor on both engines. /workspace is invisible on the node.
# Does not edit stdlib/darwin_pbpk/epistemic_pbpk28.sio or self-hosted/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PARTITION="${SOUNIO_REMOTE_PARTITION:-cpu-ops}"
CPUS="${SOUNIO_REMOTE_CPUS:-8}"
TIMELIMIT="${SOUNIO_REMOTE_TIME:-00:25:00}"
OUT="${E200_SLURM_OUT:-/tmp/e200_reverify_slurm.out}"
export SLURM_CONF="${SLURM_CONF:-/tmp/slurm-direct.conf}"

REMOTE="$ROOT/_e200_remote.sh"
cat >"$REMOTE" <<'REMOTE'
#!/usr/bin/env bash
set -uo pipefail
ulimit -s 1048576 || true
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
unset SOUC_BIN SOUNIO_SOUC_BIN || true

ROOT="$(pwd)"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="$ROOT/bin/souc"
chmod +x "$ROOT/bin/souc" "$ROOT/bin/madaros" \
  "$ROOT/bin/madaros-linux-x86_64" "$ROOT/bin/souc-lean-single-x86_64" 2>/dev/null || true

echo "HOST=$(hostname)"
echo "PWD=$ROOT"
echo "workspace_visible=$(test -d /workspace && echo yes || echo no)"
echo "orangefs_visible=$(test -d /orangefs && echo yes || echo no)"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== MADAROS_VERSION ==="
env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE \
  "$SOUC" --version 2>&1 | head -5
echo "=== LEAN_SINGLE_HELP ==="
env -u SOUC_BIN -u SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE=lean_single \
  "$SOUC" --help 2>&1 | head -8

run_one() {
  local case="$1" engine="$2" verb="$3" src="$4"
  local log="/tmp/e200_${case}_${engine}_${verb}.log"
  local rc=0
  echo ""
  echo "=== CASE case=$case engine=$engine verb=$verb src=$src ==="
  if [[ ! -f "$src" ]]; then
    echo "MISSING_SRC $src"
    echo "RESULT	$case	$engine	$verb	127	MISSING	-	missing_src"
    return
  fi
  set +e
  if [[ "$engine" == "madaros" ]]; then
    env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE \
      SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
      "$SOUC" "$verb" "$src" >"$log" 2>&1
    rc=$?
  else
    env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
      SOUNIO_SOUC_ENGINE=lean_single \
      SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
      "$SOUC" "$verb" "$src" >"$log" 2>&1
    rc=$?
  fi
  set -e
  local diag="none"
  local line="-"
  if grep -qoE 'error\[E[0-9]+\]' "$log"; then
    diag="$(grep -oE 'error\[E[0-9]+\]' "$log" | head -1)"
  elif grep -qoE 'E[0-9]{3}' "$log"; then
    diag="$(grep -oE 'E[0-9]{3}' "$log" | head -1)"
  fi
  if grep -qoE 'at line [0-9]+' "$log"; then
    line="$(grep -oE 'at line [0-9]+' "$log" | head -1 | awk '{print $3}')"
  elif grep -qoE ':[0-9]+:[0-9]+' "$log"; then
    line="$(grep -oE ':[0-9]+:[0-9]+' "$log" | head -1 | cut -d: -f2)"
  fi
  echo "rc=$rc diag=$diag line=$line"
  echo "----- log head -----"
  head -40 "$log"
  echo "----- log tail -----"
  tail -20 "$log"
  echo "RESULT	$case	$engine	$verb	$rc	$diag	$line	ok"
}

echo "RESULT_HEADER	case	engine	verb	rc	diag	line	status"

# Positive: same folder, known small program with main.
run_one pos_sort_fix madaros check stdlib/darwin_pbpk/test_sort_fix.sio
run_one pos_sort_fix lean_single check stdlib/darwin_pbpk/test_sort_fix.sio

# Target.
run_one ep28 madaros check stdlib/darwin_pbpk/epistemic_pbpk28.sio
run_one ep28 lean_single check stdlib/darwin_pbpk/epistemic_pbpk28.sio

# Isolates + negatives (check only).
for f in isolate_struct_lit isolate_call_form negative_wrong_field negative_no_value; do
  run_one "$f" madaros check "docs/audit/repro/e200/${f}.sio"
  run_one "$f" lean_single check "docs/audit/repro/e200/${f}.sio"
done

# Numeric isolates: compile+run so field values are observed.
for f in numeric_carrier numeric_reorder; do
  run_one "$f" madaros check "docs/audit/repro/e200/${f}.sio"
  run_one "$f" lean_single check "docs/audit/repro/e200/${f}.sio"
  run_one "$f" madaros run "docs/audit/repro/e200/${f}.sio"
  run_one "$f" lean_single run "docs/audit/repro/e200/${f}.sio"
done

echo "=== DONE ==="
REMOTE
chmod +x "$REMOTE"

echo "[e200] packing + srun partition=$PARTITION cpus=$CPUS"
# shellcheck disable=SC2046
tar czf - \
  _e200_remote.sh \
  bin/souc bin/madaros bin/madaros-linux-x86_64 bin/souc-lean-single-x86_64 \
  stdlib \
  docs/audit/repro/e200 \
  | srun -p "$PARTITION" -N1 -n1 -c "$CPUS" --mem=16G --time="$TIMELIMIT" \
      --job-name=e200-reverify --chdir=/tmp \
      --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp,HOME=/tmp \
      /bin/bash -lc 'set -e; rm -rf /tmp/e200-in; mkdir -p /tmp/e200-in; cd /tmp/e200-in; tar xzf -; bash _e200_remote.sh' \
      >"$OUT" 2>&1
rc=$?
echo "[e200] srun rc=$rc log=$OUT bytes=$(wc -c <"$OUT")"
rm -f "$REMOTE"
exit $rc
