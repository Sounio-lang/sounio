#!/usr/bin/env bash
# Gate: bedaquiline lung demo (demos/aris_bdq) at this commit, with a source-built compiler.
# Proves: parity with the C++23 oracle (rel 1e-5, via parity.sh), the honest positive claim passes,
#         and the three negative fixtures are refused with their stated diagnostics.
# Does not prove: that the model agrees with clinical data or that its assumptions are validated.
# Logs HEAD and the compiler sha256 as a receipt for this exact commit and binary.
set -euo pipefail
export LC_ALL=C

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

# Require a real compiler binary.
if [[ -z "${MADAROS_RAW_BIN:-}" || ! -f "$MADAROS_RAW_BIN" ]]; then
  echo "DEMO_ARIS_BDQ_GATE_FAIL: set MADAROS_RAW_BIN to a source-built compiler" >&2
  exit 64
fi

export SOUNIO_STDLIB_PATH="$repo/stdlib"
export SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_NOCACHE=1
export SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288 2>/dev/null || true

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Receipt for this commit and binary.
git rev-parse HEAD
sha256sum "$MADAROS_RAW_BIN"

failures=0

# Parity against the C++23 oracle.
rc=0
timeout 900 bash demos/aris_bdq/parity.sh > "$work/parity.log" 2>&1 || rc=$?
cat "$work/parity.log"
if [[ $rc -ne 0 ]] || ! grep -qx 'BDQ_PARITY_PASS' "$work/parity.log"; then
  echo "FAIL parity (rc=$rc)"
  failures=$((failures + 1))
fi

# The honest positive claim must run and pass.
rc=0
timeout 300 bin/souc run demos/aris_bdq/bdq_mic_claim_honest.sio > "$work/honest.log" 2>&1 || rc=$?
cat "$work/honest.log"
if [[ $rc -ne 0 ]] || ! grep -qF 'BDQ_MIC_CLAIM_HONEST_PASS' "$work/honest.log"; then
  echo "FAIL honest claim (rc=$rc)"
  failures=$((failures + 1))
fi

# Negative fixtures must be refused with the diagnostic named in their header.
for src in \
  demos/aris_bdq/bdq_mic_claim_overclaim.sio \
  demos/aris_bdq/units_refuse.sio \
  demos/aris_bdq/ontology_metabolite_refuse.sio
do
  name="$(basename "$src" .sio)"
  expect="$(sed -n 's|^//@ error-pattern: ||p' "$src" | sed -n '1p')"
  if [[ -z "$expect" ]]; then
    echo "FAIL $src (missing //@ error-pattern: header)"
    failures=$((failures + 1))
    continue
  fi

  log="$work/$name.log"
  rc=0
  timeout 300 bin/souc compile "$src" -o "$work/neg.elf" > "$log" 2>&1 || rc=$?
  if [[ $rc -ne 0 && $rc -ne 124 ]] && grep -qF -- "$expect" "$log"; then
    echo "ok   refused $src (rc=$rc)"
  else
    echo "FAIL $src (rc=$rc, expected: $expect)"
    cat "$log"
    failures=$((failures + 1))
  fi
done

if [[ $failures -eq 0 ]]; then
  echo "DEMO_ARIS_BDQ_GATE_PASS"
  exit 0
fi
echo "DEMO_ARIS_BDQ_GATE_FAIL failures=$failures"
exit 1
