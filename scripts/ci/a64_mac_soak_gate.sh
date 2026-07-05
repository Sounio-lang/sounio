#!/usr/bin/env bash
# a64 Mac hardware soak gate: cross-compiles the lvalue witness corpus to
# aarch64-macos with the committed lean_single seed, ships it to a real
# Apple Silicon Mac over ssh, codesigns, runs, and asserts every witness
# green. This is the gate form of the 2026-07-05 campaign harness that took
# the a64 lane from 3/86 to fully green (see
# docs/audit/A64_AGGREGATE_SUBSTRATE_2026-07-05.md).
#
# Usage: scripts/ci/a64_mac_soak_gate.sh
#
# Env:
#   SOUNIO_MAC_HOST        ssh destination (default demetriosagourakis@100.91.184.41)
#   SOUNIO_MAC_KEY         ssh identity file (default ~/.ssh/id_ed25519_dgx_spark)
#   SOUNIO_MAC_SOAK_SEED   compiler binary (default bin/souc-lean-single-x86_64)
#   SOUNIO_MAC_SOAK_REQUIRED  if "1", an unreachable Mac FAILS the gate
#                             (default: clean SKIP, exit 0 — the Mac is a
#                             laptop and sleeps)
#
# Exit: 0 all green (or SKIP when unreachable and not required),
#       1 any witness red / compile failure / required-but-unreachable.
#
# Remote notes (macOS): no `timeout`; default shell zsh (we pipe into
# `bash -s`); ad-hoc `codesign -s -` is mandatory for unsigned arm64.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MAC="${SOUNIO_MAC_HOST:-demetriosagourakis@100.91.184.41}"
KEY="${SOUNIO_MAC_KEY:-$HOME/.ssh/id_ed25519_dgx_spark}"
SEED="${SOUNIO_MAC_SOAK_SEED:-${REPO_ROOT}/bin/souc-lean-single-x86_64}"
REQUIRED="${SOUNIO_MAC_SOAK_REQUIRED:-0}"
SSH_OPTS=(-i "${KEY}" -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

say() { echo "[a64-mac-soak] $*"; }

# ── 1. Reachability probe (the Mac is a laptop; sleeping is normal) ──────
if ! ssh "${SSH_OPTS[@]}" "${MAC}" true 2>/dev/null; then
    if [ "${REQUIRED}" = "1" ]; then
        say "FAIL: Mac ${MAC} unreachable and SOUNIO_MAC_SOAK_REQUIRED=1"
        exit 1
    fi
    say "SKIP: Mac ${MAC} unreachable (laptop asleep?) — soak not run"
    exit 0
fi

# ── 2. Build the corpus: 80 shape-matrix witnesses + committed witnesses ─
WORK="$(mktemp -d /tmp/a64_mac_soak.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/src" "${WORK}/bin"

python3 - "${WORK}/src" <<'PYEOF'
import sys
sys.path.insert(0, "scripts/ci")
import lean_lvalue_shape_matrix as m
m.write_witnesses(sys.argv[1])
PYEOF

for f in tests/known_failures/lean_*.sio tests/known_failures/a64_*.sio; do
    # compile-fail-by-design witnesses are not runnable — skip them
    if head -1 "$f" | grep -q "compile-fail"; then continue; fi
    cp "$f" "${WORK}/src/"
done

total=0
cfail=0
for src in "${WORK}/src"/*.sio; do
    n="$(basename "${src}" .sio)"
    total=$((total + 1))
    if ! timeout 120 "${SEED}" "${src}" "${WORK}/bin/${n}.bin" --target aarch64-macos >/dev/null 2>&1; then
        say "COMPILE FAIL: ${n}"
        cfail=$((cfail + 1))
    fi
done
if [ "${cfail}" -ne 0 ]; then
    say "FAIL: ${cfail}/${total} witnesses did not cross-compile"
    exit 1
fi
say "corpus: ${total} witnesses cross-compiled to Mach-O arm64"

# ── 3. Ship + run on the Mac ─────────────────────────────────────────────
tar czf "${WORK}/batch.tgz" -C "${WORK}" bin
scp -q "${SSH_OPTS[@]}" "${WORK}/batch.tgz" "${MAC}:/tmp/sounio_a64_soak.tgz"
ssh "${SSH_OPTS[@]}" "${MAC}" 'bash -s' > "${WORK}/results.tsv" <<'REMOTE'
set -e
rm -rf /tmp/sounio_a64_soak && mkdir -p /tmp/sounio_a64_soak
tar xzf /tmp/sounio_a64_soak.tgz -C /tmp/sounio_a64_soak
cd /tmp/sounio_a64_soak/bin
for b in *.bin; do
    chmod +x "$b"; codesign -s - "$b" 2>/dev/null
    out=$("./$b" 2>&1); rc=$?
    line=$(printf '%s' "$out" | grep -m1 "PASS" || printf '%s' "$out" | head -1)
    printf '%s\t%s\t%s\n' "${b%.bin}" "$rc" "$line"
done
rm -rf /tmp/sounio_a64_soak /tmp/sounio_a64_soak.tgz
REMOTE

# ── 4. Verdict ───────────────────────────────────────────────────────────
# Green = rc 0 AND (prints PASS, or is a bare-value probe: a64_* / rc-only).
ran="$(wc -l < "${WORK}/results.tsv" | tr -d ' ')"
reds="$(awk -F'\t' '!($2==0 && ($3 ~ /PASS/ || $1 ~ /^a64_/)) {print $1" rc="$2" out="$3}' "${WORK}/results.tsv")"
if [ "${ran}" -ne "${total}" ]; then
    say "FAIL: shipped ${total} but only ${ran} results returned"
    exit 1
fi
if [ -n "${reds}" ]; then
    say "FAIL: red witnesses on Apple Silicon:"
    echo "${reds}" | sed 's/^/[a64-mac-soak]   /'
    exit 1
fi
say "PASS: ${ran}/${total} witnesses green on Apple Silicon hardware"
