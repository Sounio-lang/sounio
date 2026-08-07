#!/usr/bin/env bash
# ADR-008: claim = C l8_zd_census_fast VERDICT PASS + emits hash;
# numpy cross-hash is foreign corroboration soft unless HARD=1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "$REPO_ROOT/scripts/ci/lib_sounio_claim_oracle.sh"
SRC="${REPO_ROOT}/scripts/research/l8_zd_census_fast.c"
HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"
if [[ ! -f "${SRC}" ]]; then echo "L8_ZD_CENSUS_GATE_FAIL: missing ${SRC}"; exit 1; fi
CC_BIN="${CC:-cc}"
command -v "${CC_BIN}" >/dev/null 2>&1 || { echo "L8_ZD_CENSUS_GATE_FAIL: no C compiler"; exit 1; }
PYTHON="${REPO_ROOT}/.venv/bin/python3"
[[ -x "${PYTHON}" ]] || PYTHON="python3"
SCRATCH="$(mktemp -d)"; trap 'rm -rf "${SCRATCH}"' EXIT
BIN="${SCRATCH}/l8_zd_census_fast"
echo "Building level-8 ZD census fast contract..."
"${CC_BIN}" -O2 -Wall -Wextra -o "${BIN}" "${SRC}"
echo "Running level-8 ZD census..."
OUT="$("${BIN}")" || { echo "${OUT}"; echo "L8_ZD_CENSUS_GATE_FAIL: contract exited non-zero"; exit 1; }
echo "${OUT}"
grep -q '^L8_ZD_FAST_VERDICT PASS$' <<<"${OUT}" || { echo "L8_ZD_CENSUS_GATE_FAIL: verdict not PASS"; exit 1; }
C_HASH="$(sed -n 's/^L8_FAST_SIGN_TABLE .*fnv1a=\([0-9a-f]\{16\}\)$/\1/p' <<<"${OUT}")"
[[ -n "${C_HASH}" ]] || { echo "L8_ZD_CENSUS_GATE_FAIL: missing C hash"; exit 1; }
echo "== claim: C contract PASS (hash=${C_HASH}) =="
echo "== corroboration: numpy sign-table hash (HARD=$HARD) =="
set +e
PY_HASH="$(REPO_ROOT="${REPO_ROOT}" "${PYTHON}" - <<'EOF'
import os, sys
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "scripts", "research"))
from routon_zd_contract import get_sign_matrix
S = get_sign_matrix(8)
h = 1469598103934665603
for b in S.astype("<i1").tobytes():
    h ^= b
    h = (h * 1099511628211) % (1 << 64)
print(f"{h:016x}")
EOF
)"
set -e
echo "Sign-table cross-hash: C=${C_HASH} numpy=${PY_HASH}"
if [[ -z "${PY_HASH}" || "${C_HASH}" != "${PY_HASH}" ]]; then
  sounio_foreign_mismatch "sign-table hash C!=numpy" || true
  if [ "$HARD" = "1" ]; then echo "L8_ZD_CENSUS_GATE_FAIL foreign_hard"; exit 1; fi
fi
echo "L8_ZD_CENSUS_GATE_OK"
