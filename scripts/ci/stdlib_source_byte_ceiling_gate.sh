#!/usr/bin/env bash
# stdlib_source_byte_ceiling_gate.sh — no stdlib .sio may exceed 2 MiB.
#
# CAP = 2097152 NO LONGER matches the lexer wall, and that is deliberate. It did
# until 2026-09-05, when CURSOR_SOURCE moved to 16777216 so Madaros could read
# its own 2.1 MB lean_single.sio. This gate keeps the 2 MiB number as a STDLIB
# POLICY, not as a mirror of the lexer: the failure it was built for was
# stdlib/theorem/portfolio.sio re-monolithing, and the fix for that is splitting
# the module, which is what the line below already says. Raising this to follow
# the lexer would retire a working policy for no reason.
# Soft warn at 1.5 MiB so catalogs cannot silently re-monolith toward the hard wall.
#
# Positive control: this gate must FAIL if a file over CAP is present. The historical
# failure mode was stdlib/theorem/portfolio.sio at 2109065 bytes (silent mid-item clip).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HARD=2097152
SOFT=1572864  # 1.5 MiB

fail() { echo "FAIL stdlib_source_byte_ceiling_gate: $*" >&2; exit 1; }
pass() { echo "PASS $*"; }

HARD_HITS=()
SOFT_HITS=()
while IFS= read -r -d '' f; do
  sz=$(wc -c <"$f")
  rel=${f#"$ROOT/"}
  if (( sz > HARD )); then
    HARD_HITS+=("$rel:$sz")
  elif (( sz > SOFT )); then
    SOFT_HITS+=("$rel:$sz")
  fi
done < <(find "$ROOT/stdlib" -name '*.sio' -print0)

echo "=== stdlib_source_byte_ceiling_gate ==="
echo "hard_cap=$HARD soft_warn=$SOFT"

if ((${#HARD_HITS[@]} > 0)); then
  for h in "${HARD_HITS[@]}"; do
    echo "HARD_OVER $h" >&2
  done
  fail "${#HARD_HITS[@]} file(s) exceed lexer source-byte ceiling $HARD — split the module (do not raise CAP to green)"
fi

for h in "${SOFT_HITS[@]:-}"; do
  [[ -n "$h" ]] || continue
  echo "SOFT_WARN $h (under hard cap; split before next growth)"
done

# Positive control: façade portfolio must exist and be small; parts must exist
[[ -f stdlib/theorem/portfolio.sio ]] || fail "missing portfolio façade"
psz=$(wc -c <stdlib/theorem/portfolio.sio)
(( psz < 65536 )) || fail "portfolio.sio looks monolithic again ($psz bytes) — expected thin façade"
[[ -f stdlib/theorem/portfolio_kinds.sio ]] || fail "missing portfolio_kinds.sio split part"
[[ -f stdlib/theorem/portfolio_core.sio ]] || fail "missing portfolio_core.sio split part"

# Positive control: lorenz cert façade (pre-split; was 2095899, 1253 under CAP)
[[ -f stdlib/systems/lorenz_i256_cert.sio ]] || fail "missing lorenz_i256_cert façade"
lsz=$(wc -c <stdlib/systems/lorenz_i256_cert.sio)
(( lsz < 65536 )) || fail "lorenz_i256_cert.sio looks monolithic again ($lsz bytes) — expected thin façade"
[[ -f stdlib/systems/lorenz_i256_cert_core.sio ]] || fail "missing lorenz_i256_cert_core.sio split part"
[[ -f stdlib/systems/lorenz_i256_cert_step1.sio ]] || fail "missing lorenz_i256_cert_step1.sio split part"
[[ -f stdlib/systems/lorenz_i256_cert_cover_child0.sio ]] || fail "missing lorenz_i256_cert_cover_child0.sio split part"

pass "no stdlib .sio over hard cap $HARD; portfolio façade=${psz}B lorenz façade=${lsz}B"
echo "PASS stdlib_source_byte_ceiling_gate"
