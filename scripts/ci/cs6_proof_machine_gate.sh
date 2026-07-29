#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
contract="$repo_root/scripts/research/cs6_multiple_shooting_contract.py"
witness="$repo_root/scripts/research/cs6_multiple_shooting_witness_v1.json"
capd_source="$repo_root/scripts/research/cs6_capd_periodic_orbit.cpp"
capd_certificate="$repo_root/scripts/research/cs6_capd_periodic_orbit_certificate_v1.txt"
note="$repo_root/docs/research/cs6_proof_machine_2026-07-28.md"

expected_source_sha="5dfa204e85ec2ed62a53713adf3b72312598aa408384026d584abbfc1e15fe35"
expected_certificate_sha="5ae7a2154204870170639f6075f5292edd00059e095a650726ea4ea1a6c44054"

for artifact in \
  "$contract" \
  "$witness" \
  "$capd_source" \
  "$capd_certificate" \
  "$note"; do
  test -s "$artifact"
done

actual_source_sha="$(sha256sum "$capd_source" | awk '{print $1}')"
actual_certificate_sha="$(sha256sum "$capd_certificate" | awk '{print $1}')"
test "$actual_source_sha" = "$expected_source_sha"
test "$actual_certificate_sha" = "$expected_certificate_sha"

test "$(grep -c 'DISJOINT_FROM_X=true' "$capd_certificate")" -eq 5
grep -Fxq 'NEWTON_INTERIOR=true' "$capd_certificate"
grep -Fxq 'PRIME_PERIOD_SIX=true' "$capd_certificate"
grep -Fxq 'HYPERBOLICITY_SEPARATED=true' "$capd_certificate"
grep -Fxq 'EXPANDING_MULTIPLIER=[-5.9473828066101087, -4.6481361856663446]' \
  "$capd_certificate"
grep -Fxq 'CONTRACTING_MULTIPLIER=[-2.1618758145611223e-35, -1.6895952739109806e-35]' \
  "$capd_certificate"
python3 - "$capd_certificate" <<'PY'
import re
import sys
from decimal import Decimal
from pathlib import Path

receipt = Path(sys.argv[1]).read_text(encoding="ascii")
bounds = re.findall(r"(?:^| )NORMAL_VELOCITY=\[([^,]+), ([^]]+)\]", receipt, re.M)
if len(bounds) != 6:
    raise SystemExit(f"expected six normal-velocity enclosures, found {len(bounds)}")
if any(Decimal(lower) <= 0 for lower, _ in bounds):
    raise SystemExit("a Poincare normal-velocity enclosure reaches zero")
print("CS6_CAPD_ORIENTED_RETURNS PASS")
PY
echo "CS6_CAPD_FROZEN_CERTIFICATE PASS"

python3 -m py_compile "$contract"
python3 "$contract" --mode smoke
python3 "$contract" --check-witness "$witness"

grep -Fq 'periodic_orbit_proved = true' "$note"
grep -Fq 'hyperbolicity_proved = true' "$note"
grep -Fq 'chaos_proved = false' "$note"
grep -Fq 'U250 is not in the trusted computing base' "$note"
grep -Fq 'BLK-20260728-cs6-u250-resource-absent' "$note"
grep -Fq 'BLK-20260728-cs6-cluster-ops-auth-bridge' "$note"

if [[ "${CS6_FULL:-0}" == "1" ]]; then
  replay_dir="$(mktemp -d)"
  trap 'rm -rf "$replay_dir"' EXIT
  python3 "$contract" \
    --mode full \
    --quiet-progress \
    --replay-witness "$witness" \
    --summary-output "$replay_dir/summary.json" \
    --full-output "$replay_dir/full.json"
fi

if [[ "${CS6_CAPD_REPLAY:-0}" == "1" ]]; then
  capd_config="${CS6_CAPD_CONFIG:-capd-config}"
  if ! command -v "$capd_config" >/dev/null 2>&1; then
    echo "CS6_CAPD_REPLAY REFUSED: capd-config is unavailable" >&2
    exit 3
  fi
  replay_dir="${replay_dir:-$(mktemp -d)}"
  trap 'rm -rf "$replay_dir"' EXIT
  # capd-config intentionally emits the compiler and linker arguments.
  # shellcheck disable=SC2046
  "${CXX:-c++}" -std=c++17 -O2 "$capd_source" \
    $("$capd_config" --cflags --libs) \
    -o "$replay_dir/cs6_capd"
  "$replay_dir/cs6_capd" 1e-8 30 > "$replay_dir/capd.txt"
  cmp "$capd_certificate" "$replay_dir/capd.txt"
  echo "CS6_CAPD_REPLAY PASS"
fi

echo "CS6_PROOF_MACHINE_GATE PASS"
