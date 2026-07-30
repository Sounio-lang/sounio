#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_cone_scout.cpp"
receipt="$repo_root/scripts/research/cs6_cone_scout_receipt_v1.json"
note="$repo_root/docs/research/cs6_cone_scout_2026-07-30.md"

# Filled after the mandatory math review. Scientific receipt drift must fail
# loudly, while the optional live replay checks behavior on the current host.
expected_source_sha="ba1b3539fe9d358d8a284265638ee58b77fd0cb1042db6f1e55ed6e0a1bec9a8"
expected_receipt_sha="f5409b8df8f2275392d13232e4510d20e175ddb4a63c97d1c4d9afbdb96b6cd3"
expected_note_sha="cc3d38dcc51e2927c03a79b6e52f552b737078123c6622df572987a9fe03f0c0"

for artifact in "$source_file" "$receipt" "$note"; do
  test -s "$artifact"
done

test "$(sha256sum "$source_file" | awk '{print $1}')" = "$expected_source_sha"
test "$(sha256sum "$receipt" | awk '{print $1}')" = "$expected_receipt_sha"
test "$(sha256sum "$note" | awk '{print $1}')" = "$expected_note_sha"

python3 - "$receipt" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
assert receipt["schema"] == "sounio.cs6.cone-scout.v1"
assert receipt["map"]["name"] == "P^6"
assert receipt["map"]["returns_per_map"] == 6

execution = receipt["execution"]
assert execution["engine"] == "CAPD DPoincareMap double"
assert execution["discovery_grid"] == [17, 17]
assert execution["holdout_grid"] == [16, 16]
assert execution["discovery_layout"] == "endpoints"
assert execution["holdout_layout"] == "cell_midpoints"
assert execution["discovery_edge_derivative_records"] == 867
assert execution["holdout_edge_derivative_records"] == 768
assert execution["all_sample_records_valid"] is True
assert execution["deterministic_repeat_match"] is True

criterion = receipt["criterion"]
assert criterion["search_boundary_hit"] is False
assert criterion["weights_tuned_on_discovery"] is True
assert criterion["holdout_used_for_tuning"] is False
assert criterion["sampled_hull_arithmetic"] == (
    "long double without outward rounding"
)

for phase in (receipt["discovery"], receipt["holdout"]):
    assert phase["global_min_normalized_point_margin"] > 0
    assert phase["nonrigorous_sampled_entrywise_hulls_pd_sufficient"] is True
    assert len(phase["edges"]) == 3
    assert all(edge["sampled_hull_det_m_lower"] > 0 for edge in phase["edges"])

diagnostics = receipt["diagnostics"]
assert diagnostics["max_finite_difference_relative_error"] < 1e-4
assert diagnostics["double_precision_invertibility_resolved"] is False
assert diagnostics["sampled_direction_rates_above_one"] is False
assert diagnostics["backward_rate_resolved"] is False

claims = receipt["claims"]
assert claims["numerical_cone_candidate_found"] is True
assert claims["sampled_positive_definite_matrix_candidate_found"] is True
assert claims["nonrigorous_sampled_entrywise_hull_pd_sufficient"] is True
for key in (
    "numerical_hyperbolicity_candidate_found",
    "pairwise_chord_cone_condition_proved",
    "tangent_cone_condition_proved",
    "uniform_hyperbolicity_proved",
    "chaotic_attractor_proved",
):
    assert claims[key] is False, key

assert receipt["next_rigorous_gate"]["status"] == "not implemented and not run"
assert set(receipt["blockers"]) == {
    "BLK-20260730-cs6-c1-interval-cone",
    "BLK-20260728-cs6-cluster-ops-auth-bridge",
    "BLK-20260730-docs-registry-baseline-drift",
}
assert receipt["u250_required"] is False
assert receipt["u250_in_trusted_computing_base"] is False
print("CS6_CONE_SCOUT_RECEIPT PASS")
PY

grep -Fq 'pairwise_chord_cone_condition_proved = false' "$note"
grep -Fq 'uniform_hyperbolicity_proved = false' "$note"
grep -Fq 'chaotic_attractor_proved = false' "$note"
grep -Fq 'SAMPLED_HULL_ARITHMETIC=long-double-no-outward-rounding' "$source_file"
grep -Fq 'CANDIDATE_ONLY=true' "$source_file"
grep -Fq 'PAIRWISE_CHORD_CONE_CONDITION_PROVED=false' "$source_file"
grep -Fq 'UNIFORM_HYPERBOLICITY_PROVED=false' "$source_file"
grep -Fq 'CHAOTIC_ATTRACTOR_PROVED=false' "$source_file"
grep -Fq 'BLK-20260730-cs6-c1-interval-cone' "$note"
grep -Fq 'BLK-20260728-cs6-cluster-ops-auth-bridge' "$note"
grep -Fq 'BLK-20260730-docs-registry-baseline-drift' "$note"
grep -Fq 'default Sounio interval path used = false' "$note"
grep -Fq 'rebuilt current-source CAPD path used = true' "$note"
grep -Fq 'fallback path used = false' "$note"
grep -Fq 'legacy numerical reconnaissance kept = true' "$note"
grep -Fq 'Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY' "$note"

# The inherited periodic-orbit and C0 candidate machinery remains a
# prerequisite. Its optional CAPD replay stays opt-in in its own gate.
bash "$repo_root/scripts/ci/cs6_fibonacci_scout_gate.sh"

if [[ "${CS6_CAPD_CONE_REPLAY:-0}" == "1" ]]; then
  capd_config="${CS6_CAPD_CONFIG:-capd-config}"
  if ! command -v "$capd_config" >/dev/null 2>&1; then
    echo "CS6_CAPD_CONE_REPLAY REFUSED: capd-config is unavailable" >&2
    exit 3
  fi

  replay_dir="$(mktemp -d)"
  trap 'rm -rf "$replay_dir"' EXIT
  # capd-config intentionally emits compiler and linker arguments.
  # shellcheck disable=SC2046
  "${CXX:-c++}" -std=c++17 -O3 "$source_file" \
    $("$capd_config" --cflags --libs) \
    -o "$replay_dir/cs6_cone_scout"

  "$replay_dir/cs6_cone_scout" selftest > "$replay_dir/selftest.txt"
  grep -Fxq 'NONFINITE_DERIVATIVE_REJECTED=true' "$replay_dir/selftest.txt"
  grep -Fxq 'HULL_ARITHMETIC_SELFTEST_PASS=true' "$replay_dir/selftest.txt"
  grep -Fxq 'SEARCH_TIEBREAK_SELFTEST_PASS=true' "$replay_dir/selftest.txt"
  grep -Fxq 'SELFTEST_PASS=true' "$replay_dir/selftest.txt"

  "$replay_dir/cs6_cone_scout" scout 17 > "$replay_dir/run-1.txt"
  "$replay_dir/cs6_cone_scout" scout 17 > "$replay_dir/run-2.txt"
  cmp "$replay_dir/run-1.txt" "$replay_dir/run-2.txt"

  grep -Fxq 'ALL_SAMPLE_RECORDS_VALID=true' "$replay_dir/run-1.txt"
  grep -Fxq 'SEARCH_BOUNDARY_HIT=false' "$replay_dir/run-1.txt"
  grep -Fxq 'HOLDOUT_USED_FOR_TUNING=false' "$replay_dir/run-1.txt"
  grep -Fxq \
    'SAMPLED_HULL_ARITHMETIC=long-double-no-outward-rounding' \
    "$replay_dir/run-1.txt"
  grep -Fxq \
    'NONRIGOROUS_SAMPLED_ENTRYWISE_HULL_PD_SUFFICIENT=true' \
    "$replay_dir/run-1.txt"
  grep -Fxq 'DOUBLE_PRECISION_INVERTIBILITY_RESOLVED=false' \
    "$replay_dir/run-1.txt"
  grep -Fxq 'NUMERICAL_CONE_CANDIDATE_FOUND=true' "$replay_dir/run-1.txt"
  grep -Fxq 'NUMERICAL_HYPERBOLICITY_CANDIDATE_FOUND=false' \
    "$replay_dir/run-1.txt"
  grep -Fxq 'PAIRWISE_CHORD_CONE_CONDITION_PROVED=false' \
    "$replay_dir/run-1.txt"
  grep -Fxq 'UNIFORM_HYPERBOLICITY_PROVED=false' "$replay_dir/run-1.txt"
  grep -Fxq 'CHAOTIC_ATTRACTOR_PROVED=false' "$replay_dir/run-1.txt"

  python3 - "$replay_dir/run-1.txt" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="ascii")
edge_lines = [line for line in text.splitlines() if line.startswith("PHASE=") and " EDGE=" in line]
assert len(edge_lines) == 6, len(edge_lines)
assert all("NONRIGOROUS_SAMPLED_HULL_PD_SUFFICIENT=true" in line for line in edge_lines)

def scalar(name: str) -> float:
    matches = re.findall(rf"^{re.escape(name)}=([^\s]+)$", text, flags=re.MULTILINE)
    assert len(matches) == 1, (name, matches)
    return float(matches[0])

assert scalar("DISCOVERY_GLOBAL_MIN_NORMALIZED_CONE_MARGIN") > 0
assert scalar("HOLDOUT_GLOBAL_MIN_NORMALIZED_CONE_MARGIN") > 0
assert scalar("MAX_FINITE_DIFFERENCE_REL_ERROR") < 1e-4
for line in edge_lines:
    assert float(re.search(r"SAMPLED_HULL_M00_LOWER=([^ ]+)", line).group(1)) > 0
    assert float(re.search(r"SAMPLED_HULL_DET_M_LOWER=([^ ]+)", line).group(1)) > 0
print("CS6_CONE_SCOUT_LIVE_REPLAY PASS")
PY

  if "$replay_dir/cs6_cone_scout" scout 201 \
      > "$replay_dir/heavy.out" 2> "$replay_dir/heavy.err"; then
    echo "cone scout accepted a prohibited local-heavy grid" >&2
    exit 1
  fi
  grep -Fq 'grid must be an odd integer in [3,41]' "$replay_dir/heavy.err"
  echo "CS6_CONE_SCOUT_OUTPUT_SHA256=$(sha256sum "$replay_dir/run-1.txt" | awk '{print $1}')"
fi

echo "CS6_CONE_SCOUT_GATE PASS"
