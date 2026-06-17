#!/usr/bin/env bash
# Preflight one solver candidate source through both chi6 halves.
#
# This is a classifier, not a theorem producer. A source package can pass exact
# rational geometry while still failing the no-5 SAT/cube-cover route; that is
# useful evidence for the search loop and must remain non-promotional.
set -euo pipefail

usage() {
  echo "usage: $0 <candidate-source.json> [<cube-batch> <cover-drup-or-rup>]" >&2
  echo "example: WORK=/tmp/chi6-preflight $0 candidate-source.json" >&2
  echo "example: WORK=/tmp/chi6-preflight $0 candidate-source.json cubes.txt cover.drup" >&2
}

if [[ $# -ne 1 && $# -ne 3 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_JSON="$1"
CUBE_BATCH_IN="${2:-}"
COVER_DRUP_IN="${3:-}"
WORK="${WORK:-$(mktemp -d)}"

SOURCE_VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
GEOM_MAKER="$ROOT/examples/erdos/make_chi6_rational_geometry_candidate_manifest.sh"
SAT_MAKER_SPLIT="$ROOT/examples/erdos/make_chi6_external_cube_cover_candidate_manifest.sh"
SAT_MAKER_ARBITRARY="$ROOT/examples/erdos/make_chi6_external_arbitrary_cube_cover_candidate_manifest.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
[[ -s "$SOURCE_JSON" ]] || { echo "error: missing/empty candidate source JSON: $SOURCE_JSON" >&2; exit 2; }

mkdir -p "$WORK"
SOURCE_OUT="$WORK/source_validator.out"
SOURCE_ERR="$WORK/source_validator.err"
GEOM_OUT="$WORK/geometry_maker.out"
GEOM_ERR="$WORK/geometry_maker.err"
SAT_OUT="$WORK/sat_maker.out"
SAT_ERR="$WORK/sat_maker.err"

field() {
  local key="$1"
  local values
  mapfile -t values < <(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print}' "$SOURCE_OUT")
  if [[ "${#values[@]}" -ne 1 ]]; then
    echo "error: source validator emitted ${#values[@]} rows for required key: $key" >&2
    exit 2
  fi
  printf '%s\n' "${values[0]}"
}

backup_existing_dir() {
  local dir="$1"
  [[ -e "$dir" ]] || return 0
  local base backup
  base="$(basename "$dir")"
  backup="$WORK/.previous-${base}.$(date +%Y%m%dT%H%M%S).$$"
  mv "$dir" "$backup"
}

echo "chi6_integrated_candidate_preflight: workdir=$WORK"
if ! "$SOURCE_VALIDATOR" "$SOURCE_JSON" > "$SOURCE_OUT" 2> "$SOURCE_ERR"; then
  cat "$SOURCE_ERR" >&2
  exit 2
fi
if ! rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$SOURCE_OUT"; then
  echo "error: source validator did not emit VALID_SOLVER_CANDIDATE_PACKAGE" >&2
  exit 2
fi

CANDIDATE_ID="$(field candidate_id)"
EDGE_ABS="$(field edge_path_abs)"
EDGE_SHA="$(field edge_sha256)"
COORDS_ABS="$(field coords_path_abs)"
COORDS_SHA="$(field coords_sha256)"
SPLIT_VERTICES="$(field split_vertices)"
N="$(field n)"
M="$(field m)"
K="$(field k)"

[[ -n "$CANDIDATE_ID" && -n "$EDGE_ABS" && -n "$COORDS_ABS" && -n "$SPLIT_VERTICES" ]] || {
  echo "error: source validator output missing required preflight fields" >&2
  exit 2
}
[[ "$(sha256sum "$EDGE_ABS" | awk '{print $1}')" == "$EDGE_SHA" ]] || {
  echo "error: source validator edge_sha256 no longer matches edge artifact" >&2
  exit 1
}
[[ "$(sha256sum "$COORDS_ABS" | awk '{print $1}')" == "$COORDS_SHA" ]] || {
  echo "error: source validator coords_sha256 no longer matches coordinate artifact" >&2
  exit 1
}

if [[ $# -eq 3 ]]; then
  sat_route_mode="arbitrary_complement_cube_cover"
else
  sat_route_mode="split_vertices_product_cube_cover"
fi

backup_existing_dir "$WORK/geometry"
backup_existing_dir "$WORK/sat"

geometry_status="FAIL"
geometry_blocker="none"
if WORK="$WORK/geometry" "$GEOM_MAKER" "$SOURCE_JSON" > "$GEOM_OUT" 2> "$GEOM_ERR"; then
  geometry_status="PASS"
else
  geometry_blocker="geometry_exactness_or_lean_generation"
fi

sat_status="FAIL"
sat_blocker="none"
if [[ "$sat_route_mode" == "arbitrary_complement_cube_cover" ]]; then
  if [[ ! -s "$CUBE_BATCH_IN" || ! -s "$COVER_DRUP_IN" ]]; then
    sat_blocker="sat_arbitrary_cube_cover_inputs_absent"
    {
      echo "error: arbitrary SAT route requires non-empty cube batch and cover DRUP/RUP"
      echo "cube_batch=${CUBE_BATCH_IN:-NONE}"
      echo "cover_drup_or_rup=${COVER_DRUP_IN:-NONE}"
    } > "$SAT_ERR"
  elif WORK="$WORK/sat" "$SAT_MAKER_ARBITRARY" "$EDGE_ABS" "$CANDIDATE_ID" \
      "$CUBE_BATCH_IN" "$COVER_DRUP_IN" > "$SAT_OUT" 2> "$SAT_ERR"; then
    sat_status="PASS"
  else
    sat_blocker="sat_arbitrary_cube_cover_refutation_absent"
  fi
else
  if WORK="$WORK/sat" "$SAT_MAKER_SPLIT" "$EDGE_ABS" "$CANDIDATE_ID" "$SPLIT_VERTICES" \
      > "$SAT_OUT" 2> "$SAT_ERR"; then
    sat_status="PASS"
  else
    sat_blocker="sat_no5_cube_cover_refutation_absent"
  fi
fi

if [[ "$geometry_status" == "PASS" && "$sat_status" == "PASS" ]]; then
  integrated_status="READY_FOR_CANDIDATE_PROMOTION_WIRING"
  first_blocker="none"
elif [[ "$geometry_status" != "PASS" ]]; then
  integrated_status="INCOMPLETE"
  first_blocker="$geometry_blocker"
else
  integrated_status="INCOMPLETE"
  first_blocker="$sat_blocker"
fi

echo "candidate_id=$CANDIDATE_ID"
echo "n=$N"
echo "m=$M"
echo "k=$K"
echo "edge_path_abs=$EDGE_ABS"
echo "edge_sha256=$EDGE_SHA"
echo "coords_path_abs=$COORDS_ABS"
echo "coords_sha256=$COORDS_SHA"
echo "split_vertices=$SPLIT_VERTICES"
echo "sat_route_mode=$sat_route_mode"
echo "cube_batch_input=$([[ -n "$CUBE_BATCH_IN" ]] && printf '%s' "$CUBE_BATCH_IN" || printf 'NONE')"
echo "cover_drup_or_rup_input=$([[ -n "$COVER_DRUP_IN" ]] && printf '%s' "$COVER_DRUP_IN" || printf 'NONE')"
echo "source_status=PASS"
echo "geometry_status=$geometry_status"
echo "geometry_manifest=$([[ -s "$WORK/geometry/candidate.manifest" ]] && printf '%s' "$WORK/geometry/candidate.manifest" || printf 'NONE')"
echo "geometry_blocker=$geometry_blocker"
echo "sat_status=$sat_status"
echo "sat_manifest=$([[ -s "$WORK/sat/candidate.manifest" ]] && printf '%s' "$WORK/sat/candidate.manifest" || printf 'NONE')"
echo "sat_blocker=$sat_blocker"
echo "integrated_status=$integrated_status"
echo "first_blocker=$first_blocker"
echo "claim_scope=integrated_preflight_only"
echo "promotable=0"
echo "geometry_claim=exact_rational_squared_distance_edges_only_if_geometry_status_PASS"
echo "sat_claim=checked_no5_cube_cover_only_if_sat_status_PASS"
echo "chromatic_claim=none"
echo "source_validator_log=$SOURCE_OUT"
echo "source_validator_stderr=$SOURCE_ERR"
echo "geometry_stdout=$GEOM_OUT"
echo "geometry_stderr=$GEOM_ERR"
echo "sat_stdout=$SAT_OUT"
echo "sat_stderr=$SAT_ERR"
echo "chi6_integrated_candidate_preflight: PASS"
