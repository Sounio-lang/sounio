#!/usr/bin/env bash
# Assemble a promotable chi>=6 candidate manifest from an integrated preflight.
#
# This is intentionally fail-closed. It refuses unless the preflight has already
# classified the same source as READY_FOR_CANDIDATE_PROMOTION_WIRING, meaning
# exact rational geometry and checked cube-cover SAT artifacts both exist. It
# does not run search and does not invent missing no-5 evidence.
set -euo pipefail

usage() {
  echo "usage: $0 <integrated-preflight.out>" >&2
  echo "requires: OFFLOAD_REVIEW_RAW=/path/to/offload-review.txt for READY inputs" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFLIGHT="$1"
WORK="${WORK:-$(mktemp -d)}"
LEAN_DIR="$ROOT/formal/lean4"

GEN="$ROOT/examples/erdos/gen_lean_chi6_promotable_candidate.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
PROMOTABLE_VALIDATOR="$ROOT/examples/erdos/validate_chi6_promotable_candidate.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -s "$PREFLIGHT" ]] || { echo "error: missing/empty integrated preflight: $PREFLIGHT" >&2; exit 2; }
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
python3 -m py_compile "$GEN"

mkdir -p "$WORK"

declare -A PRE
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" == *=* ]] || continue
  key="${line%%=*}"
  val="${line#*=}"
  PRE[$key]="$val"
done < "$PREFLIGHT"

need_pre() {
  local key="$1"
  if [[ -z "${PRE[$key]+x}" || -z "${PRE[$key]}" ]]; then
    echo "error: preflight missing $key" >&2
    exit 2
  fi
  printf '%s\n' "${PRE[$key]}"
}

CANDIDATE_ID="$(need_pre candidate_id)"
GEOMETRY_STATUS="$(need_pre geometry_status)"
SAT_STATUS="$(need_pre sat_status)"
INTEGRATED_STATUS="$(need_pre integrated_status)"
FIRST_BLOCKER="${PRE[first_blocker]:-unknown}"
GEOM_MANIFEST="$(need_pre geometry_manifest)"
SAT_MANIFEST="$(need_pre sat_manifest)"

if [[ "$GEOMETRY_STATUS" != "PASS" || "$SAT_STATUS" != "PASS" || \
      "$INTEGRATED_STATUS" != "READY_FOR_CANDIDATE_PROMOTION_WIRING" ]]; then
  echo "error: preflight is not promotion-ready: geometry_status=$GEOMETRY_STATUS sat_status=$SAT_STATUS integrated_status=$INTEGRATED_STATUS first_blocker=$FIRST_BLOCKER" >&2
  exit 2
fi
[[ "$GEOM_MANIFEST" != "NONE" && -s "$GEOM_MANIFEST" ]] || {
  echo "error: promotion-ready preflight lacks concrete geometry_manifest" >&2
  exit 2
}
[[ "$SAT_MANIFEST" != "NONE" && -s "$SAT_MANIFEST" ]] || {
  echo "error: promotion-ready preflight lacks concrete sat_manifest" >&2
  exit 2
}
[[ -n "${OFFLOAD_REVIEW_RAW:-}" && -s "$OFFLOAD_REVIEW_RAW" ]] || {
  echo "error: READY promotion requires OFFLOAD_REVIEW_RAW pointing to a non-empty review artifact" >&2
  exit 2
}

declare -A GEOM SAT
read_manifest_into() {
  local file="$1"
  local arr_name="$2"
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    eval "$arr_name[\"\$key\"]=\"\$val\""
  done < "$file"
}
read_manifest_into "$GEOM_MANIFEST" GEOM
read_manifest_into "$SAT_MANIFEST" SAT

mneed() {
  local arr_name="$1"
  local key="$2"
  local value
  value="$(eval "printf '%s' \"\${$arr_name[$key]:-}\"")"
  if [[ -z "$value" ]]; then
    echo "error: manifest missing $key" >&2
    exit 2
  fi
  printf '%s\n' "$value"
}

resolve_from_manifest() {
  local manifest="$1"
  local raw="$2"
  if [[ "$raw" == "NONE" ]]; then
    printf 'NONE\n'
  elif [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/%s\n' "$(cd "$(dirname "$manifest")" && pwd)" "$raw"
  fi
}

sha() {
  [[ -f "$1" ]] || { echo "error: file not found for hashing: $1" >&2; exit 1; }
  sha256sum "$1" | awk '{print $1}'
}

module_suffix() {
  printf '%s\n' "$1" | sed 's/[^A-Za-z0-9_]/_/g; s/^[0-9]/_&/'
}

for key in candidate_id n m k edge_sha256; do
  if [[ "$(mneed GEOM "$key")" != "$(mneed SAT "$key")" ]]; then
    echo "error: component manifest mismatch for $key" >&2
    exit 2
  fi
done

SUFFIX="$(module_suffix "$CANDIDATE_ID")"
[[ -n "$SUFFIX" ]] || { echo "error: empty module suffix for candidate_id=$CANDIDATE_ID" >&2; exit 2; }
SAT_MODULE="SounioChi6Promotable_${SUFFIX}_Sat"
GEOM_MODULE="SounioChi6Promotable_${SUFFIX}_Geometry"
JOIN_MODULE="SounioChi6Promotable_${SUFFIX}"

SAT_SRC="$(resolve_from_manifest "$SAT_MANIFEST" "$(mneed SAT lean_sat_module_path)")"
GEOM_SRC="$(resolve_from_manifest "$GEOM_MANIFEST" "$(mneed GEOM geometry_module_path)")"
SAT_DST="$LEAN_DIR/$SAT_MODULE.lean"
GEOM_DST="$LEAN_DIR/$GEOM_MODULE.lean"
JOIN_DST="$LEAN_DIR/$JOIN_MODULE.lean"
MANIFEST="$WORK/candidate.manifest"

for path in "$SAT_DST" "$GEOM_DST" "$JOIN_DST"; do
  [[ ! -e "$path" ]] || { echo "error: refusing to overwrite existing Lean module: $path" >&2; exit 2; }
done

cp "$SAT_SRC" "$SAT_DST"
cp "$GEOM_SRC" "$GEOM_DST"
python3 "$GEN" "$GEOM_MANIFEST" "$SAT_MANIFEST" "$JOIN_DST" \
  --module "$JOIN_MODULE" \
  --namespace "$JOIN_MODULE" \
  --sat-import "$SAT_MODULE" \
  --geometry-import "$GEOM_MODULE" > "$WORK/gen_join.out"

rg -q '^status=lean_chi6_promotable_candidate_emitted$' "$WORK/gen_join.out"
SAT_EDGES_TERM="$(awk -F= '$1 == "sat_edges_term" {print $2; exit}' "$WORK/gen_join.out")"
[[ -n "$SAT_EDGES_TERM" ]] || { echo "error: join generator did not report sat_edges_term" >&2; exit 1; }

abs_field() {
  local manifest="$1"
  local raw="$2"
  resolve_from_manifest "$manifest" "$raw"
}

cat > "$MANIFEST" <<EOF
candidate_manifest_version=1
promotable=1
candidate_id=$CANDIDATE_ID
n=$(mneed SAT n)
m=$(mneed SAT m)
k=5
edge_path=$(abs_field "$SAT_MANIFEST" "$(mneed SAT edge_path)")
edge_sha256=$(mneed SAT edge_sha256)
cnf_path=$(abs_field "$SAT_MANIFEST" "$(mneed SAT cnf_path)")
cnf_sha256=$(mneed SAT cnf_sha256)
drat_or_lrat_path=$(abs_field "$SAT_MANIFEST" "$(mneed SAT drat_or_lrat_path)")
drat_or_lrat_sha256=$(mneed SAT drat_or_lrat_sha256)
lean_sat_module_path=$SAT_DST
lean_sat_module_sha256=$(sha "$SAT_DST")
geometry_module_path=$JOIN_DST
geometry_module_sha256=$(sha "$JOIN_DST")
geometry_source_path=$(abs_field "$GEOM_MANIFEST" "$(mneed GEOM geometry_source_path)")
geometry_source_sha256=$(mneed GEOM geometry_source_sha256)
geometry_proof_type=euclidean
sat_proof_route=$(mneed SAT sat_proof_route)
triangle_sb=$(mneed SAT triangle_sb)
generator_commit=$(git -C "$ROOT" rev-parse --verify HEAD)
producer_command=WORK=$WORK OFFLOAD_REVIEW_RAW=$OFFLOAD_REVIEW_RAW examples/erdos/make_chi6_promotable_candidate_manifest.sh $PREFLIGHT
lean_build_command=lake build $SAT_MODULE $GEOM_MODULE $JOIN_MODULE
offload_review_raw=$OFFLOAD_REVIEW_RAW
offload_review_sha256=$(sha "$OFFLOAD_REVIEW_RAW")
source_meta_path=$(abs_field "$SAT_MANIFEST" "${SAT[source_meta_path]:-NONE}")
source_meta_sha256=${SAT[source_meta_sha256]:-NONE}
cube_batch_path=$(abs_field "$SAT_MANIFEST" "${SAT[cube_batch_path]:-NONE}")
cube_batch_sha256=${SAT[cube_batch_sha256]:-NONE}
cube_refutation_summary_path=$(abs_field "$SAT_MANIFEST" "${SAT[cube_refutation_summary_path]:-NONE}")
cube_refutation_summary_sha256=${SAT[cube_refutation_summary_sha256]:-NONE}
cube_cover_certificate_path=$(abs_field "$SAT_MANIFEST" "${SAT[cube_cover_certificate_path]:-NONE}")
cube_cover_certificate_sha256=${SAT[cube_cover_certificate_sha256]:-NONE}
cube_cover_complement_cnf_path=$(abs_field "$SAT_MANIFEST" "${SAT[cube_cover_complement_cnf_path]:-NONE}")
cube_cover_complement_cnf_sha256=${SAT[cube_cover_complement_cnf_sha256]:-NONE}
cube_cover_complement_lrat_path=$(abs_field "$SAT_MANIFEST" "${SAT[cube_cover_complement_lrat_path]:-NONE}")
cube_cover_complement_lrat_sha256=${SAT[cube_cover_complement_lrat_sha256]:-NONE}
lean_module=$JOIN_MODULE
lean_sat_edges_term=$SAT_EDGES_TERM
lean_point_type=UnitDistanceChromatic.$JOIN_MODULE.point_type
lean_unit_term=UnitDistanceChromatic.$JOIN_MODULE.unit
lean_geometry_term=UnitDistanceChromatic.$JOIN_MODULE.geometry
lean_edges_sync_term=UnitDistanceChromatic.$JOIN_MODULE.edgesSync
lean_no_five_witness_term=UnitDistanceChromatic.$JOIN_MODULE.noFiveWitness
lean_final_theorem=UnitDistanceChromatic.$JOIN_MODULE.finalTheorem
lean_real_unit_term=UnitDistanceChromatic.$JOIN_MODULE.realUnit
lean_real_unit_iff_standard=UnitDistanceChromatic.$JOIN_MODULE.realUnitIffStandard
lean_real_final_theorem=UnitDistanceChromatic.$JOIN_MODULE.realFinalTheorem
EOF

"$VALIDATOR" "$MANIFEST" > "$WORK/manifest_validator.out"
"$PROMOTABLE_VALIDATOR" "$MANIFEST" > "$WORK/promotable_validator.out"

echo "manifest=$MANIFEST"
echo "lean_sat_module=$SAT_DST"
echo "lean_geometry_source_module=$GEOM_DST"
echo "lean_join_module=$JOIN_DST"
echo "chi6_promotable_candidate_manifest: PASS"
