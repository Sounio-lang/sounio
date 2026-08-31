#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="stdlib/ontology/generated/MANIFEST.tsv"
EXPECTED_HEADER=$'bundle\tstub\tontology\tclass_count\tclass_limit\tconst_count\tdisjoint_count\tdisjoint_limit\tsurface\tpositive_witness\tnegative_witnesses\ttyped_bridge\ttyped_witness\tupstream_release\tfetched_at\tsource_uri\tsource_sha256'

if [[ ! -f "$MANIFEST" ]]; then
  echo "FAIL  missing generated ontology manifest: $MANIFEST" >&2
  exit 1
fi

header="$(sed -n '1p' "$MANIFEST")"
if [[ "$header" != "$EXPECTED_HEADER" ]]; then
  echo "FAIL  generated ontology manifest header is not stable" >&2
  echo "expected: $EXPECTED_HEADER" >&2
  echo "actual:   $header" >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Version provenance columns (added 2026-08-19, dispatch bundle-version).
# Why these names:
#   * upstream_release   - exact identifier of the community release the
#                          bundle corresponds to (e.g. "ChEBI release 227",
#                          "GO 2026-03-01", "HPO v2026-01-16", "LOINC 2.77").
#                          Not the bundle's own version, not the import date.
#   * fetched_at         - ISO 8601 (YYYY-MM-DD) of when the source file
#                          was obtained from upstream. UTC, no time zone.
#   * source_uri         - canonical URI of the upstream artefact. Not the
#                          local in-tree path.
#   * source_sha256      - SHA-256 of the in-repo source file
#                          (stdlib/data/data/ontology/source/<bundle>_slice.json).
#                          Computed now and compared for drift, not stored
#                          against the upstream.
# UNKNOWN policy:
#   The founder's directive is that we record what the bundle came FROM,
#   not what we wish it came from. "UNKNOWN" is a legitimate value for
#   upstream_release, fetched_at, and source_uri when no community release
#   has been imported; it is NEVER acceptable to invent a plausible
#   identifier. The gate does not fail on UNKNOWN, but it ALWAYS prints
#   the list of bundles with any UNKNOWN column even when passing - same
#   visibility pattern the founder chose for Reserved-Since and
#   Evidence-Does-Not-Count. No expiration, no age blocking; the visibility
#   is the only constraint.
#   source_sha256 is NEVER allowed to be UNKNOWN: the source file is always
#   present in the tree, so a real hash must be computable.
# --------------------------------------------------------------------------
readonly UNKNOWN_RELEASE="UNKNOWN"

assert_sha256() {
  local value="$1"
  local bundle="$2"
  if [[ "$value" == "$UNKNOWN_RELEASE" ]]; then
    echo "FAIL  source_sha256 must be a real SHA-256 for $bundle (got UNKNOWN)" >&2
    echo "Why: the source file is in-tree, so the hash is always computable." >&2
    exit 1
  fi
  if ! [[ "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FAIL  source_sha256 must be a 64-char lowercase hex for $bundle (got: $value)" >&2
    exit 1
  fi
}

assert_optional_unknown_or_nonempty() {
  # Empty string is not allowed for any of the three provenance columns;
  # UNKNOWN is the only missing-information marker.
  local value="$1"
  local column="$2"
  local bundle="$3"
  if [[ -z "$value" ]]; then
    echo "FAIL  $column is empty for $bundle (use UNKNOWN if not recorded)" >&2
    exit 1
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

actual_stems="$TMP_DIR/actual-stems.txt"
expected_stems="$TMP_DIR/expected-stems.txt"

cat >"$expected_stems" <<'EOF'
alg
chebi
go
hpo
loinc
part
phys
qm
snomed
EOF

awk -F '\t' 'NR > 1 {
  bundle = $1
  sub(/^stdlib\/data\/data\/ontology\/bundles\//, "", bundle)
  sub(/\.dontology$/, "", bundle)
  print bundle
}' "$MANIFEST" | sort >"$actual_stems"

if ! diff -u "$expected_stems" "$actual_stems" >"$TMP_DIR/stem.diff"; then
  echo "FAIL  generated ontology manifest does not cover the stable public bundle set" >&2
  cat "$TMP_DIR/stem.diff" >&2
  exit 1
fi

rows=0
unknown_bundles=()
# Read into an array to avoid bash's IFS=$'\t' + many-vars known issue: when
# an interior field is empty, "read" with N variables on a line of N fields
# collapses the empty fields to the rightmost variable rather than leaving
# position 14 empty. Reading into an array keeps the column <-> index mapping
# stable.
while IFS=$'\t' read -r -a row; do
  if [[ "${row[0]}" == "bundle" ]]; then
    continue
  fi
  if (( ${#row[@]} != 17 )); then
    echo "FAIL  manifest row has ${#row[@]} columns, expected 17:" >&2
    printf '        %s\n' "${row[@]}" >&2
    exit 1
  fi
  bundle="${row[0]}"
  stub="${row[1]}"
  ontology="${row[2]}"
  class_count="${row[3]}"
  class_limit="${row[4]}"
  const_count="${row[5]}"
  disjoint_count="${row[6]}"
  disjoint_limit="${row[7]}"
  surface="${row[8]}"
  positive_witness="${row[9]}"
  negative_witnesses="${row[10]}"
  typed_bridge="${row[11]}"
  typed_witness="${row[12]}"
  upstream_release="${row[13]}"
  fetched_at="${row[14]}"
  source_uri="${row[15]}"
  source_sha256="${row[16]}"
  rows=$((rows + 1))

  # Version provenance columns (Phase D: present but possibly UNKNOWN).
  assert_sha256 "$source_sha256" "$bundle"
  assert_optional_unknown_or_nonempty "$upstream_release" "upstream_release" "$bundle"
  assert_optional_unknown_or_nonempty "$fetched_at" "fetched_at" "$bundle"
  assert_optional_unknown_or_nonempty "$source_uri" "source_uri" "$bundle"
  if [[ "$fetched_at" != "$UNKNOWN_RELEASE" ]]; then
    if ! [[ "$fetched_at" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
      echo "FAIL  fetched_at for $bundle must be YYYY-MM-DD or UNKNOWN (got: $fetched_at)" >&2
      exit 1
    fi
  fi
  if [[ "$source_uri" != "$UNKNOWN_RELEASE" ]]; then
    if ! [[ "$source_uri" =~ ^https?:// ]]; then
      echo "FAIL  source_uri for $bundle must be an http(s) URL or UNKNOWN (got: $source_uri)" >&2
      exit 1
    fi
  fi
  # Check whether the stored hash matches the live file (drift detection).
  bundle_stem="$(basename "$bundle" .dontology)"
  source_file="stdlib/data/data/ontology/source/${bundle_stem}_slice.json"
  if [[ -f "$source_file" ]]; then
    actual_hash="$(sha256sum "$source_file" | awk '{print $1}')"
    if [[ "$actual_hash" != "$source_sha256" ]]; then
      echo "FAIL  source_sha256 drift for $bundle_stem: manifest=$source_sha256 actual=$actual_hash" >&2
      echo "Why: the in-tree source file changed; the manifest must be refreshed" >&2
      echo "     before any release-grade claim is made about this bundle." >&2
      exit 1
    fi
  fi
  if [[ "$upstream_release" == "$UNKNOWN_RELEASE" \
        || "$fetched_at" == "$UNKNOWN_RELEASE" \
        || "$source_uri" == "$UNKNOWN_RELEASE" ]]; then
    unknown_bundles+=("$bundle_stem")
  fi

  for path in "$bundle" "$stub" "$positive_witness" "$typed_bridge" "$typed_witness"; do
    if [[ ! -f "$path" ]]; then
      echo "FAIL  manifest path is missing: $path" >&2
      exit 1
    fi
  done

  actual_classes="$(grep -c '^[[:space:]]*class ' "$stub" || true)"
  if [[ "$actual_classes" != "$class_count" ]]; then
    echo "FAIL  class_count mismatch for $stub: manifest=$class_count actual=$actual_classes" >&2
    exit 1
  fi
  if (( class_count > class_limit )); then
    echo "FAIL  class_count exceeds class_limit for $stub: $class_count > $class_limit" >&2
    exit 1
  fi

  actual_consts="$(grep -c '^const ' "$stub" || true)"
  if [[ "$actual_consts" != "$const_count" ]]; then
    echo "FAIL  const_count mismatch for $stub: manifest=$const_count actual=$actual_consts" >&2
    exit 1
  fi

  actual_disjoint="$(grep -c '^[[:space:]]*disjoint ' "$stub" || true)"
  if [[ "$actual_disjoint" != "$disjoint_count" ]]; then
    echo "FAIL  disjoint_count mismatch for $stub: manifest=$disjoint_count actual=$actual_disjoint" >&2
    exit 1
  fi
  if (( disjoint_count > disjoint_limit )); then
    echo "FAIL  disjoint_count exceeds disjoint_limit for $stub: $disjoint_count > $disjoint_limit" >&2
    exit 1
  fi

  if [[ "$surface" != classes+subclass+* ]]; then
    echo "FAIL  manifest surface does not declare classes+subclass for $stub: $surface" >&2
    exit 1
  fi
  if (( const_count == 0 )); then
    expected_const_surface="no-numeric-constants"
  else
    expected_const_surface="numeric-constants"
  fi
  if [[ "$surface" != *"$expected_const_surface"* ]]; then
    echo "FAIL  manifest surface does not match const_count for $stub: $surface" >&2
    exit 1
  fi
  if (( disjoint_count == 0 )); then
    expected_disjoint_surface="no-disjointness"
  else
    expected_disjoint_surface="disjoint"
  fi
  if [[ "$surface" != *"$expected_disjoint_surface"* ]]; then
    echo "FAIL  manifest surface does not match disjoint_count for $stub: $surface" >&2
    exit 1
  fi

  if [[ "$negative_witnesses" != "-" ]]; then
    IFS=',' read -ra witnesses <<<"$negative_witnesses"
    for witness in "${witnesses[@]}"; do
      if [[ ! -f "$witness" ]]; then
        echo "FAIL  manifest negative witness is missing: $witness" >&2
        exit 1
      fi
    done
  fi

  if ! grep -q 'The \.dontology importer only ingests data; Sounio owns reasoning\.' "$positive_witness"; then
    echo "FAIL  positive witness does not preserve importer/reasoner boundary note: $positive_witness" >&2
    exit 1
  fi
done <"$MANIFEST"

if [[ "$rows" -ne 9 ]]; then
  echo "FAIL  generated ontology manifest should have 9 stable rows, found $rows" >&2
  exit 1
fi

echo "Generated ontology manifest gate passed."
echo "This proves the stable public manifest covers the nine generated bundles,"
echo "their stubs, positive/negative witnesses, typed bridges, declared"
echo "class/const/disjoint limits, and upstream-source provenance, without"
echo "making Python part of PL reasoning."
if (( ${#unknown_bundles[@]} > 0 )); then
  echo ""
  echo "Bundles with UNKNOWN upstream provenance (visibility, not failure):"
  for bundle in "${unknown_bundles[@]}"; do
    echo "  - $bundle"
  done
  echo "These bundles have no recorded upstream community release. UNKNOWN is"
  echo "legitimate per founder directive (no plausible-number invention), but the"
  echo "list above is always printed - same visibility pattern as Reserved-Since"
  echo "and Evidence-Does-Not-Count. No expiration, no age blocking."
fi
