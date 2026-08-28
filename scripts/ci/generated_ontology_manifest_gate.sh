#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="stdlib/ontology/generated/MANIFEST.tsv"
EXPECTED_HEADER=$'bundle\tstub\tontology\tclass_count\tclass_limit\tconst_count\tdisjoint_count\tdisjoint_limit\tsurface\tpositive_witness\tnegative_witnesses\ttyped_bridge\ttyped_witness'

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
while IFS=$'\t' read -r bundle stub ontology class_count class_limit const_count disjoint_count disjoint_limit surface positive_witness negative_witnesses typed_bridge typed_witness; do
  if [[ "$bundle" == "bundle" ]]; then
    continue
  fi
  rows=$((rows + 1))

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
echo "their stubs, positive/negative witnesses, typed bridges, and declared"
echo "class/const/disjoint limits without making Python part of PL reasoning."
