#!/usr/bin/env bash
# Fetch the upstream ontologies the ontology-frontiers probes are measured against.
#
# These are THIRD-PARTY artefacts under their own licences and they are large
# (UBERON ~1.5M lines, CL ~1.0M). They were previously vendored into
# artifacts/ontology-frontiers/multi-ontology/downloads/ inside a PR, which is
# both a licence question and 2.5M lines of git history for files that are a URL
# away. A pinned checksum reproduces the exact input more convincingly than a
# copy does: a copy proves what someone committed, a checksum proves what they
# measured.
#
# Usage:  bash scripts/research/ontology-frontiers/fetch_upstream_ontologies.sh
# Then:   sha256sum -c scripts/research/ontology-frontiers/UPSTREAM_ONTOLOGIES.sha256
set -euo pipefail
DEST="${DEST:-artifacts/ontology-frontiers/multi-ontology/downloads}"
mkdir -p "$DEST"
fetch() {
  local url="$1" out="$DEST/$2"
  if [[ -s "$out" ]]; then echo "have  $out"; return; fi
  echo "fetch $url"
  curl -fsSL --retry 3 -o "$out" "$url"
}
fetch "http://purl.obolibrary.org/obo/uberon.owl" "uberon.owl"
fetch "http://purl.obolibrary.org/obo/cl.owl"     "cl.owl"
fetch "http://purl.obolibrary.org/obo/ro.owl"     "ro.owl"
echo
echo "Verify before using these as measurement inputs:"
echo "  sha256sum -c scripts/research/ontology-frontiers/UPSTREAM_ONTOLOGIES.sha256"
