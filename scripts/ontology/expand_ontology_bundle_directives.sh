#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMPORTER="${SOUNIO_DONTOLOGY_IMPORTER:-/tmp/sounio-dontology-ffi-importer}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ontology/expand_ontology_bundle_directives.sh [--cache-dir DIR] <input.sio> <output.sio>

Expands source lines of the form:

  //@ ontology-bundle: "stdlib/data/data/ontology/bundles/snomed.dontology"

into ingestion-only ontology stubs emitted by the C .dontology importer. This is
a pre-native bridge: it proves the directive payload can be ingested without
putting Python in the PL reasoning path, but it is not yet the compiler-native
side-table loader or native side-table .ontocache persistence path.

When --cache-dir is supplied, each expanded bundle is cached as a deterministic
text .ontocache keyed by the bundle bytes and C importer source bytes. This is
still a pre-native cache of expanded declarations, not compiler side tables.
EOF
}

cache_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --cache-dir requires a directory" >&2
        exit 2
      fi
      cache_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unsupported option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

INPUT="$1"
OUTPUT="$2"

case "$INPUT" in
  /*) INPUT_ABS="$INPUT" ;;
  *) INPUT_ABS="$ROOT_DIR/$INPUT" ;;
esac
case "$OUTPUT" in
  /*) OUTPUT_ABS="$OUTPUT" ;;
  *) OUTPUT_ABS="$ROOT_DIR/$OUTPUT" ;;
esac

if [[ ! -f "$INPUT_ABS" ]]; then
  echo "error: input not found: $INPUT" >&2
  exit 1
fi

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

"$ROOT_DIR/scripts/ontology/build_dontology_ffi_importer.sh" "$IMPORTER" >/dev/null
mkdir -p "$(dirname "$OUTPUT_ABS")"
if [[ -n "$cache_dir" ]]; then
  case "$cache_dir" in
    /*) CACHE_DIR_ABS="$cache_dir" ;;
    *) CACHE_DIR_ABS="$ROOT_DIR/$cache_dir" ;;
  esac
  mkdir -p "$CACHE_DIR_ABS"
else
  CACHE_DIR_ABS=""
fi

: >"$OUTPUT_ABS"
{
  printf '// Expanded by scripts/ontology/expand_ontology_bundle_directives.sh\n'
  printf '// Pre-native ontology-bundle bridge: C importer ingests data; Sounio checker owns reasoning.\n\n'
} >>"$OUTPUT_ABS"

found=0
while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]ontology-bundle:[[:space:]]\"([^\"]+)\"[[:space:]]*$ ]]; then
    bundle="${BASH_REMATCH[1]}"
    case "$bundle" in
      /*) bundle_abs="$bundle" ;;
      *) bundle_abs="$ROOT_DIR/$bundle" ;;
    esac
    if [[ ! -f "$bundle_abs" ]]; then
      echo "error: ontology bundle not found: $bundle" >&2
      exit 1
    fi
    found=1
    printf '// Begin expanded ontology-bundle: %s\n' "$bundle" >>"$OUTPUT_ABS"
    if [[ -n "$CACHE_DIR_ABS" ]]; then
      bundle_hash="$(hash_file "$bundle_abs")"
      importer_hash="$(hash_file "$ROOT_DIR/scripts/ontology/dontology_ffi_importer.c")"
      bundle_base="$(basename "$bundle_abs" .dontology)"
      cache_file="$CACHE_DIR_ABS/${bundle_base}.${bundle_hash:0:16}.${importer_hash:0:16}.ontocache"
      if [[ -f "$cache_file" ]]; then
        printf 'ontology-bundle cache HIT %s\n' "$cache_file" >&2
      else
        tmp_cache="$cache_file.tmp.$$"
        {
          printf '// Sounio pre-native ontology .ontocache v1\n'
          printf '// bundle: %s\n' "$bundle"
          printf '// bundle_sha256: %s\n' "$bundle_hash"
          printf '// importer_sha256: %s\n\n' "$importer_hash"
          "$IMPORTER" "$bundle_abs" ""
        } >"$tmp_cache"
        mv "$tmp_cache" "$cache_file"
        printf 'ontology-bundle cache MISS %s\n' "$cache_file" >&2
      fi
      cat "$cache_file" >>"$OUTPUT_ABS"
    else
      "$IMPORTER" "$bundle_abs" "" >>"$OUTPUT_ABS"
    fi
    printf '\n// End expanded ontology-bundle: %s\n\n' "$bundle" >>"$OUTPUT_ABS"
  else
    printf '%s\n' "$line" >>"$OUTPUT_ABS"
  fi
done <"$INPUT_ABS"

if [[ "$found" -eq 0 ]]; then
  echo "error: no //@ ontology-bundle directive found in $INPUT" >&2
  exit 1
fi
