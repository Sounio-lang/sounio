#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_ARTIFACTS_DIR="$ROOT_DIR/paper/artifacts"
OUT_BASE="${PAPER_SUBMISSION_PACK_DIR:-$ROOT_DIR/artifacts/omega/paper_submission_pack}"

SOURCE_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/paper/paper_submission_pack.sh [--source-dir PATH] [--out-dir PATH]

Options:
  --source-dir PATH   Source paper artifact run directory (default: latest with status_summary.v1.json)
  --out-dir PATH      Output base directory (default: artifacts/omega/paper_submission_pack)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_BASE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_DIR" ]]; then
  latest_status="$(ls -1dt "$PAPER_ARTIFACTS_DIR"/*/status_summary.v1.json 2>/dev/null | head -n1 || true)"
  if [[ -z "$latest_status" ]]; then
    echo "error: could not find paper/artifacts/*/status_summary.v1.json" >&2
    exit 1
  fi
  SOURCE_DIR="$(dirname "$latest_status")"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "error: source artifact directory not found: $SOURCE_DIR" >&2
  exit 1
fi

PACK_ID="$(date +%Y-%m-%d_%H%M%S)"
PACK_DIR="$OUT_BASE/$PACK_ID"
BUNDLE_DIR="$PACK_DIR/bundle"
MANIFEST_PATH="$PACK_DIR/MANIFEST.v1.json"
CHECKSUMS_PATH="$PACK_DIR/SHA256SUMS.txt"
TARBALL_PATH="$PACK_DIR/paper_submission_pack_${PACK_ID}.tar.gz"
TARBALL_SHA_PATH="$PACK_DIR/paper_submission_pack_${PACK_ID}.tar.gz.sha256"

mkdir -p "$BUNDLE_DIR"
rsync -a "$SOURCE_DIR"/ "$BUNDLE_DIR"/

(
  cd "$BUNDLE_DIR"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$CHECKSUMS_PATH"

python3 - "$BUNDLE_DIR" "$MANIFEST_PATH" "$SOURCE_DIR" "$PACK_ID" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

bundle_dir = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
source_dir = pathlib.Path(sys.argv[3])
pack_id = sys.argv[4]

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = []
for path in sorted(bundle_dir.rglob("*")):
    if path.is_file():
        rel = path.relative_to(bundle_dir).as_posix()
        files.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

manifest = {
    "schema": "sounio.paper.submission-pack.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "pack_id": pack_id,
    "source_artifact_dir": os.path.abspath(str(source_dir)),
    "file_count": len(files),
    "files": files,
}

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

tar -czf "$TARBALL_PATH" -C "$PACK_DIR" bundle SHA256SUMS.txt MANIFEST.v1.json
sha256sum "$TARBALL_PATH" >"$TARBALL_SHA_PATH"

echo "[paper-submission-pack] source_dir=${SOURCE_DIR#$ROOT_DIR/}"
echo "[paper-submission-pack] pack_dir=${PACK_DIR#$ROOT_DIR/}"
echo "[paper-submission-pack] manifest=${MANIFEST_PATH#$ROOT_DIR/}"
echo "[paper-submission-pack] checksums=${CHECKSUMS_PATH#$ROOT_DIR/}"
echo "[paper-submission-pack] tarball=${TARBALL_PATH#$ROOT_DIR/}"
echo "[paper-submission-pack] tarball_sha256=${TARBALL_SHA_PATH#$ROOT_DIR/}"
