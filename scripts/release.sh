#!/usr/bin/env bash
# scripts/release.sh — Assemble a distributable Sounio release tarball (G6).
#
# Stages the install layout into a clean directory and tarballs it. The
# resulting archive is "untar + add to PATH" — no compilation required.
#
# Usage:
#   bash scripts/release.sh [--target TRIPLE] [--version VER] [--out DIR]
#
# Output:
#   <out>/sounio-<version>-<target>.tar.gz
#   <out>/sounio-<version>-<target>.tar.gz.sha256

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${SOUNIO_RELEASE_VERSION:-}"
if [[ -z "$VERSION" ]]; then
  # RELEASE_POLICY.md: CITATION.cff is the single source of truth for version
  # metadata. Do NOT scrape bin/souc --version banner text -- that broke twice
  # (a stray newline in June, then the last-field grab landing on the word
  # "compiler" once the banner text changed again). A structured file does not
  # drift under free-text edits the way a human-readable banner does.
  VERSION="$(grep -m1 "^version:" "$ROOT_DIR/CITATION.cff" 2>/dev/null | sed "s/^version:[[:space:]]*//")"
  VERSION="${VERSION:-1.0.0-beta.6}"
fi
TARGET=""
OUT_DIR="$ROOT_DIR/dist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)     TARGET="$2"; shift 2;;
    --target=*)   TARGET="${1#*=}"; shift;;
    --version)    VERSION="$2"; shift 2;;
    --version=*)  VERSION="${1#*=}"; shift;;
    --out)        OUT_DIR="$2"; shift 2;;
    --out=*)      OUT_DIR="${1#*=}"; shift;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "error: unknown arg $1" >&2; exit 2;;
  esac
done

if [[ -z "$TARGET" ]]; then
  case "$(uname -s):$(uname -m)" in
    Linux:x86_64|Linux:amd64)   TARGET="x86_64";;
    Darwin:arm64|Darwin:aarch64) TARGET="arm64-macos";;
    Darwin:x86_64|Darwin:amd64) TARGET="x86_64-macos";;
    *) echo "error: unsupported host; pass --target" >&2; exit 2;;
  esac
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

PREFIX_DIR="$STAGE_DIR/sounio-$VERSION-$TARGET"
mkdir -p "$PREFIX_DIR"

# Reuse install.sh to stage the layout.
bash "$ROOT_DIR/scripts/install.sh" --prefix="$PREFIX_DIR" --target="$TARGET" --force >/dev/null

# Drop a bundled INSTALL-RELEASE.md describing post-extract steps.
cat > "$PREFIX_DIR/README.txt" <<EOF
Sounio $VERSION ($TARGET)
=========================

Extract and add to PATH:
    tar -xzf sounio-$VERSION-$TARGET.tar.gz
    export PATH="\$PWD/sounio-$VERSION-$TARGET/bin:\$PATH"
    export SOUNIO_STDLIB_PATH="\$PWD/sounio-$VERSION-$TARGET/lib/sounio/stdlib"
    souc --version
    souc info
    madaros --version

Documentation:
    share/doc/sounio/INSTALL.md
    share/doc/sounio/KNOWN_LIMITATIONS.md

Maturity: this is a beta release. Honest scope lives in
share/doc/sounio/KNOWN_LIMITATIONS.md. Read it before claims.
EOF

mkdir -p "$OUT_DIR"
ARCHIVE="$OUT_DIR/sounio-$VERSION-$TARGET.tar.gz"
(cd "$STAGE_DIR" && tar -czf "$ARCHIVE" "sounio-$VERSION-$TARGET")

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$OUT_DIR" && sha256sum "$(basename "$ARCHIVE")" > "$ARCHIVE.sha256")
elif command -v shasum >/dev/null 2>&1; then
  (cd "$OUT_DIR" && shasum -a 256 "$(basename "$ARCHIVE")" > "$ARCHIVE.sha256")
fi

echo "release artefact:"
ls -lh "$ARCHIVE" "$ARCHIVE.sha256" 2>/dev/null

echo
echo "smoke check (extract + --version in temp dir):"
TMP_VERIFY="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR" "$TMP_VERIFY"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP_VERIFY"
"$TMP_VERIFY/sounio-$VERSION-$TARGET/bin/souc" --version
if [[ -x "$TMP_VERIFY/sounio-$VERSION-$TARGET/bin/madaros" ]]; then
  "$TMP_VERIFY/sounio-$VERSION-$TARGET/bin/madaros" --version
fi
