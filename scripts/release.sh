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

# Version source: the shipped compiler (mini_native) has no `--version` and its
# usage line is not a version, so do NOT parse it — doing so yielded a garbage,
# newline-containing string (and, under `set -o pipefail`, fired the `|| echo`
# fallback too, concatenating both). Prefer an explicit override, else a VERSION
# file, else a pinned fallback. Sanitised below before it names any artefact.
VERSION="${SOUNIO_RELEASE_VERSION:-}"
if [[ -z "$VERSION" ]]; then
  if [[ -f "$ROOT_DIR/VERSION" ]]; then
    VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
  else
    VERSION="1.0.0-beta.5"
  fi
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

# Reject a version that would corrupt the artefact name (whitespace, newline,
# slash). Cheap guard against the class of bug that produced a newline in the
# tarball filename.
if [[ -z "$VERSION" || "$VERSION" =~ [[:space:]/] ]]; then
  echo "error: invalid release version: '$VERSION'" >&2
  echo "hint: pass --version VER or set SOUNIO_RELEASE_VERSION" >&2
  exit 2
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
echo "smoke check (extract + compile + run a program from the package):"
TMP_VERIFY="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR" "$TMP_VERIFY"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP_VERIFY"
PKG_ROOT="$TMP_VERIFY/sounio-$VERSION-$TARGET"
PKG_SOUC="$PKG_ROOT/bin/souc"
# Compile and RUN a self-contained program with the packaged compiler, then
# assert its exit code. `--version` was a hollow check: the shipped binary
# ignores it, so it proved nothing about whether the package can compile.
SMOKE_SRC="$TMP_VERIFY/smoke.sio"
SMOKE_ELF="$TMP_VERIFY/smoke.elf"
printf 'fn main() -> i64 {\n    let x = 21\n    x * 2\n}\n' > "$SMOKE_SRC"
smoke_ok=0
if "$PKG_SOUC" "$SMOKE_SRC" "$SMOKE_ELF" >/dev/null 2>&1 && [[ -f "$SMOKE_ELF" ]]; then
  chmod +x "$SMOKE_ELF" 2>/dev/null || true
  rc=0; "$SMOKE_ELF" || rc=$?
  if [[ "$rc" -eq 42 ]]; then
    echo "  ok  packaged compiler built and ran a program (21*2 -> exit 42)"
    smoke_ok=1
  else
    echo "  FAIL  packaged program ran with exit $rc (expected 42)" >&2
  fi
else
  echo "  FAIL  packaged compiler could not compile a trivial program" >&2
fi
if [[ "$smoke_ok" -ne 1 ]]; then
  echo "error: release smoke check failed (artefact at $ARCHIVE)" >&2
  exit 1
fi
