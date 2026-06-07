#!/usr/bin/env bash
# scripts/release.sh — Assemble a distributable Sounio release tarball (G6).
#
# Stages the install layout into a clean directory, builds the real
# self-hosted modular compiler into that staged layout, and tarballs it.
# The resulting archive is "untar + add to PATH" — no compilation required
# by the user.
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

# install.sh stages the historical bootstrap/native artifacts. For a release
# artefact, build and ship the real self-hosted modular compiler produced from
# self-hosted/compiler/main.sio, then make the public launcher resolve to it.
if [[ "$TARGET" != "x86_64" ]]; then
  echo "error: real self-hosted modular release packaging is currently validated only for x86_64" >&2
  echo "hint: pass --target x86_64 on a Linux/x86_64 host, or add a target-specific mc build path first" >&2
  exit 2
fi

echo "building self-hosted modular compiler for release..."
MODULAR_SOUC="$PREFIX_DIR/bin/souc-modular"
rm -f "$MODULAR_SOUC"
(ulimit -s 1048576; "$ROOT_DIR/bin/souc" "$ROOT_DIR/self-hosted/compiler/main.sio" "$MODULAR_SOUC")
chmod +x "$MODULAR_SOUC"

MODULAR_VERSION_OUTPUT="$("$MODULAR_SOUC" --version 2>&1 || true)"
MODULAR_VERSION="${MODULAR_VERSION_OUTPUT%%$'\n'*}"
if [[ "$MODULAR_VERSION" != *Mad* ]]; then
  echo "error: staged modular compiler did not report a Madáres/Madares version" >&2
  echo "got: $MODULAR_VERSION" >&2
  exit 1
fi

# Keep the bootstrap mini_native, but label it honestly. The installed `souc`
# command is a tiny launcher for the real modular compiler.
if [[ -x "$PREFIX_DIR/bin/souc" ]]; then
  mv "$PREFIX_DIR/bin/souc" "$PREFIX_DIR/bin/souc-bootstrap-mini_native"
fi
cat > "$PREFIX_DIR/bin/souc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$BIN_DIR/souc-modular" "$@"
EOF
chmod +x "$PREFIX_DIR/bin/souc"

# Replace misleading/stale self-hosted and host-compat names with the real mc.
install -m 0755 "$MODULAR_SOUC" "$PREFIX_DIR/bin/souc-self-hosted-$TARGET"
install -m 0755 "$MODULAR_SOUC" "$PREFIX_DIR/bin/souc-linux-x86_64"

cat > "$PREFIX_DIR/bin/BINARIES.txt" <<EOF
Sounio release binaries
=======================

bin/souc
    PATH-friendly launcher that execs bin/souc-modular.

bin/souc-modular
    Real self-hosted modular compiler, built during release from
    self-hosted/compiler/main.sio.
    Version probe: $MODULAR_VERSION

bin/souc-self-hosted-$TARGET
bin/souc-linux-x86_64
    Compatibility names for the same real self-hosted modular compiler.

bin/souc-bootstrap-mini_native
    Historical bootstrap mini_native binary preserved for bootstrap/debugging.
    It is not the release compiler.
EOF

# Drop a bundled INSTALL-RELEASE.md describing post-extract steps.
cat > "$PREFIX_DIR/README.txt" <<EOF
Sounio $VERSION ($TARGET)
=========================

Extract and add to PATH:
    tar -xzf sounio-$VERSION-$TARGET.tar.gz
    export PATH="\$PWD/sounio-$VERSION-$TARGET/bin:\$PATH"
    export SOUNIO_STDLIB_PATH="\$PWD/sounio-$VERSION-$TARGET/lib/sounio/stdlib"
    souc --version
    souc --check path/to/file.sio
    souc --native-v2-compile path/to/file.sio -o program.elf

Documentation:
    share/doc/sounio/INSTALL.md
    share/doc/sounio/KNOWN_LIMITATIONS.md

Binary labels:
    bin/souc-modular is the real self-hosted modular compiler.
    bin/souc is a launcher for bin/souc-modular.
    bin/souc-bootstrap-mini_native is the historical bootstrap compiler.
    See bin/BINARIES.txt.

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
PKG_SOUC="$PKG_ROOT/bin/souc-modular"
# Compile and RUN a self-contained program with the packaged compiler, then
# assert its exit code. Use the real modular compiler explicitly; the staged
# `bin/souc` launcher also resolves to this binary.
SMOKE_SRC="$TMP_VERIFY/smoke.sio"
SMOKE_ELF="$TMP_VERIFY/smoke.elf"
printf 'fn main() -> i64 { 42 }\n' > "$SMOKE_SRC"
smoke_ok=0
version_ok=0
PKG_VERSION_OUTPUT="$("$PKG_SOUC" --version 2>&1 || true)"
PKG_VERSION="${PKG_VERSION_OUTPUT%%$'\n'*}"
if [[ "$PKG_VERSION" == *Mad* ]]; then
  echo "  ok  packaged modular compiler reports Madáres/Madares version: $PKG_VERSION"
  version_ok=1
else
  echo "  FAIL  packaged modular compiler did not report a Madáres/Madares version" >&2
  echo "        got: $PKG_VERSION" >&2
fi
rm -f "$SMOKE_ELF"
if "$PKG_SOUC" --native-v2-compile "$SMOKE_SRC" -o "$SMOKE_ELF" >/dev/null 2>&1 && [[ -f "$SMOKE_ELF" ]]; then
  chmod +x "$SMOKE_ELF" 2>/dev/null || true
  rc=0; "$SMOKE_ELF" || rc=$?
  if [[ "$rc" -eq 42 ]]; then
    echo "  ok  packaged modular compiler built and ran an import-free program (exit 42)"
    smoke_ok=1
  else
    echo "  FAIL  packaged program ran with exit $rc (expected 42)" >&2
  fi
else
  echo "  FAIL  packaged compiler could not compile a trivial program" >&2
fi
if [[ "$version_ok" -ne 1 || "$smoke_ok" -ne 1 ]]; then
  echo "error: release smoke check failed (artefact at $ARCHIVE)" >&2
  exit 1
fi
