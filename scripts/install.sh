#!/usr/bin/env bash
# scripts/install.sh — Public install path for Sounio (G6).
#
# Installs the checked self-hosted compiler artifacts, launcher, stdlib,
# and supporting libraries into a prefix.
#
# Usage:
#   bash scripts/install.sh [--prefix DIR] [--target TRIPLE] [--force]
#                           [--no-stdlib] [--symlink-souc]
#
# Defaults:
#   --prefix     $HOME/.local
#   --target     auto-detected from `uname -s/-m`
#
# Layout produced under <prefix>:
#   <prefix>/bin/souc                    (launcher; PATH-friendly)
#   <prefix>/bin/souc-self-hosted-<host> (native compiler binary)
#   <prefix>/bin/madaros                 (Stage1 modular compiler launcher, when built)
#   <prefix>/bin/madaros-linux-x86_64    (Stage1 modular compiler ELF, when built)
#   <prefix>/lib/sounio/stdlib/...       (stdlib tree)
#   <prefix>/lib/sounio/scripts/lib/...  (resolver + helpers used by launcher)
#   <prefix>/share/doc/sounio/...        (INSTALL.md, KNOWN_LIMITATIONS.md)
#
# Discriminator (G6):
#   bash scripts/install.sh --prefix=/tmp/sounio-test
#   /tmp/sounio-test/bin/souc --version    # prints wrapper version
#   /tmp/sounio-test/bin/souc info         # confirms paths

set -euo pipefail

# ------------------------------------------------------------------ args
PREFIX="${PREFIX:-$HOME/.local}"
TARGET=""
FORCE=0
COPY_STDLIB=1
SYMLINK_SOUC=0

print_usage() {
  sed -n '2,25p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)        PREFIX="$2"; shift 2;;
    --prefix=*)      PREFIX="${1#*=}"; shift;;
    --target)        TARGET="$2"; shift 2;;
    --target=*)      TARGET="${1#*=}"; shift;;
    --force)         FORCE=1; shift;;
    --no-stdlib)     COPY_STDLIB=0; shift;;
    --symlink-souc)  SYMLINK_SOUC=1; shift;;
    -h|--help)       print_usage; exit 0;;
    *)
      echo "error: unknown argument: $1" >&2
      print_usage >&2
      exit 2
      ;;
  esac
done

# Resolve repository root from this script's location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------ host
HOST_OS="${SOUNIO_HOST_OS_OVERRIDE:-$(uname -s)}"
HOST_ARCH="${SOUNIO_HOST_ARCH_OVERRIDE:-$(uname -m)}"
detect_target() {
  case "$HOST_OS:$HOST_ARCH" in
    Linux:x86_64|Linux:amd64)     echo "x86_64";;
    Linux:aarch64|Linux:arm64)    echo "aarch64";;
    Darwin:arm64|Darwin:aarch64)  echo "arm64-macos";;
    Darwin:x86_64|Darwin:amd64)   echo "x86_64-macos";;
    *) return 1;;
  esac
}

if [[ -z "$TARGET" ]]; then
  if ! TARGET="$(detect_target)"; then
    echo "error: unsupported host $HOST_OS:$HOST_ARCH; pass --target explicitly" >&2
    exit 2
  fi
fi

# Map target to shipped artifact filename.
case "$TARGET" in
  x86_64)         ART_BIN="$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64";;
  arm64-macos)    ART_BIN="$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-arm64-macos";;
  x86_64-macos)   ART_BIN="$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64-macos";;
  aarch64)
    echo "error: aarch64-linux artifact not shipped in this snapshot." >&2
    echo "hint: rebuild via 'make build' on an aarch64-linux host first." >&2
    exit 2
    ;;
  *) echo "error: unrecognised --target $TARGET" >&2; exit 2;;
esac
MADAROS_ART_BIN="$ROOT_DIR/artifacts/self-hosted/madaros"
MADAROS_PREBUILT_BIN="$ROOT_DIR/bin/madaros-linux-x86_64"
MADAROS_SHIP_BIN=""
if [[ -x "$MADAROS_ART_BIN" ]]; then
  MADAROS_SHIP_BIN="$MADAROS_ART_BIN"
elif [[ "$TARGET" == "x86_64" && -x "$MADAROS_PREBUILT_BIN" ]]; then
  MADAROS_SHIP_BIN="$MADAROS_PREBUILT_BIN"
fi

if [[ ! -x "$ART_BIN" ]]; then
  echo "error: missing artifact: $ART_BIN" >&2
  echo "hint: this checkout does not ship the binary for $TARGET" >&2
  exit 1
fi

# ------------------------------------------------------------------ plan
BIN_DIR="$PREFIX/bin"
LIB_ROOT="$PREFIX/lib/sounio"
LIB_STDLIB="$LIB_ROOT/stdlib"
LIB_SCRIPTS="$LIB_ROOT/scripts/lib"
DOC_DIR="$PREFIX/share/doc/sounio"

# Pre-flight: refuse to overwrite an existing install unless --force.
EXISTING_LAUNCHER="$BIN_DIR/souc"
if [[ -e "$EXISTING_LAUNCHER" && "$FORCE" -eq 0 ]]; then
  echo "error: $EXISTING_LAUNCHER already exists; pass --force to overwrite" >&2
  exit 2
fi

echo "Sounio install plan:"
echo "  prefix:  $PREFIX"
echo "  target:  $TARGET"
echo "  binary:  $ART_BIN -> $BIN_DIR/souc-self-hosted-$TARGET"
if [[ -n "$MADAROS_SHIP_BIN" ]]; then
  echo "  madaros: $MADAROS_SHIP_BIN -> $BIN_DIR/madaros-linux-x86_64"
else
  echo "  madaros: <not built; run make build-madaros to include Stage1>"
fi
echo "  stdlib:  $(du -sh stdlib | awk '{print $1}') -> $LIB_STDLIB  (copy: $COPY_STDLIB)"
echo "  doc:     $DOC_DIR"
echo

mkdir -p "$BIN_DIR" "$LIB_ROOT" "$LIB_SCRIPTS" "$DOC_DIR"

# ------------------------------------------------------------------ copy compiler artifact
DEST_NATIVE="$BIN_DIR/souc-self-hosted-$TARGET"
install -m 0755 "$ART_BIN" "$DEST_NATIVE"

# Also place a host-named compatibility binary for the launcher's
# `resolve_host_binary` lookup (which looks under $ROOT_DIR/bin/).
case "$TARGET" in
  x86_64)         install -m 0755 "$ART_BIN" "$BIN_DIR/souc-linux-x86_64";;
  arm64-macos)    install -m 0755 "$ART_BIN" "$BIN_DIR/souc-aarch64-macos";;
  x86_64-macos)   install -m 0755 "$ART_BIN" "$BIN_DIR/souc-x86_64-macos";;
esac

# ------------------------------------------------------------------ copy support libs
install -m 0644 "$ROOT_DIR/scripts/lib/resolve_souc.sh"      "$LIB_SCRIPTS/"
install -m 0644 "$ROOT_DIR/scripts/lib/macos_codesign.sh"    "$LIB_SCRIPTS/"

# ------------------------------------------------------------------ install launcher
# The shipped launcher resolves its location via $ROOT_DIR=$(.. up from bin/souc).
# After installation that root becomes $PREFIX, so the layout matches exactly:
#   <prefix>/bin/souc                                -> launcher
#   <prefix>/bin/souc-linux-x86_64 (or host-named)   -> native binary
#   <prefix>/scripts/lib/...                         -> launcher helpers
#   <prefix>/stdlib/...                              -> stdlib (linked from lib/sounio)
# We symlink top-level $PREFIX/stdlib and $PREFIX/scripts to the canonical
# lib/ locations so the launcher's relative lookups resolve cleanly.
install -m 0755 "$ROOT_DIR/bin/souc" "$BIN_DIR/souc"

if [[ -n "$MADAROS_SHIP_BIN" ]]; then
  install -m 0755 "$ROOT_DIR/bin/madaros" "$BIN_DIR/madaros"
  install -m 0755 "$MADAROS_SHIP_BIN" "$BIN_DIR/madaros-linux-x86_64"
fi

# Mirror scripts/lib at $PREFIX/scripts/lib so launcher's
# $ROOT_DIR/scripts/lib/macos_codesign.sh path resolves.
mkdir -p "$PREFIX/scripts/lib"
ln -sfn "$LIB_SCRIPTS/resolve_souc.sh"   "$PREFIX/scripts/lib/resolve_souc.sh"
ln -sfn "$LIB_SCRIPTS/macos_codesign.sh" "$PREFIX/scripts/lib/macos_codesign.sh"

# ------------------------------------------------------------------ stdlib
if [[ "$COPY_STDLIB" -eq 1 ]]; then
  rm -rf "$LIB_STDLIB"
  mkdir -p "$LIB_STDLIB"
  # Copy without preserving owner (cross-user installs).
  (cd "$ROOT_DIR" && tar -cf - stdlib) | (cd "$LIB_ROOT" && tar -xf -)
  # The launcher looks at $ROOT_DIR/stdlib; provide a symlink.
  ln -sfn "$LIB_STDLIB" "$PREFIX/stdlib"
fi

# ------------------------------------------------------------------ docs
install -m 0644 "$ROOT_DIR/INSTALL.md" "$DOC_DIR/INSTALL.md"
if [[ -f "$ROOT_DIR/docs/compiler/KNOWN_LIMITATIONS.md" ]]; then
  install -m 0644 "$ROOT_DIR/docs/compiler/KNOWN_LIMITATIONS.md" \
    "$DOC_DIR/KNOWN_LIMITATIONS.md"
fi

# ------------------------------------------------------------------ tools/repl.sh (G5b backing)
# The native REPL is a souc-native file-based driver. Ship it.
if [[ -f "$ROOT_DIR/tools/repl.sh" ]]; then
  mkdir -p "$LIB_ROOT/tools"
  install -m 0755 "$ROOT_DIR/tools/repl.sh" "$LIB_ROOT/tools/repl.sh"
  mkdir -p "$PREFIX/tools"
  ln -sfn "$LIB_ROOT/tools/repl.sh" "$PREFIX/tools/repl.sh"
fi

# ------------------------------------------------------------------ smoke
echo
echo "Sounio installed at: $PREFIX"
echo
echo "Smoke check:"
if "$BIN_DIR/souc" --version 2>/dev/null; then
  echo "  ok  souc --version"
else
  echo "  WARN  souc --version returned non-zero"
fi
if [[ -x "$BIN_DIR/madaros" ]]; then
  if "$BIN_DIR/madaros" --version 2>/dev/null; then
    echo "  ok  madaros --version"
  else
    echo "  WARN  madaros --version returned non-zero"
  fi
fi

cat <<EOF

Next steps:
  export PATH="$BIN_DIR:\$PATH"
  export SOUNIO_STDLIB_PATH="$LIB_STDLIB"
  souc --version
  souc info
  madaros --version       # if make build-madaros was run before install
  souc check examples/hello.sio   # if running from this checkout

Uninstall:
  rm -f $BIN_DIR/souc $BIN_DIR/souc-self-hosted-$TARGET $BIN_DIR/souc-linux-x86_64 $BIN_DIR/madaros $BIN_DIR/madaros-linux-x86_64
  rm -rf $LIB_ROOT $DOC_DIR $PREFIX/stdlib $PREFIX/scripts/lib $PREFIX/tools/repl.sh
EOF
