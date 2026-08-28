#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <souc-bin> <selfhost-entry.sio> [args...]" >&2
  exit 2
fi

ROOT_DIR="$(pwd)"

SOUC_ARG="$1"
shift
ENTRY_PATH="$1"
shift

case "$SOUC_ARG" in
  /*) SOUC_BIN="$SOUC_ARG" ;;
  *) SOUC_BIN="$ROOT_DIR/$SOUC_ARG" ;;
esac

if [ ! -x "$SOUC_BIN" ]; then
  echo "souc binary not executable: $SOUC_BIN" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-selfhost-fresh.XXXXXX")"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

mkdir -p "$TMP_ROOT/self-hosted"
find "$ROOT_DIR/self-hosted" -mindepth 1 -maxdepth 1 ! -name '.*' -exec cp -R {} "$TMP_ROOT/self-hosted/" \;

if [ -d "$ROOT_DIR/stdlib" ]; then
  cp -R "$ROOT_DIR/stdlib" "$TMP_ROOT/"
fi
if [ -d "$ROOT_DIR/examples" ]; then
  cp -R "$ROOT_DIR/examples" "$TMP_ROOT/"
fi
if [ -d "$ROOT_DIR/tests" ]; then
  cp -R "$ROOT_DIR/tests" "$TMP_ROOT/"
fi
if [ -d "$ROOT_DIR/artifacts" ]; then
  ln -s "$ROOT_DIR/artifacts" "$TMP_ROOT/artifacts"
fi

case "$ENTRY_PATH" in
  "$ROOT_DIR"/*) ENTRY_REL="${ENTRY_PATH#"$ROOT_DIR"/}" ;;
  /*) echo "entry path must be inside $ROOT_DIR: $ENTRY_PATH" >&2; exit 2 ;;
  *) ENTRY_REL="$ENTRY_PATH" ;;
esac

cd "$TMP_ROOT"
env \
  SOUNIO_LOWER_SUMMARY_TRACE=0 \
  SOUNIO_LOWER_LIVE_TRACE=0 \
  SOUNIO_LOWER_BODY_TRACE=0 \
  SOUNIO_LOWER_ORDERED_TRACE=0 \
  SOUNIO_NATIVE_STREAM_DIAG=0 \
  "$SOUC_BIN" run "$ENTRY_REL" "$@"
