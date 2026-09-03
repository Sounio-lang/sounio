#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <first executable> <second executable>" >&2
  exit 64
fi

FIRST="$1"
SECOND="$2"
HOST_OS="${SOUNIO_HOST_OS_OVERRIDE:-$(uname -s 2>/dev/null || echo unknown)}"

if [[ "$HOST_OS" != "Darwin" ]] || ! command -v codesign >/dev/null 2>&1; then
  cmp -s "$FIRST" "$SECOND"
  exit $?
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-payload-compare.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
FIRST_COPY="$WORK_DIR/first"
SECOND_COPY="$WORK_DIR/second"
cp "$FIRST" "$FIRST_COPY"
cp "$SECOND" "$SECOND_COPY"

if codesign -dv "$FIRST_COPY" >/dev/null 2>&1; then
  codesign --remove-signature "$FIRST_COPY"
fi
if codesign -dv "$SECOND_COPY" >/dev/null 2>&1; then
  codesign --remove-signature "$SECOND_COPY"
fi

cmp -s "$FIRST_COPY" "$SECOND_COPY"
