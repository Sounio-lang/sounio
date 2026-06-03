#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${SOUC_NATIVE_BIN:-}" && -x "${SOUC_NATIVE_BIN:-}" ]]; then
  export SOUNIO_SOUC_BIN="$SOUC_NATIVE_BIN"
fi

if [[ -n "${SOUNIO_SOUC_BIN:-}" && -x "${SOUNIO_SOUC_BIN:-}" ]]; then
  case "${1:-}" in
    check)
      if [[ $# -lt 2 ]]; then
        echo "usage: souc check <source.sio>" >&2
        exit 2
      fi
      tmp_out="$(mktemp /tmp/sounio-check-XXXXXX.elf)"
      trap 'rm -f "$tmp_out"' EXIT
      "$SOUNIO_SOUC_BIN" "$2" "$tmp_out"
      exit $?
      ;;
    run)
      if [[ $# -lt 2 ]]; then
        echo "usage: souc run <source.sio> [args...]" >&2
        exit 2
      fi
      src="$2"
      shift 2
      tmp_out="$(mktemp /tmp/sounio-run-XXXXXX.elf)"
      trap 'rm -f "$tmp_out"' EXIT
      "$SOUNIO_SOUC_BIN" "$src" "$tmp_out"
      chmod +x "$tmp_out"
      "$tmp_out" "$@"
      exit $?
      ;;
    compile)
      if [[ $# -lt 4 || "${3:-}" != "-o" ]]; then
        echo "usage: souc compile <source.sio> -o <out>" >&2
        exit 2
      fi
      "$SOUNIO_SOUC_BIN" "$2" "$4"
      exit $?
      ;;
  esac
fi

exec "$ROOT_DIR/bin/souc" "$@"
