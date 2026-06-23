#!/usr/bin/env bash
# Run/check shim over the Seq<T>-capable lean_single ELF, for gates that need
# Seq<T> (the K-AXI fusion witness + seq_* tests). Madaros is the default
# user-facing engine (bin/souc) but does not yet carry Seq<T> — restoring Seq<T>
# in the modular compiler is tracked as a follow-up. This shim keeps the
# dissertation gate on the Seq-capable engine without changing the default.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ELF="${SOUNIO_SEQ_LEANSINGLE_ELF:-$ROOT/bin/souc-linux-x86_64}"
verb="${1:-}"; shift || true
case "$verb" in
  run)     src="$1"; out="$(mktemp)"; "$ELF" "$src" "$out" >/dev/null 2>&1; chmod +x "$out"; "$out"; rc=$?; rm -f "$out"; exit $rc ;;
  check)   src="$1"; tmp="$(mktemp)"; "$ELF" "$src" "$tmp" >/dev/null 2>&1; rc=$?; rm -f "$tmp"; exit $rc ;;
  compile) src="$1"; cmp_out="/dev/null"; shift; while [[ $# -gt 0 ]]; do [[ "${1:-}" == "-o" ]] && { cmp_out="${2:-/dev/null}"; shift 2; } || shift; done; "$ELF" "$src" "$cmp_out" >/dev/null 2>&1; rc=$?; [[ $rc -eq 0 && "$cmp_out" != "/dev/null" ]] && chmod +x "$cmp_out"; exit $rc ;;
  *)       "$ELF" "$verb" "$@" ;;
esac
