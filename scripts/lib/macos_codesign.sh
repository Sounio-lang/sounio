#!/usr/bin/env bash

sounio_host_os() {
  printf '%s\n' "${SOUNIO_HOST_OS_OVERRIDE:-$(uname -s 2>/dev/null || echo unknown)}"
}

sounio_host_is_darwin() {
  [[ "$(sounio_host_os)" == "Darwin" ]]
}

sounio_ad_hoc_codesign() {
  local path="${1:-}"
  if [[ -z "$path" || ! -e "$path" ]]; then
    return 0
  fi

  if ! sounio_host_is_darwin; then
    return 0
  fi

  if ! command -v codesign >/dev/null 2>&1; then
    return 0
  fi

  chmod +x "$path" 2>/dev/null || true
  if ! codesign --force --sign - "$path" >/dev/null 2>&1; then
    if [[ "${SOUNIO_CODESIGN_STRICT:-0}" == "1" ]]; then
      return 1
    fi
    if [[ "${SOUNIO_SOUC_VERBOSE:-0}" == "1" ]]; then
      printf 'warning: ad hoc codesign failed for %s\n' "$path" >&2
    fi
  fi
  return 0
}
