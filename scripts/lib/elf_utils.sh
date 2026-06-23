#!/usr/bin/env bash
# Sourceable ELF helpers shared by install scripts and gates.

if [[ -n "${_SOUNIO_ELF_UTILS_LOADED:-}" ]]; then
  return 0
fi
_SOUNIO_ELF_UTILS_LOADED=1

sounio_is_elf_binary() {
  [[ -n "${1:-}" && -x "$1" ]] || return 1
  [[ "$(LC_ALL=C head -c 4 "$1" 2>/dev/null | od -An -tx1 | tr -d ' \n')" == "7f454c46" ]]
}
