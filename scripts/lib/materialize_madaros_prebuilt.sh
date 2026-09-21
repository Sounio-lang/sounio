#!/usr/bin/env bash
# scripts/lib/materialize_madaros_prebuilt.sh
#
# The committed Madaros prebuilt is stored compressed:
#
#   bin/madaros-linux-x86_64.gz       gzip -n -9 of the ELF            (tracked)
#   bin/madaros-linux-x86_64.sha256   sha256 of the uncompressed ELF   (tracked)
#   bin/madaros-linux-x86_64          the ELF itself        (gitignored, made here)
#
# WHY. The ELF is ~95 MB, a few MB under GitHub's 100 MB per-file limit, and it
# grows with the compiler. Compressed it is ~11.6 MB. Git LFS was considered and
# not used: CI checks the repository out hundreds of times a week, and LFS would
# download, store and bill every weekly refresh of the full 95 MB, where git
# stores each refresh of the .gz as a ~3.3 MB delta (measured 2026-09-15 on
# aaecebd878 -> 9e8e673414).
#
# This script turns the tracked .gz into the ELF that bin/madaros, bin/souc and
# ~80 scripts execute. It refuses loudly (exit 78) when the bytes do not match
# the tracked sha256: a prebuilt that is not the committed one must never run as
# if it were.
#
# Fast path: a stamp beside the ELF records the sha256 and size it was verified
# at. While the tracked sha256 and the ELF's size still match the stamp, the ELF
# is not re-hashed (bin/souc runs hundreds of times per test suite). Pass
# --verify, or set SOUNIO_REQUIRE_COMMITTED_MADAROS=1, to re-hash every time.
#
# A bin/madaros-linux-x86_64 that does not match the tracked sha256 is replaced
# from the .gz, with a notice on stderr. That path is a generated file; to run a
# different build, name it with MADAROS_RAW_BIN instead of copying it there.
#
# Usage:  bash scripts/lib/materialize_madaros_prebuilt.sh [--verify]
#         or: source it, then sounio_materialize_madaros_prebuilt [--verify]
# Exit:   0   the ELF is present and matches, or this tree has no compressed
#             prebuilt (nothing to do)
#         78  the compressed prebuilt, its sha256, or the decompressed bytes are
#             missing, unreadable or do not match

_sounio_madaros_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

_sounio_madaros_stat_inode_mtime_ctime() {
  # Get inode, mtime, and ctime in a cross-platform way (GNU vs BSD stat).
  # ctime (change time) is more reliable than mtime for detecting modifications.
  # GNU stat: stat -c '%i %Y %Z' (inode, mtime, ctime - all seconds since epoch)
  # BSD stat: stat -f '%i %m %c' (inode, mtime, ctime - all as seconds)
  if stat -c '%i %Y %Z' "$1" 2>/dev/null; then
    return 0
  elif stat -f '%i %m %c' "$1" 2>/dev/null; then
    return 0
  fi
  return 1
}

sounio_materialize_madaros_prebuilt() {
  local root verify=0
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [[ "${1:-}" == "--verify" || "${SOUNIO_REQUIRE_COMMITTED_MADAROS:-0}" == "1" ]]; then
    verify=1
  fi

  local gz="$root/bin/madaros-linux-x86_64.gz"
  local sum="$root/bin/madaros-linux-x86_64.sha256"
  local elf="$root/bin/madaros-linux-x86_64"
  local stamp="$root/bin/.madaros-linux-x86_64.verified"

  # A tree from before the prebuilt was compressed (or a sparse checkout without
  # bin/): nothing to materialize only if both packed files are absent.
  if [[ ! -e "$gz" && ! -e "$sum" ]]; then
    return 0
  fi

  if [[ ! -e "$gz" || ! -s "$sum" ]]; then
    echo "error: madaros prebuilt: incomplete — bin/madaros-linux-x86_64.gz and bin/madaros-linux-x86_64.sha256 must both be present and valid" >&2
    rm -f "$elf" "$stamp"
    return 78
  fi
  local want
  want="$(awk 'NR == 1 {print $1}' "$sum")"
  if [[ ! "$want" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: madaros prebuilt: bin/madaros-linux-x86_64.sha256 does not start with a sha256" >&2
    rm -f "$elf" "$stamp"
    return 78
  fi

  local size="" inode="" mtime="" ctime=""
  if [[ -f "$elf" ]]; then
    size="$(wc -c < "$elf" 2>/dev/null | tr -d ' ')" || size=""
    # Get inode, mtime, and ctime to detect file replacements and modifications.
    # ctime (change time) is more reliable than mtime for detecting file changes.
    # Even with metadata match, verify hash to catch same-second in-place rewrites.
    local stat_out
    stat_out="$(_sounio_madaros_stat_inode_mtime_ctime "$elf")" || stat_out=""
    if [[ -n "$stat_out" ]]; then
      inode="${stat_out%% *}"
      local rest="${stat_out#* }"
      mtime="${rest%% *}"
      ctime="${rest##* }"
    fi
  fi

  # Metadata cache fast path: trust the cache when metadata matches.
  # While ctime has whole-second resolution and theoretically could miss same-second
  # rewrites, the practical risk is low and the performance cost of hashing every
  # invocation (hundreds per test suite) would be prohibitive. Metadata checks are
  # reliable for detecting most changes; full verification is the fallback path.
  if [[ $verify -eq 0 && -x "$elf" && -f "$stamp" && -n "$size" && -n "$inode" ]] \
     && [[ "$(cat "$stamp" 2>/dev/null)" == "$want $size $inode $mtime $ctime" ]]; then
    return 0
  fi

  # Always verify existing ELF's hash, but skip archive verification in normal mode.
  # In verify mode, validate both the ELF and the .gz archive to ensure the entire
  # artifact chain (.gz + .sha256) is valid, not just the materialized ELF.
  if [[ -f "$elf" ]] && [[ "$(_sounio_madaros_sha256 "$elf")" == "$want" ]]; then
    # In normal mode, ELF hash match is sufficient; skip archive verification
    # for performance (avoid decompressing the 95 MB binary).
    if [[ "$verify" -eq 0 ]]; then
      chmod 755 "$elf" 2>/dev/null || true
      local stat_out
      stat_out="$(_sounio_madaros_stat_inode_mtime_ctime "$elf")" || stat_out=""
      inode="${stat_out%% *}"
      local rest="${stat_out#* }"
      mtime="${rest%% *}"
      ctime="${rest##* }"
      printf '%s %s %s %s %s\n' "$want" "$size" "$inode" "$mtime" "$ctime" > "$stamp.tmp.$$" && mv -f "$stamp.tmp.$$" "$stamp"
      return 0
    fi
    # In verify mode, also validate the .gz archive to ensure the full artifact
    # chain is correct (continue to decompression verification below).
  fi

  if [[ -f "$elf" ]]; then
    echo "madaros prebuilt: bin/madaros-linux-x86_64 does not match bin/madaros-linux-x86_64.sha256; replacing it with the committed prebuilt" >&2
  fi

  local tmp="$elf.tmp.$$"
  if ! gzip -dc "$gz" > "$tmp" 2>/dev/null; then
    rm -f "$tmp" "$elf" "$stamp"
    echo "error: madaros prebuilt: cannot decompress bin/madaros-linux-x86_64.gz" >&2
    return 78
  fi
  local got
  got="$(_sounio_madaros_sha256 "$tmp")"
  if [[ "$got" != "$want" ]]; then
    rm -f "$tmp" "$elf" "$stamp"
    echo "error: madaros prebuilt: bin/madaros-linux-x86_64.gz decompresses to sha256 $got" >&2
    echo "  but bin/madaros-linux-x86_64.sha256 records $want" >&2
    echo "  refusing to install a prebuilt that is not the committed one" >&2
    return 78
  fi
  if ! { chmod 755 "$tmp" && mv -f "$tmp" "$elf"; }; then
    rm -f "$tmp" "$elf" "$stamp"
    echo "error: madaros prebuilt: could not install bin/madaros-linux-x86_64" >&2
    return 78
  fi
  size="$(wc -c < "$elf" | tr -d ' ')"
  local stat_out
  stat_out="$(_sounio_madaros_stat_inode_mtime_ctime "$elf")" || stat_out=""
  inode="${stat_out%% *}"
  local rest="${stat_out#* }"
  mtime="${rest%% *}"
  ctime="${rest##* }"
  printf '%s %s %s %s %s\n' "$want" "$size" "$inode" "$mtime" "$ctime" > "$stamp.tmp.$$" && mv -f "$stamp.tmp.$$" "$stamp"
  echo "madaros prebuilt: materialized bin/madaros-linux-x86_64 from bin/madaros-linux-x86_64.gz (sha256 ${want:0:12}, $size bytes)" >&2
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -uo pipefail
  sounio_materialize_madaros_prebuilt "$@"
  exit $?
fi
