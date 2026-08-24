#!/usr/bin/env bash
# Fetch the upstream ontologies that gen_multi_data.py and gen_chebi_data.py read
# from ./downloads/.  Run from artifacts/ontology-frontiers/multi-ontology/.
#
#   ./fetch_downloads.sh            fetch whatever is missing or corrupt (pinned only)
#   ./fetch_downloads.sh --check    verify what is present; fetch nothing
#   ./fetch_downloads.sh --unpinned also fetch chebi and pato (see the warning below)
#
# WHY THIS EXISTS.  Before this script the three .owl inputs were committed — 163 MB for
# cl + uberon + ro — while chebi.owl and pato.owl, whose generated outputs ARE committed
# (1,063,089 lines), were not present at all.  The repository stored the expensive,
# regenerable half and omitted the half that cannot be regenerated without it.
#
# THREE CATEGORIES, because the five ontologies are genuinely not alike.
#
# PINNED (cl, ro).  Each carries an owl:versionIRI naming a dated release, that release
# URL resolves, and the download was verified byte-identical to the copy that used to be
# committed.  These are fetched and checksummed.
#
# COMMITTED-ONLY (uberon).  Its versionIRI names release 2026-06-19, and that URL is a
# 404 today: `purl.obolibrary.org/obo/<ont>/releases/<date>/` redirects to the project's
# GitHub release tag, and `v2026-06-19` no longer exists in obophenotype/uberon.  The
# exact input that produced the committed uberon_* artifacts is therefore NOT retrievable
# from upstream, so uberon.owl must stay in the tree.  This script verifies its checksum
# and never tries to fetch it.  Re-pinning it to a current release is possible but is a
# real decision: the regenerated artifacts would no longer match the committed ones.
#
# UNPINNED (chebi, pato).  No release is recorded anywhere in this repository, and
# `purl.obolibrary.org` redirects both to MOVING targets (EBI's current chebi.owl;
# pato's master branch on GitHub).  Skipped unless you pass --unpinned, and what you get
# is whatever upstream publishes today — NOT known to be the version the committed
# chebi_*/pato_* artifacts came from.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$DIR/downloads"

# name | url | sha256 ("-" = unpinned)
PINNED=(
  "cl|http://purl.obolibrary.org/obo/cl/releases/2026-06-08/cl.owl|6abe12f1569d077507e03c1ad0168ebbb9ed725973a7eddba8ab3b9aeaf7a68d"
  "ro|http://purl.obolibrary.org/obo/ro/releases/2025-12-17/ro.owl|a9f644d4a865747e0b4aba7ca3f19aac1e0b072cab89e24a2e476df3abb10aaf"
)
# Verified if present, never fetched — its release is gone from upstream (see the header).
COMMITTED=(
  "uberon|http://purl.obolibrary.org/obo/uberon/releases/2026-06-19/uberon.owl|938f51e7c3fc9fcbe5a2863eb346da8033737e568af5836958891c4c6bfb1192"
)
UNPINNED=(
  "chebi|http://purl.obolibrary.org/obo/chebi.owl|-"
  "pato|http://purl.obolibrary.org/obo/pato.owl|-"
)

MODE=fetch
WANT_UNPINNED=0
for arg in "$@"; do
  case "$arg" in
    --check)    MODE=check ;;
    --unpinned) WANT_UNPINNED=1 ;;
    -h|--help)  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

digest() { sha256sum "$1" | cut -d' ' -f1; }

rc=0
process() {
  local name url want file got tmp
  IFS='|' read -r name url want <<<"$1"
  file="$DEST/$name.owl"

  if [[ -f "$file" ]]; then
    if [[ "$want" == "-" ]]; then
      printf '  %-7s present, unpinned    sha256 %s\n' "$name" "$(digest "$file")"
      return 0
    fi
    got="$(digest "$file")"
    if [[ "$got" == "$want" ]]; then
      printf '  %-7s ok\n' "$name"
      return 0
    fi
    printf '  %-7s SHA MISMATCH\n           expected %s\n           actual   %s\n' \
      "$name" "$want" "$got" >&2
    if [[ "$MODE" == check ]]; then rc=1; return 0; fi
    printf '           refetching\n' >&2
  else
    if [[ "$MODE" == check ]]; then
      printf '  %-7s MISSING\n' "$name" >&2; rc=1; return 0
    fi
  fi

  printf '  %-7s fetching %s\n' "$name" "$url"
  tmp="$(mktemp "$DEST/.$name.XXXXXX")"
  if ! curl -fsSL --retry 3 --retry-delay 2 -o "$tmp" "$url"; then
    rm -f "$tmp"
    printf '  %-7s DOWNLOAD FAILED — %s is unreachable. Check the network, or the release\n           URL if upstream has retired it.\n' "$name" "$url" >&2
    rc=1; return 0
  fi
  got="$(digest "$tmp")"
  if [[ "$want" == "-" ]]; then
    mv "$tmp" "$file"
    printf '  %-7s fetched, UNPINNED    sha256 %s\n' "$name" "$got"
    printf '           To pin it, move this entry into PINNED with that sha256 — but first\n'
    printf '           confirm it is the version the committed %s_* artifacts came from.\n' "$name"
    return 0
  fi
  if [[ "$got" != "$want" ]]; then
    rm -f "$tmp"
    printf '  %-7s SHA MISMATCH after download\n           expected %s\n           actual   %s\n' \
      "$name" "$want" "$got" >&2
    printf '           Upstream changed this release, or the download was truncated.\n           Do not pin the new digest without checking that the generated artifacts\n           still reproduce.\n' >&2
    rc=1; return 0
  fi
  mv "$tmp" "$file"
  printf '  %-7s fetched and verified\n' "$name"
}

committed_only() {   # verify, never fetch
  local name url want file got
  IFS='|' read -r name url want <<<"$1"
  file="$DEST/$name.owl"
  if [[ ! -f "$file" ]]; then
    printf '  %-7s MISSING and NOT FETCHABLE\n' "$name" >&2
    printf '           %s is a 404: that release tag no longer exists upstream.\n' "$url" >&2
    printf '           This file must be restored from git history — it is the one input\n' >&2
    printf '           that cannot be recovered from the network.\n' >&2
    rc=1; return 0
  fi
  got="$(digest "$file")"
  if [[ "$got" == "$want" ]]; then
    printf '  %-7s ok (committed; upstream release retired)\n' "$name"
  else
    printf '  %-7s SHA MISMATCH\n           expected %s\n           actual   %s\n' "$name" "$want" "$got" >&2
    printf '           Cannot refetch: the pinned release is a 404 upstream.\n' >&2
    rc=1
  fi
}

mkdir -p "$DEST"
echo "ontology downloads -> $DEST"
for e in "${PINNED[@]}"; do process "$e"; done
for e in "${COMMITTED[@]}"; do committed_only "$e"; done
if [[ "$WANT_UNPINNED" == 1 ]]; then
  for e in "${UNPINNED[@]}"; do process "$e"; done
else
  printf '  %-7s skipped (unpinned; pass --unpinned)\n' "chebi"
  printf '  %-7s skipped (unpinned; pass --unpinned)\n' "pato"
fi

if [[ $rc -ne 0 ]]; then
  echo "FAILED" >&2
else
  echo "OK"
fi
exit $rc
