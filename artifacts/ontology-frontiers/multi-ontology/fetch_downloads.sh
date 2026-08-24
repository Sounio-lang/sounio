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
# CHECKSUM-PINNED ON A MOVING URL (uberon).  Its versionIRI names release 2026-06-19 and
# that dated URL is a 404 — `purl.obolibrary.org/obo/<ont>/releases/<date>/` redirects to
# the project's GitHub release tag, and `v2026-06-19` is not there.  But the undated purl
# serves that very release: the bytes it returns today are identical to the copy that was
# committed, same sha256, same versionIRI.  So uberon is fetched from the undated URL and
# the CHECKSUM is the authority, not the URL.
#
# That means a mismatch on uberon is not corruption — it is upstream having published a
# NEW release.  Do not just update the digest: adopting a new uberon changes the generated
# uberon_* artifacts, and the numbers in RESULTS.md and the technical note were measured
# against 2026-06-19.  Decide, then regenerate, then update the docs.
#
# PATO moved into PINNED (2026-08-24).  Its versionIRI names release 2025-05-14, that URL
# returns 200, and regenerating from it reproduces the committed pato_* artifacts BYTE FOR
# BYTE in 2 s (gen_chebi_data.py --only pato).  So it is fetched and checksummed like cl.
#
# UNPINNED (chebi only), and MEASURED to be unrecoverable — 2026-08-24.  Its versionIRI
# names release 254, obo/chebi/254/chebi.owl is a 404, and the undated purl serves EBI's
# current file, which turns over each release.  The question of whether "current" happens
# to be the input behind the committed chebi_* artifacts was settled by running it:
#
#   downloaded chebi.owl   865,772,908 bytes, sha256 4557df5b6683...,
#                          versionIRI .../chebi/254/chebi.owl
#   regenerated (76 s)     chebi_classes.tsv, chebi_elplus_tbox.txt, chebi_packed.txt
#                          ALL THREE DIFFER from the committed copies
#   classes                218,254 committed vs 218,421 from release 254  (+167)
#   packed header          role-edge counts move ~10%, e.g. 29,846,298 -> 26,844,129
#                          and 22,643,261 -> 19,631,739
#
# So the committed chebi_* artifacts came from an OLDER, UNRECORDED release, and release
# 254 is not a substitute: the derived closure counts are what the drivers report and what
# RESULTS.md and the technical note quote.  Nothing in this repository names the release
# used, and EBI does not serve old ones at a stable URL.
#
# CONSEQUENCE: chebi.owl cannot be pinned, and the chebi_* artifacts must stay committed —
# they are the only surviving record of that input.  Deleting them loses information.
# Adopting 254 is possible but is a measurement change, not a cleanup: regenerate, then
# update every number quoted from them.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$DIR/downloads"

# name | url | sha256 ("-" = unpinned)
# Dated release URLs; the digest confirms the download.
PINNED=(
  "cl|http://purl.obolibrary.org/obo/cl/releases/2026-06-08/cl.owl|6abe12f1569d077507e03c1ad0168ebbb9ed725973a7eddba8ab3b9aeaf7a68d"
  "ro|http://purl.obolibrary.org/obo/ro/releases/2025-12-17/ro.owl|a9f644d4a865747e0b4aba7ca3f19aac1e0b072cab89e24a2e476df3abb10aaf"
  "pato|http://purl.obolibrary.org/obo/pato/releases/2025-05-14/pato.owl|73a80487130a81a3696f1e03c551288f741ed1be5a07639e69e7ecd8b6f0371c"
)
# UNDATED url — the digest, not the url, pins the version (release 2026-06-19).
# A mismatch here means upstream released a new uberon; see the header before touching it.
MOVING=(
  "uberon|http://purl.obolibrary.org/obo/uberon.owl|938f51e7c3fc9fcbe5a2863eb346da8033737e568af5836958891c4c6bfb1192"
)
UNPINNED=(
  "chebi|http://purl.obolibrary.org/obo/chebi.owl|-"
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
    printf '           Either the download was truncated, or upstream published a new release.\n           For an entry whose url carries no release date, the second is the likely one.\n           Do not just update the digest: regenerate the artifacts and check the numbers\n           in RESULTS.md and the technical note, which were measured against the old one.\n' >&2
    rc=1; return 0
  fi
  mv "$tmp" "$file"
  printf '  %-7s fetched and verified\n' "$name"
}

mkdir -p "$DEST"
echo "ontology downloads -> $DEST"
for e in "${PINNED[@]}"; do process "$e"; done
for e in "${MOVING[@]}"; do process "$e"; done
if [[ "$WANT_UNPINNED" == 1 ]]; then
  for e in "${UNPINNED[@]}"; do process "$e"; done
else
  printf '  %-7s skipped (unpinned; pass --unpinned)\n' "chebi"
fi

if [[ $rc -ne 0 ]]; then
  echo "FAILED" >&2
else
  echo "OK"
fi
exit $rc
