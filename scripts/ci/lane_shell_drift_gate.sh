#!/usr/bin/env bash
# The fleet launcher lives at $HOME/bin/sounio-lane-shell and is NOT tracked by
# git. Twenty-three lanes start through it, and a bad edit there is invisible to
# every review in this repository.
#
# scripts/dev/sounio-lane-shell.reference is a committed copy. This gate says
# whether the live file still matches it -- so a change is either reviewed here
# or reported as drift, rather than landing unseen.
#
# It does NOT fail on drift. The live file is legitimately edited on the pod
# and the repo copy is the record, not the authority. Silence would be the
# failure; an unexplained difference is the finding.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "lane_shell_drift"

REF="$ROOT_DIR/scripts/dev/sounio-lane-shell.reference"
LIVE="${SOUNIO_LANE_SHELL:-/workspace/.home/openvscode-server/bin/sounio-lane-shell}"

require_nonempty_file "$REF" "the committed reference copy is missing"

# gate_pass PRINTS, it does not exit. Writing `if ok; then gate_pass; fi` and
# carrying on falls straight through into the failure branch -- this gate
# reported drift between two byte-identical files on its first run because of
# exactly that. Every success path here exits explicitly.
if [[ ! -f "$LIVE" ]]; then
  echo "  live launcher not present at $LIVE (not this machine) -- reference only"
  gate_pass "reference copy present; no live file to compare on this host"
  exit 0
fi

ref_sha="$(sha256sum "$REF" | awk '{print $1}')"
live_sha="$(sha256sum "$LIVE" | awk '{print $1}')"

if [[ "$ref_sha" == "$live_sha" ]]; then
  gate_pass "live launcher matches the committed reference ($ref_sha)"
  exit 0
fi

echo "  DRIFT: the live launcher differs from the committed reference."
echo "    reference $ref_sha"
echo "    live      $live_sha"
echo
diff -u "$REF" "$LIVE" | head -60 | sed 's/^/    /'
echo
echo "  If the live change is wanted, refresh the copy and say why in the commit:"
echo "    cp $LIVE $REF"
gate_pass "drift reported (not a failure: the live file is legitimately edited on the pod)"
exit 0
