#!/usr/bin/env bash
# apt_ensure.sh PKG... -- make sure the named Debian packages are installed,
# without letting one Ubuntu mirror decide whether CI is green.
#
# 1. Packages already installed are not fetched (the hosted ubuntu-24.04 image
#    ships g++, jq and libquadmath0); when nothing is missing, apt is not touched.
# 2. `apt-get update` is tried against a sequence of official Ubuntu mirrors.
#    A mirror caught mid-sync ("File has unexpected size ... Mirror sync in
#    progress?", run 45803197304 on 2026-11-26) fails only that attempt.
# 3. The install is retried the same way. A package that is genuinely missing
#    from every mirror still fails the step: this script never hides a real
#    absence, it only removes one mirror as a single point of failure.
set -uo pipefail

[[ $# -gt 0 ]] || { echo "usage: $0 PKG..." >&2; exit 2; }

missing=()
for p in "$@"; do
  dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
done
if [[ ${#missing[@]} -eq 0 ]]; then
  echo "apt_ensure: already installed: $*"
  exit 0
fi
echo "apt_ensure: missing: ${missing[*]}"

SUDO=""
[[ "$(id -u)" -eq 0 ]] || SUDO="sudo"
opts=(-o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::Retries=2 -o APT::Update::Error-Mode=any)
mirrors=(
  "https://archive.ubuntu.com/ubuntu/"
  "http://azure.archive.ubuntu.com/ubuntu/"
  "http://mirrors.edge.kernel.org/ubuntu/"
  "https://archive.ubuntu.com/ubuntu/"
)
mirror_file=/etc/apt/apt-mirrors.txt

attempt=0
for m in "${mirrors[@]}"; do
  attempt=$((attempt + 1))
  if [[ -f "$mirror_file" ]]; then
    printf '%s\n' "$m" | $SUDO tee "$mirror_file" >/dev/null
  fi
  echo "apt_ensure: attempt $attempt via $m"
  if timeout 300 $SUDO apt-get "${opts[@]}" update \
     && timeout 600 $SUDO apt-get "${opts[@]}" install -y --no-install-recommends "${missing[@]}"; then
    echo "apt_ensure: installed ${missing[*]} (attempt $attempt)"
    exit 0
  fi
  sleep $((attempt * 15))
done
echo "apt_ensure: FAIL: could not install ${missing[*]} from ${#mirrors[@]} mirror attempts" >&2
exit 1
