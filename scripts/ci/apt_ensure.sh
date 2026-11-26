#!/usr/bin/env bash
# apt_ensure.sh PKG... -- make sure the named Debian packages are installed,
# without letting one Ubuntu mirror decide whether CI is green.
#
# 1. Packages already installed are not fetched (the hosted ubuntu-24.04 image
#    ships g++, jq and libquadmath0); when nothing is missing, apt is not touched.
# 2. `apt-get update` is tried against a sequence of official Ubuntu mirrors.
#    A mirror caught mid-sync ("File has unexpected size ... Mirror sync in
#    progress?", run 45803197304 on 2026-11-26) fails only that attempt. The
#    Azure mirror is not in the list: selecting it failed every setup (#3547).
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
[[ "$(id -u)" -eq 0 && -z "${APT_ENSURE_FORCE_SUDO:-}" ]] || SUDO="sudo"
opts=(-o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::Retries=3 -o APT::Update::Error-Mode=any)
mirrors=(
  "https://archive.ubuntu.com/ubuntu/"
  "https://us.archive.ubuntu.com/ubuntu/"
  "http://mirrors.edge.kernel.org/ubuntu/"
  "https://archive.ubuntu.com/ubuntu/"
)
mirror_file="${APT_ENSURE_MIRROR_FILE:-/etc/apt/apt-mirrors.txt}"
sources_glob="${APT_ENSURE_SOURCES:-/etc/apt/sources.list /etc/apt/sources.list.d/*}"

# Rotation only means something where apt actually reads the mirror file
# through a `mirror+file:` source, as on the GitHub-hosted image. Elsewhere,
# self-hosted DARWIN runners included, the configured sources are retried
# unchanged, and the log says so. The original file is restored on exit, so
# a shared runner never keeps a mirror this script chose.
rotate=0
# shellcheck disable=SC2086
if [[ -f "$mirror_file" ]] && grep -qsF "mirror+file:$mirror_file" $sources_glob; then
  rotate=1
  saved="$(mktemp)"
  cp "$mirror_file" "$saved"
  trap 'cat "$saved" | $SUDO tee "$mirror_file" >/dev/null; rm -f "$saved"' EXIT
else
  echo "apt_ensure: apt sources do not read $mirror_file via mirror+file; retrying the configured sources unchanged"
fi

attempt=0
for m in "${mirrors[@]}"; do
  attempt=$((attempt + 1))
  if [[ $rotate -eq 1 ]]; then
    printf '%s\n' "$m" | $SUDO tee "$mirror_file" >/dev/null
    echo "apt_ensure: attempt $attempt via $m"
  else
    echo "apt_ensure: attempt $attempt via the configured sources"
  fi
  if timeout 300 $SUDO apt-get "${opts[@]}" update \
     && timeout 600 $SUDO apt-get "${opts[@]}" install -y --no-install-recommends "${missing[@]}"; then
    echo "apt_ensure: installed ${missing[*]} (attempt $attempt)"
    exit 0
  fi
  sleep $((attempt * ${APT_ENSURE_BACKOFF_SECS:-15}))
done
echo "apt_ensure: FAIL: could not install ${missing[*]} after ${#mirrors[@]} attempts" >&2
exit 1
