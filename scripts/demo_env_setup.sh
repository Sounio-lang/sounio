#!/usr/bin/env bash
# scripts/demo_env_setup.sh -- make a bare Ubuntu 24.04 able to run the tour.
#
# Used by .devcontainer/demo (GitHub Codespaces: "New with options" ->
# configuration "sounio-demo"). Safe to re-run. After it finishes:
#
#   bash scripts/tour.sh                 # 13 claims, about 35 s
#   bash scripts/tour.sh --full --lean   # all 20, about 4-5 min
#
# Installs only what the tour needs: a C toolchain for the bootstrap
# (make build), elan with the toolchain pinned in formal/lean-toolchain, and
# the committed Madaros prebuilt (sha256-checked).
set -euo pipefail
cd "$(dirname "$0")/.."

need_apt=()
command -v cc   >/dev/null || need_apt+=(build-essential)
command -v make >/dev/null || need_apt+=(make)
command -v curl >/dev/null || need_apt+=(curl ca-certificates)
if [ ${#need_apt[@]} -gt 0 ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${need_apt[@]}"
fi

if ! command -v lake >/dev/null && [ ! -x "$HOME/.elan/bin/lake" ]; then
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none --no-modify-path
fi
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && elan toolchain install "$(cat lean-toolchain)" >/dev/null 2>&1 || true)
(cd formal && lake --version)

bash scripts/lib/materialize_madaros_prebuilt.sh
./bin/souc --version

echo
echo "sounio demo environment ready. Next: bash scripts/tour.sh"
