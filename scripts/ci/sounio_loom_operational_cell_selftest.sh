#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
fail() {
  printf 'sounio-loom-operational-cell-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

bash "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_routing_authority_selftest.sh"
bash "$ROOT_DIR/scripts/ci/sounio_loom_message_bridge_selftest.sh"

case "$(uname -s)" in
  Darwin)
    bash "$ROOT_DIR/tools/loom/apple/validate-apple.sh"
    apple_gate=PASS
    ;;
  *)
    domain="$ROOT_DIR/tools/loom/apple/Sources/LoomDomain/LoomFleetAPI.swift"
    store="$ROOT_DIR/tools/loom/apple/Sources/LoomSpatial/LoomStore.swift"
    panel="$ROOT_DIR/tools/loom/apple/Sources/LoomSpatial/OperationalPanels.swift"
    grep -Fq 'func routeOperation(taskID:' "$domain" ||
      fail 'Apple client lacks task lifecycle polling'
    grep -Fq 'func cancelRoute(taskID:' "$domain" ||
      fail 'Apple client lacks route cancellation'
    grep -Fq 'case .completed, .committed:' "$panel" ||
      fail 'Apple UI lacks the completed route state'
    grep -Fq 'case .cancelled:' "$panel" ||
      fail 'Apple UI lacks the cancelled route state'
    grep -Fq 'messageClient.routeOperation(taskID:' "$store" ||
      fail 'Apple store does not poll the selected route lifecycle'
    apple_gate=SOURCE_CONTRACT_ONLY
    ;;
esac

printf 'sounio-loom-operational-cell-selftest: PASS core=OCaml authority=Sounio-action-9032 routing=concurrent lifecycle=terminal cancellation=idempotent apple=%s\n' \
  "$apple_gate"
