#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${OMEGA_NO_RUST_SURFACE_OUT:-artifacts/omega/no_rust_execution_surface.v1.json}"
STRICT="${OMEGA_NO_RUST_SURFACE_STRICT:-1}"
REQUIRE_RUNTIME_ABSENCE="${OMEGA_NO_RUST_REQUIRE_RUNTIME_ABSENCE:-$STRICT}"

TARGETS=(
  "scripts/omega_sprint1_gate.sh"
  "scripts/selfhost/selfhost_independence_gate.sh"
  "scripts/selfhost/selfhost_cycle_gate.sh"
  "scripts/selfhost/selfhost_zero_fallback_gate.sh"
  "scripts/selfhost_driver_output_gate.sh"
  "scripts/selfhost/selfhost_driver_output_parity_gate.sh"
  "scripts/bootstrap/bootstrap_kernel_gate.sh"
  "scripts/e2e_gate.sh"
  "scripts/full_gate.sh"
  ".github/workflows/ci.yml"
  ".github/workflows/release-gate.yml"
)

mkdir -p "$(dirname "$OUT_PATH")"

violations=()
for path in "${TARGETS[@]}"; do
  if [ ! -f "$path" ]; then
    if [ "$STRICT" = "1" ]; then
      violations+=("$path:missing")
    fi
    continue
  fi

  while IFS= read -r line; do
    violations+=("$path:$line")
  done < <(
    python3 - "$path" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
cmd_re = re.compile(r"(^|[;&(|\s])(cargo|rustc)(\s|$)")

for idx, raw in enumerate(path.read_text().splitlines(), start=1):
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if cmd_re.search(raw):
        print(f"{idx}:{raw.strip()}")
PY
  )
done

runtime_tools=()
runtime_violations=()
for tool in cargo rustc; do
  if tool_path="$(command -v "$tool" 2>/dev/null)"; then
    runtime_tools+=("$tool:$tool_path")
    if [ "$REQUIRE_RUNTIME_ABSENCE" = "1" ]; then
      runtime_violations+=("runtime:$tool:$tool_path")
    fi
  else
    runtime_tools+=("$tool:absent")
  fi
done
for violation in "${runtime_violations[@]}"; do
  violations+=("$violation")
done

status="pass"
if [ "${#violations[@]}" -gt 0 ]; then
  status="fail"
fi

RUNTIME_TOOLS_JOINED="$(printf '%s\n' "${runtime_tools[@]}")"
VIOLATIONS_JOINED="$(printf '%s\n' "${violations[@]}")"
RUNTIME_TOOLS="$RUNTIME_TOOLS_JOINED" \
VIOLATIONS="$VIOLATIONS_JOINED" \
python3 - "$OUT_PATH" "$status" "$STRICT" "$REQUIRE_RUNTIME_ABSENCE" "${#TARGETS[@]}" "${#runtime_violations[@]}" "${#violations[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
status = sys.argv[2]
strict = sys.argv[3] == "1"
require_runtime_absence = sys.argv[4] == "1"
target_count = int(sys.argv[5])
runtime_viol_count = int(sys.argv[6])
viol_count = int(sys.argv[7])
runtime_tools = [line for line in os.environ.get("RUNTIME_TOOLS", "").splitlines() if line]
violations = [line for line in os.environ.get("VIOLATIONS", "").splitlines() if line]

payload = {
    "schema": "sounio.omega.no-rust-execution-surface.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": status,
    "strict": strict,
    "require_runtime_absence": require_runtime_absence,
    "runtime_tools": runtime_tools,
    "runtime_violation_count": runtime_viol_count,
    "target_count": target_count,
    "violation_count": viol_count,
    "violations": violations,
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY

if [ "$status" = "pass" ]; then
  echo "omega_no_rust_execution_surface: status=pass targets=${#TARGETS[@]} report=$OUT_PATH"
  exit 0
fi

echo "omega_no_rust_execution_surface: status=fail violations=${#violations[@]} report=$OUT_PATH" >&2
for v in "${violations[@]}"; do
  echo "error: $v" >&2
done
exit 2
