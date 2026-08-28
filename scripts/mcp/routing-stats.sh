#!/usr/bin/env bash
# routing-stats.sh — Weekly routing log summary
#
# Usage: bash scripts/mcp/routing-stats.sh [--days N]
#   --days N   analyse last N days (default: 7)
#
# Reads: ~/.claude/routing.log (JSONL)
# Outputs: calls per tool, estimated token savings vs Opus

set -euo pipefail

DAYS="${2:-7}"
LOG_FILE="${HOME}/.claude/routing.log"

if [[ "${1:-}" == "--days" ]]; then
  DAYS="${2:-7}"
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "No routing log found at $LOG_FILE"
  echo "Have you made any darwin_* or triangle_* tool calls yet?"
  exit 0
fi

echo "=== Routing stats (last ${DAYS} days) ==="
echo ""

# Filter to the last N days using Python for portability
python3 - "$LOG_FILE" "$DAYS" <<'PYEOF'
import json, sys, datetime

log_path = sys.argv[1]
days = int(sys.argv[2])
cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)

from collections import defaultdict
calls = defaultdict(int)
tokens_in = defaultdict(int)
tokens_out = defaultdict(int)

with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            ts = datetime.datetime.fromisoformat(r["ts"].rstrip("Z"))
            if ts < cutoff:
                continue
            tool = r.get("tool", "unknown")
            calls[tool] += 1
            tokens_in[tool] += r.get("tokens_in", 0)
            tokens_out[tool] += r.get("tokens_out", 0)
        except Exception:
            continue

if not calls:
    print("No routing events in the last", days, "days.")
    sys.exit(0)

# Opus cost estimate: ~$15/M input + $75/M output (claude-opus-4.x approximate)
# These calls avoided Opus entirely
OPUS_INPUT_COST  = 15.0 / 1_000_000
OPUS_OUTPUT_COST = 75.0 / 1_000_000

print(f"{'Tool':<22} {'Calls':>6}  {'Tokens in':>10}  {'Tokens out':>11}  {'Avoided cost ~':>14}")
print("-" * 72)
total_saved = 0.0
for tool in sorted(calls):
    ti = tokens_in[tool]
    to = tokens_out[tool]
    saved = ti * OPUS_INPUT_COST + to * OPUS_OUTPUT_COST
    total_saved += saved
    print(f"{tool:<22} {calls[tool]:>6}  {ti:>10,}  {to:>11,}  ${saved:>13.4f}")
print("-" * 72)
print(f"{'TOTAL SAVED (vs Opus)':<22} {'':>6}  {'':>10}  {'':>11}  ${total_saved:>13.4f}")
PYEOF
