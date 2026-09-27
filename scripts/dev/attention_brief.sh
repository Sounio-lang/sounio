#!/usr/bin/env bash
# Attention brief — shepherd ritual for 5=1+2 governance.
# Usage:
#   bash scripts/dev/attention_brief.sh
#   bash scripts/dev/attention_brief.sh --freeze
#   bash scripts/dev/attention_brief.sh --prune

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FREEZE=0
PRUNE=0
for arg in "$@"; do
  case "$arg" in
    --freeze) FREEZE=1 ;;
    --prune) PRUNE=1 ;;
    -h|--help)
      sed -n '2,7p' "$0"
      exit 0
      ;;
    *)
      printf 'unknown option: %s\n' "$arg" >&2
      exit 2
      ;;
  esac
done

COORD="$ROOT/bin/sounio-coord"
P0="$ROOT/.claude/attention_p0.v1.json"
CHARTER="$ROOT/.claude/ATTENTION_CHARTER.md"

printf '== Attention Brief (5 = 1 + 2) ==\n'
printf 'utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'root: %s\n' "$ROOT"
printf 'charter: %s\n' "$CHARTER"

if [[ -f "$P0" ]]; then
  python3 - <<'PY' "$P0"
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
print(f"equation: {d.get('equation')}")
print(f"active_p0: {d.get('active_p0')}")
print(f"updated_utc: {d.get('updated_utc')}")
print("-- slots --")
for s in d.get("slots", []):
    print(
        f"  {s.get('id')} h={s.get('horizon')} status={s.get('status')} "
        f"owner={s.get('owner')} :: {s.get('title')}"
    )
for n in d.get("notes", []):
    print(f"note: {n}")
PY
else
  printf 'WARNING: missing %s\n' "$P0"
fi

printf '\n== Coordination ==\n'
"$COORD" brief

if ((PRUNE)); then
  printf '\n== Prune expired claims ==\n'
  "$COORD" prune
fi

if ((FREEZE)); then
  printf '\n== Broadcast freeze ==\n'
  "$COORD" send \
    --agent shepherd \
    --lane attention \
    --kind info \
    --message "FREEZE: /workspace/sounio is control-only under ATTENTION_CHARTER (5=1+2). No new write claims on the control worktree except shepherd + declared active_p0. Heavy 1/2 work must use a dedicated claimed worktree. Read .claude/ATTENTION_CHARTER.md and .claude/attention_p0.v1.json. Ack after reading."
  printf 'freeze message broadcast.\n'
fi

printf '\n== Next ==\n'
printf '1. Confirm active_p0 owner (or assign one).\n'
printf '2. Refuse Garden writes on the control worktree.\n'
printf '3. Agents: MCP sounio-coord or bin/sounio-coord send/inbox/ack.\n'
