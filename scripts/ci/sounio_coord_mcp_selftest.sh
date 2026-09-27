#!/usr/bin/env bash
# Smoke: Attention Charter artifacts + sounio-coord MCP import + brief tool path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

[[ -f .claude/ATTENTION_CHARTER.md ]] || fail "missing ATTENTION_CHARTER.md"
[[ -f .claude/attention_p0.v1.json ]] || fail "missing attention_p0.v1.json"
[[ -f scripts/mcp/sounio_coord_mcp.py ]] || fail "missing sounio_coord_mcp.py"
[[ -x bin/sounio-coord ]] || fail "bin/sounio-coord not executable"
[[ -x scripts/dev/attention_brief.sh ]] || fail "attention_brief.sh not executable"

python3 - <<'PY'
import json
from pathlib import Path
p = Path(".claude/attention_p0.v1.json")
d = json.loads(p.read_text(encoding="utf-8"))
assert d.get("schema") == "sounio.attention_p0.v1", d.get("schema")
assert d.get("equation") == "5 = 1 + 2", d.get("equation")
assert "slots" in d and isinstance(d["slots"], list) and d["slots"]
ids = {s["id"] for s in d["slots"]}
assert {"A", "B", "C", "D", "E"} <= ids, ids
print("attention_p0.v1.json OK")
PY

PY="${ROOT}/.venv/bin/python3"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" - <<'PY'
import importlib.util
from pathlib import Path
path = Path("scripts/mcp/sounio_coord_mcp.py")
spec = importlib.util.spec_from_file_location("sounio_coord_mcp", path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
try:
    spec.loader.exec_module(mod)
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("mcp"):
        print("WARN: mcp package not installed; skipped import (CLI path still valid)")
    else:
        raise
else:
    assert hasattr(mod, "attention_p0")
    assert hasattr(mod, "coord_brief")
    assert hasattr(mod, "coord_send")
    assert hasattr(mod, "coord_inbox")
    print("sounio_coord_mcp import OK")
PY

bin/sounio-coord brief >/tmp/sounio_coord_mcp_brief.out
grep -q 'Sounio coordination status\|Claims' /tmp/sounio_coord_mcp_brief.out \
  || fail "coord brief unexpected output"

# attention_brief must not require freeze
bash scripts/dev/attention_brief.sh >/tmp/attention_brief.out
grep -q 'Attention Brief' /tmp/attention_brief.out || fail "attention_brief header missing"
grep -q 'active_p0' /tmp/attention_brief.out || fail "attention_brief missing active_p0"

# .mcp.json must register sounio-coord
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path(".mcp.json").read_text(encoding="utf-8"))
servers = cfg.get("mcpServers") or {}
assert "sounio-coord" in servers, sorted(servers)
args = servers["sounio-coord"].get("args") or []
assert any("sounio_coord_mcp.py" in str(a) for a in args), args
print(".mcp.json sounio-coord OK")
PY

printf 'SOUNIO_COORD_MCP_SELFTEST_OK\n'
