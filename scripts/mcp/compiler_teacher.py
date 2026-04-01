#!/usr/bin/env python3
"""
Compiler as Teacher — Sounio educational feedback service.

POST /check  { "code": "<sounio source>" }
→ { "ok": bool, "errors": [...], "feedback": "...", "fixed": "..." }

Also logs each interaction to datasets/sounio-contrastive/teacher_log.jsonl
for use as fine-tuning data.

Usage:
    python3 scripts/mcp/compiler_teacher.py [--port 8765] [--host 127.0.0.1]
"""
import os
import sys
import json
import time
import tempfile
import subprocess
import argparse
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_SCRIPT = REPO_ROOT / "scripts/dev/sounio-lint.py"
SOUC = REPO_ROOT / "bin/souc"
LOG_PATH = REPO_ROOT / "datasets/sounio-contrastive/teacher_log.jsonl"

# ---------------------------------------------------------------------------
# Error explanation catalog — maps lint rule → human-readable explanation
# ---------------------------------------------------------------------------
RULE_EXPLANATIONS = {
    "no_semicolons": (
        "Sounio does not use semicolons as statement terminators. "
        "Remove the trailing `;` from each line. "
        "Example: `let x = 5` not `let x = 5;`"
    ),
    "var_not_let_mut": (
        "Sounio uses `var` for mutable bindings, not Rust's `let mut`. "
        "Replace `let mut x` with `var x`."
    ),
    "exclusive_ref": (
        "Sounio uses `&!T` for exclusive (mutable) references, not Rust's `&mut T`. "
        "Replace `&mut x` with `&!x`."
    ),
    "no_macros": (
        "Sounio has no macro invocation syntax. "
        "Use `assert(cond)` not `assert!(cond)`, and `println(msg)` not `println!(msg)`. "
        "Remove the `!` from all macro-style calls."
    ),
    "no_unary_minus": (
        "Sounio does not support unary minus literals. "
        "Write `0 - 42` instead of `-42`. "
        "For float: `0.0 - 3.14` instead of `-3.14`."
    ),
    "effects_mandatory": (
        "Sounio functions must declare all effects they use. "
        "`fn main()` must be `fn main() with IO` if it does any I/O. "
        "Effects: IO, Mut, Div, Panic, Observe."
    ),
    "no_generics": (
        "Sounio does not support Rust-style generics like `Vec<T>` or `HashMap<K,V>`. "
        "Use fixed-size arrays `[T; N]` or define concrete types."
    ),
    "no_closures": (
        "Sounio does not support closure literals like `|x| x + 1`. "
        "Define a named function instead: `fn square(x: i64) -> i64 { x * x }` "
        "then use `let f = square`."
    ),
    "no_attributes": (
        "Sounio has no `#[attribute]` syntax. "
        "Remove `#[test]`, `#[derive(...)]`, and similar attributes."
    ),
    "compound_assign": (
        "Sounio does not support compound assignment operators like `+=`, `-=`. "
        "Use `x = x + 1` instead of `x += 1`."
    ),
    "shift_operand": (
        "Sounio requires shift amounts to be `u8`. "
        "Use `x << (n as u8)` or use a `u8` literal: `x << 1u8`."
    ),
}


def run_lint(code: str) -> list[dict]:
    """Run sounio-lint.py on code, return list of diagnostic dicts."""
    with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, str(LINT_SCRIPT), "--json", tmp],
            capture_output=True, text=True, timeout=10
        )
        out = (result.stdout or "").strip()
        if out:
            parsed = json.loads(out)
            # JSON output is {"file":..., "diagnostics": [...]}
            if isinstance(parsed, dict):
                return parsed.get("diagnostics", [])
            if isinstance(parsed, list):
                return parsed
        return []
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def run_souc_check(code: str) -> tuple[bool, str]:
    """Run souc check on code. Returns (ok, stderr)."""
    if not SOUC.exists():
        return True, ""  # compiler not available, skip
    with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [str(SOUC), "check", tmp],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "SOUNIO_STDLIB_PATH": str(REPO_ROOT / "stdlib")}
        )
        ok = result.returncode == 0
        return ok, result.stderr
    except subprocess.TimeoutExpired:
        return False, "souc check timed out"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp)


def apply_auto_fix(code: str) -> str:
    """Run sounio-lint.py --fix on code, return fixed version (written to stdout)."""
    with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, str(LINT_SCRIPT), "--fix", tmp],
            capture_output=True, text=True, timeout=10
        )
        fixed = result.stdout.strip()
        return fixed if fixed else code
    except Exception:
        return code
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def build_feedback(lint_errors: list[dict], souc_stderr: str) -> str:
    """Build an educational feedback message from errors."""
    if not lint_errors and not souc_stderr:
        return "No issues found. Code looks syntactically valid."

    parts = []
    seen_rules = set()

    if lint_errors:
        parts.append(f"Found {len(lint_errors)} issue(s):\n")
        for e in lint_errors[:10]:  # cap at 10
            rule = e.get("rule", "unknown")
            line = e.get("line", "?")
            msg = e.get("message", "")
            parts.append(f"  Line {line}: {msg}")
            if rule not in seen_rules and rule in RULE_EXPLANATIONS:
                parts.append(f"  → {RULE_EXPLANATIONS[rule]}")
                seen_rules.add(rule)
        if len(lint_errors) > 10:
            parts.append(f"  ... and {len(lint_errors) - 10} more.")

    if souc_stderr and souc_stderr.strip():
        parts.append("\nCompiler output:")
        # First 500 chars of compiler stderr
        parts.append(souc_stderr.strip()[:500])

    return "\n".join(parts)


def log_interaction(code: str, lint_errors: list, souc_ok: bool, souc_stderr: str, fixed: str):
    """Append interaction to teacher_log.jsonl for fine-tuning data collection."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "input": code,
        "lint_errors": lint_errors,
        "souc_ok": souc_ok,
        "souc_stderr": souc_stderr[:500] if souc_stderr else "",
        "fixed": fixed,
        "had_errors": bool(lint_errors) or not souc_ok,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


class TeacherHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress default per-request logging; use our own
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._json(200, {"status": "ok", "service": "compiler-teacher"})
        elif path == "/":
            self._html(200, HOMEPAGE_HTML)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/check":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        code = payload.get("code", "")
        if not code.strip():
            self._json(400, {"error": "code field is required"})
            return

        try:
            lint_errors = run_lint(code)
            souc_ok, souc_stderr = run_souc_check(code)

            fixed = code
            if lint_errors:
                fixed = apply_auto_fix(code)

            feedback = build_feedback(lint_errors, souc_stderr if not souc_ok else "")
            ok = not lint_errors and souc_ok

            log_interaction(code, lint_errors, souc_ok, souc_stderr, fixed)

            self._json(200, {
                "ok": ok,
                "lint_errors": lint_errors,
                "souc_ok": souc_ok,
                "souc_stderr": souc_stderr[:500] if souc_stderr else "",
                "feedback": feedback,
                "fixed": fixed,
            })
        except Exception:
            self._json(500, {"error": traceback.format_exc()})

    def _json(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, status: int, html: str):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HOMEPAGE_HTML = """<!DOCTYPE html>
<html>
<head><title>Sounio Compiler as Teacher</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }
  h1 { color: #58a6ff; }
  textarea { width: 100%; height: 300px; background: #161b22; color: #c9d1d9; border: 1px solid #30363d; padding: 10px; font-family: monospace; font-size: 13px; }
  button { background: #238636; color: white; border: none; padding: 8px 20px; cursor: pointer; margin: 8px 0; font-size: 14px; }
  button:hover { background: #2ea043; }
  #result { background: #161b22; border: 1px solid #30363d; padding: 15px; margin-top: 15px; white-space: pre-wrap; }
  .ok { color: #3fb950; }
  .err { color: #f85149; }
  .fix { color: #d2a8ff; }
  #fixed-area { display: none; margin-top: 10px; }
</style>
</head>
<body>
<h1>Sounio Compiler as Teacher</h1>
<p>Paste Sounio code below. The service will detect Rust hallucinations and explain how to fix them.</p>
<textarea id="code" placeholder="fn main() with IO {
    println(&quot;Hello, Sounio!&quot;)
}"></textarea>
<br>
<button onclick="check()">Check Code</button>
<div id="result"></div>
<div id="fixed-area">
<p class="fix">Auto-fixed version:</p>
<textarea id="fixed" readonly></textarea>
</div>
<script>
async function check() {
  const code = document.getElementById('code').value;
  const res = document.getElementById('result');
  res.innerHTML = 'Checking...';
  try {
    const r = await fetch('/check', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code})
    });
    const d = await r.json();
    if (d.ok) {
      res.innerHTML = '<span class="ok">✓ Valid Sounio</span>\\n\\n' + (d.feedback || '');
    } else {
      res.innerHTML = '<span class="err">✗ Issues found</span>\\n\\n' + d.feedback;
      if (d.fixed && d.fixed !== code) {
        document.getElementById('fixed-area').style.display = 'block';
        document.getElementById('fixed').value = d.fixed;
      }
    }
  } catch(e) {
    res.innerHTML = '<span class="err">Error: ' + e + '</span>';
  }
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Sounio Compiler as Teacher service")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Sounio Compiler as Teacher")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Logging interactions to {LOG_PATH}")
    print(f"Press Ctrl+C to stop.\n")

    server = HTTPServer((args.host, args.port), TeacherHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
