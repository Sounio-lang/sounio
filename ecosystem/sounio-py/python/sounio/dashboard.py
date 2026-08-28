"""REST dashboard for the Sounio drug discovery pipeline.

Self-contained Flask application with embedded HTML/JS — no frontend build step.
Serves a real-time web UI showing Knowledge values from the pipeline with
uncertainty visualisation.

Usage
-----
>>> import sounio
>>> sounio.serve_dashboard()          # opens http://127.0.0.1:8765

From the command line:
    python -m sounio.dashboard
    python -m sounio.dashboard --port 9000 --pipeline path/to/pipeline.sio

API
---
GET  /                          HTML dashboard
POST /api/pipeline/run          Run the pipeline; returns JSON results
GET  /api/pipeline/status       Last run status
GET  /api/knowledge/<prov_id>   Single Knowledge value with chain (if tracked)
"""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

try:
    from flask import Flask, Response, jsonify, request
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


# ---------------------------------------------------------------------------
# Embedded single-page HTML dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sounio Drug Discovery Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5; }
    header {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: white; padding: 18px 28px;
      display: flex; align-items: center; gap: 16px;
    }
    header h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; }
    header p  { font-size: 0.8rem; opacity: 0.6; margin-top: 3px; }
    .badge {
      background: rgba(79,70,229,0.3); border: 1px solid rgba(79,70,229,0.5);
      color: #a5b4fc; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem;
    }
    main { max-width: 960px; margin: 28px auto; padding: 0 20px; }
    .controls {
      display: flex; gap: 12px; align-items: center; margin-bottom: 20px;
    }
    button {
      padding: 10px 22px; border-radius: 8px; border: none;
      cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: .15s;
    }
    #run-btn  { background: #4f46e5; color: white; }
    #run-btn:hover    { background: #4338ca; transform: translateY(-1px); }
    #run-btn:disabled { background: #9ca3af; cursor: not-allowed; transform: none; }
    #refresh-btn { background: #e5e7eb; color: #374151; }
    #refresh-btn:hover { background: #d1d5db; }
    .status-bar {
      background: white; border-radius: 10px; padding: 13px 18px;
      margin-bottom: 20px; border-left: 4px solid #6b7280;
      font-size: 0.88rem; color: #374151; display: flex; align-items: center; gap: 8px;
    }
    .status-bar.idle    { border-color: #6b7280; }
    .status-bar.ok      { border-color: #10b981; }
    .status-bar.error   { border-color: #ef4444; color: #dc2626; }
    .status-bar.running { border-color: #f59e0b; }
    .dot {
      width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    }
    .idle    .dot { background: #9ca3af; }
    .ok      .dot { background: #10b981; }
    .error   .dot { background: #ef4444; }
    .running .dot { background: #f59e0b; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    .section-title {
      font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: .07em; color: #6b7280; margin-bottom: 14px;
    }
    .grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
      gap: 16px;
    }
    .card {
      background: white; border-radius: 12px; padding: 18px;
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
      transition: box-shadow .15s, transform .15s;
    }
    .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.12); transform: translateY(-2px); }
    .card-label {
      font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: .06em; color: #9ca3af; margin-bottom: 8px;
    }
    .card-value { font-size: 1.7rem; font-weight: 800; color: #111827; line-height: 1; }
    .card-epsilon { font-size: 0.82rem; color: #6b7280; margin: 5px 0 12px; }
    .bar-track { background: #f3f4f6; border-radius: 6px; height: 7px; overflow: hidden; }
    .bar-fill  {
      height: 7px; border-radius: 6px;
      background: linear-gradient(90deg, #4f46e5, #7c3aed);
      transition: width .5s ease-out;
    }
    .conf-label { font-size: 0.72rem; color: #9ca3af; margin-top: 5px; }
    .prov-tag {
      display: inline-block; margin-top: 10px; padding: 2px 8px;
      background: #f3f4f6; border-radius: 6px;
      font-size: 0.67rem; font-family: monospace; color: #6b7280;
    }
    .empty-state {
      text-align: center; padding: 60px 0; color: #9ca3af; font-size: 0.9rem;
    }
    .empty-state svg { margin-bottom: 12px; opacity: 0.4; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Sounio Drug Discovery Dashboard</h1>
      <p>Epistemic uncertainty-aware pharmacokinetic analysis</p>
    </div>
    <span class="badge">Knowledge[T] · GUM</span>
  </header>

  <main>
    <div class="controls">
      <button id="run-btn" onclick="runPipeline()">&#9654; Run Pipeline</button>
      <button id="refresh-btn" onclick="fetchStatus()">&#8635; Refresh</button>
    </div>

    <div id="status" class="status-bar idle">
      <span class="dot"></span>
      <span id="status-text">Ready. Click "Run Pipeline" to start.</span>
    </div>

    <p class="section-title">Epistemic Results</p>
    <div id="cards" class="grid">
      <div class="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 8v4m0 4h.01"/>
        </svg>
        <p>No results yet. Run the pipeline to see epistemic values.</p>
      </div>
    </div>
  </main>

  <script>
    function setStatus(type, text) {
      const el = document.getElementById('status');
      el.className = 'status-bar ' + type;
      document.getElementById('status-text').textContent = text;
    }

    function renderCards(results) {
      const grid = document.getElementById('cards');
      if (!results || results.length === 0) {
        grid.innerHTML = '<div class="empty-state"><p>No epistemic values returned.</p></div>';
        return;
      }
      grid.innerHTML = results.map(k => {
        const absVal = Math.abs(k.value);
        const relUnc = absVal > 0 ? k.epsilon / absVal : 0;
        const conf = Math.max(0, Math.min(100, (1 - relUnc) * 100));
        const valStr = Math.abs(k.value) >= 1000
          ? k.value.toExponential(3)
          : k.value.toFixed(4);
        const epsStr = k.epsilon.toFixed(4);
        return `<div class="card">
          <div class="card-label">${k.provenance}</div>
          <div class="card-value">${valStr}</div>
          <div class="card-epsilon">&plusmn;&nbsp;${epsStr}&nbsp;&nbsp;(k=1)</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${conf.toFixed(1)}%"></div>
          </div>
          <div class="conf-label">Confidence: ${conf.toFixed(1)}%</div>
          <span class="prov-tag">${k.provenance}</span>
        </div>`;
      }).join('');
    }

    function runPipeline() {
      const btn = document.getElementById('run-btn');
      btn.disabled = true;
      setStatus('running', 'Running pipeline\u2026');
      fetch('/api/pipeline/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: '{}'
      })
      .then(r => r.json())
      .then(data => {
        btn.disabled = false;
        if (data.status === 'ok') {
          setStatus('ok',
            `Completed in ${data.elapsed_ms}ms \u2014 ${data.results.length} epistemic values`);
          renderCards(data.results);
        } else {
          setStatus('error', 'Error: ' + (data.error || 'unknown'));
        }
      })
      .catch(e => {
        btn.disabled = false;
        setStatus('error', 'Request failed: ' + e);
      });
    }

    function fetchStatus() {
      fetch('/api/pipeline/status')
        .then(r => r.json())
        .then(data => {
          if (data.status === 'ok' && data.results && data.results.length > 0) {
            setStatus('ok',
              `Last run: ${data.results.length} epistemic values (${data.elapsed_ms}ms)`);
            renderCards(data.results);
          } else if (data.status === 'error') {
            setStatus('error', 'Last run failed: ' + (data.error || 'unknown'));
          }
        })
        .catch(() => {});
    }

    window.onload = fetchStatus;
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------


def create_app(pipeline_path: Optional[str] = None) -> "Flask":
    """Create the Flask dashboard application.

    Parameters
    ----------
    pipeline_path : str, optional
        Path to a ``.sio`` pipeline file. Defaults to the bundled
        drug-discovery example when discoverable.

    Returns
    -------
    Flask
        Configured application object.

    Raises
    ------
    ImportError
        If Flask is not installed.
    """
    if not _HAS_FLASK:
        raise ImportError(
            "Flask is required for the dashboard. "
            "Install with: pip install flask"
        )

    app = Flask(__name__)
    app.config["PIPELINE_PATH"] = pipeline_path or _find_default_pipeline()
    app.config["LAST_RESULT"] = {
        "status": "idle",
        "results": [],
        "elapsed_ms": 0,
    }

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return Response(_DASHBOARD_HTML, mimetype="text/html")

    @app.route("/api/pipeline/run", methods=["POST"])
    def run_pipeline():
        path = app.config["PIPELINE_PATH"]
        if not path or not Path(path).exists():
            err = {"status": "error", "error": f"Pipeline file not found: {path}"}
            app.config["LAST_RESULT"] = err
            return jsonify(err), 404

        try:
            from ._executor import SounioExecutor
        except ImportError:
            err = {"status": "error", "error": "sounio._executor not available"}
            return jsonify(err), 500

        t0 = time.time()
        executor = SounioExecutor()
        result = executor.run_file(path, timeout=120)
        elapsed_ms = int((time.time() - t0) * 1000)

        if not result.ok:
            payload = {
                "status": "error",
                "error": result.stderr or "pipeline returned non-zero exit code",
                "elapsed_ms": elapsed_ms,
            }
            app.config["LAST_RESULT"] = payload
            return jsonify(payload), 500

        results = [
            {"value": k.value, "epsilon": k.epsilon, "provenance": k.provenance}
            for k in result.knowledge_values
        ]
        payload = {"status": "ok", "results": results, "elapsed_ms": elapsed_ms}
        app.config["LAST_RESULT"] = payload
        return jsonify(payload)

    @app.route("/api/pipeline/status")
    def pipeline_status():
        return jsonify(app.config["LAST_RESULT"])

    @app.route("/api/knowledge/<prov_id>")
    def get_knowledge(prov_id: str):
        last = app.config["LAST_RESULT"]
        for item in last.get("results", []):
            if item["provenance"] == prov_id:
                return jsonify(item)
        return jsonify({"error": f"No Knowledge value with provenance '{prov_id}'"}), 404

    return app


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def serve_dashboard(
    host: str = "127.0.0.1",
    port: int = 8765,
    pipeline_path: Optional[str] = None,
    open_browser: bool = True,
) -> None:
    """Start the Sounio dashboard server (blocking).

    Parameters
    ----------
    host : str
        Hostname to bind to (default ``"127.0.0.1"``).
    port : int
        Port number (default ``8765``).
    pipeline_path : str, optional
        Custom ``.sio`` pipeline file to run.
    open_browser : bool
        Open the dashboard in the default browser after 1 second (default True).
    """
    app = create_app(pipeline_path=pipeline_path)
    url = f"http://{host}:{port}"
    print(f"Sounio Dashboard running at {url}")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------


def _find_default_pipeline() -> Optional[str]:
    """Search common locations for the drug-discovery pipeline file."""
    candidates = [
        "drug-discovery/examples/full_pipeline.sio",
        "../drug-discovery/examples/full_pipeline.sio",
        str(
            Path(__file__).resolve().parents[4]
            / "ecosystem"
            / "drug-discovery"
            / "examples"
            / "full_pipeline.sio"
        ),
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p.resolve())
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sounio Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pipeline", default=None, dest="pipeline_path")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    serve_dashboard(
        host=args.host,
        port=args.port,
        pipeline_path=args.pipeline_path,
        open_browser=not args.no_browser,
    )
