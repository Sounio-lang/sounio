#!/usr/bin/env python3
"""
registry_serve.py — Local development registry for the Sounio Package Manager.

Simulates registry.sounio.org/api/v1 endpoints on localhost:8080.
Use this to develop/test `souc pkg install` and `souc pkg publish`
without network access to the production registry.

Usage:
    python3 scripts/dev/registry_serve.py [--port 8080] [--db data/registry.json]

Endpoints:
    GET  /api/v1/search?q=<query>&min_score=<0-100>&regulatory=<bool>
    GET  /api/v1/packages/<name>/<version|latest>
    GET  /api/v1/packages/<name>/epistemic-report
    POST /api/v1/packages         — publish
    GET  /health

Epistemic scoring enforced:
    - Minimum epistemic-score for publish: 60%
    - Regulatory tier requires >= 90% + human review flag
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# In-memory package database — seeded with curated packages
# ---------------------------------------------------------------------------

SEED_PACKAGES = [
    {
        "name":              "epistemic-core",
        "version":           "0.1.0",
        "description":       "Core epistemic types: Knowledge<T>, GUM propagation, confidence gates",
        "epistemic_score_pct": 97,
        "gum_compliant":     True,
        "regulatory_ready":  True,
        "provenance_level":  "strong",
        "tier":              3,
        "sha256":            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "source":            "registry+https://registry.sounio.org",
        "published_at":      "2026-04-20T00:00:00Z",
        "downloads":         0,
    },
    {
        "name":              "epistemic-stats",
        "version":           "0.1.0",
        "description":       "Bayesian inference, hypothesis testing, credibility intervals",
        "epistemic_score_pct": 88,
        "gum_compliant":     True,
        "regulatory_ready":  False,
        "provenance_level":  "strong",
        "tier":              2,
        "sha256":            "beef0011beef0011beef0011beef0011beef0011beef0011beef0011beef0011",
        "source":            "registry+https://registry.sounio.org",
        "published_at":      "2026-04-20T00:00:00Z",
        "downloads":         0,
    },
    {
        "name":              "darwin-pbpk",
        "version":           "0.4.2",
        "description":       "Physiologically-based pharmacokinetic models with epistemic uncertainty",
        "epistemic_score_pct": 94,
        "gum_compliant":     True,
        "regulatory_ready":  True,
        "provenance_level":  "strong",
        "tier":              3,
        "sha256":            "dead0000dead0000dead0000dead0000dead0000dead0000dead0000dead0000",
        "source":            "registry+https://registry.sounio.org",
        "published_at":      "2026-04-20T00:00:00Z",
        "downloads":         0,
    },
    {
        "name":              "snn-fractal",
        "version":           "0.2.1",
        "description":       "Spiking neural networks with fractal dynamics",
        "epistemic_score_pct": 79,
        "gum_compliant":     False,
        "regulatory_ready":  False,
        "provenance_level":  "medium",
        "tier":              1,
        "sha256":            "cafe0000cafe0000cafe0000cafe0000cafe0000cafe0000cafe0000cafe0000",
        "source":            "registry+https://registry.sounio.org",
        "published_at":      "2026-04-20T00:00:00Z",
        "downloads":         0,
    },
]

# Working copy — mutable (accepts publishes during the session)
_db: list[dict] = list(SEED_PACKAGES)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_name(t: int) -> str:
    return {0: "experimental", 1: "community", 2: "curated", 3: "regulatory"}.get(t, "unknown")


def _epistemic_report(pkg: dict) -> dict:
    return {
        "name":                pkg["name"],
        "version":             pkg["version"],
        "epistemic_score_pct": pkg["epistemic_score_pct"],
        "tier":                _tier_name(pkg["tier"]),
        "gum_compliant":       pkg["gum_compliant"],
        "regulatory_ready":    pkg["regulatory_ready"],
        "provenance_level":    pkg["provenance_level"],
        "sha256":              pkg["sha256"],
        "source":              pkg["source"],
        "published_at":        pkg["published_at"],
        "confidence_gate_default": "0.90" if pkg["epistemic_score_pct"] >= 90 else "0.80",
    }


def _search(query: str, min_score: int, regulatory_only: bool) -> list[dict]:
    q = query.lower()
    results = []
    for p in _db:
        if q and q not in p["name"].lower() and q not in p["description"].lower():
            continue
        if p["epistemic_score_pct"] < min_score:
            continue
        if regulatory_only and not p["regulatory_ready"]:
            continue
        results.append(p)
    results.sort(key=lambda x: x["epistemic_score_pct"], reverse=True)
    return results


def _get_package(name: str, version: str) -> dict | None:
    for p in reversed(_db):  # latest first
        if p["name"] == name and (version in ("latest", p["version"])):
            return p
    return None

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class RegistryHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default noisy logging
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {fmt % args}", flush=True)

    def _send_json(self, code: int, data):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, msg: str):
        self._send_json(code, {"error": msg, "code": code})

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = parse_qs(parsed.query)

        # -- Health ---------------------------------------------------------
        if path == "/health":
            self._send_json(200, {"status": "ok", "packages": len(_db)})
            return

        # -- Search ---------------------------------------------------------
        if path == "/api/v1/search":
            q            = qs.get("q", [""])[0]
            min_score    = int(qs.get("min_score", ["0"])[0])
            regulatory   = qs.get("regulatory", ["false"])[0].lower() == "true"
            results      = _search(q, min_score, regulatory)
            self._send_json(200, {
                "total":    len(results),
                "packages": results,
            })
            return

        # -- Package metadata or epistemic report ---------------------------
        parts = path.split("/")
        # /api/v1/packages/<name>/<version_or_report>
        if len(parts) >= 5 and parts[1] == "api" and parts[2] == "v1" and parts[3] == "packages":
            name = parts[4]
            sub  = parts[5] if len(parts) > 5 else "latest"

            if sub == "epistemic-report":
                pkg = _get_package(name, "latest")
                if not pkg:
                    self._send_error(404, f"Package '{name}' not found")
                    return
                self._send_json(200, _epistemic_report(pkg))
                return

            pkg = _get_package(name, sub)
            if not pkg:
                self._send_error(404, f"Package '{name}@{sub}' not found")
                return
            self._send_json(200, pkg)
            return

        self._send_error(404, f"Unknown path: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path != "/api/v1/packages":
            self._send_error(404, f"Unknown path: {path}")
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        name    = data.get("name", "")
        version = data.get("version", "0.0.0")
        score   = int(data.get("epistemic_score_pct", 0))
        gum     = bool(data.get("gum_compliant", False))
        reg     = bool(data.get("regulatory_ready", False))

        if not name:
            self._send_error(400, "name is required")
            return
        if score < 60:
            self._send_error(422, f"epistemic_score_pct {score} below minimum (60)")
            return

        tier = 0
        if score >= 90 and reg:   tier = 3
        elif score >= 85 and gum: tier = 2
        elif score >= 70:         tier = 1

        entry = {
            "name":                name,
            "version":             version,
            "description":         data.get("description", ""),
            "epistemic_score_pct": score,
            "gum_compliant":       gum,
            "regulatory_ready":    reg,
            "provenance_level":    data.get("provenance_level", "medium"),
            "tier":                tier,
            "sha256":              data.get("sha256", "0" * 64),
            "source":              "registry+http://localhost:8080",
            "published_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "downloads":           0,
            "review_required":     (tier == 3),
        }
        _db.append(entry)

        status = "published"
        msg    = f"Package '{name}@{version}' accepted (tier={_tier_name(tier)})"
        if tier == 3:
            status = "pending_review"
            msg    = f"Package '{name}@{version}' queued for human review (regulatory tier)"

        print(f"[registry] PUBLISH {name}@{version} score={score}% tier={_tier_name(tier)}")
        self._send_json(201, {
            "status":          status,
            "message":         msg,
            "tier":            _tier_name(tier),
            "review_required": tier == 3,
        })

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sounio local dev registry")
    ap.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    ap.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    ap.add_argument("--db", help="Load additional packages from JSON file")
    args = ap.parse_args()

    if args.db and os.path.exists(args.db):
        with open(args.db) as f:
            extra = json.load(f)
            _db.extend(extra if isinstance(extra, list) else [extra])
        print(f"[registry] Loaded extra packages from {args.db}")

    print(f"[registry] Sounio local dev registry")
    print(f"[registry] Listening on http://{args.host}:{args.port}")
    print(f"[registry] Seeded with {len(_db)} packages")
    print(f"[registry] Base URL: http://{args.host}:{args.port}/api/v1")
    print(f"[registry] Health:   http://{args.host}:{args.port}/health")
    print(f"[registry] Search:   http://{args.host}:{args.port}/api/v1/search?q=epistemic")
    print(f"[registry] Use --local flag in souc commands to hit this registry")
    print(f"[registry] Press Ctrl+C to stop")
    print()

    server = HTTPServer((args.host, args.port), RegistryHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[registry] Stopped.")

if __name__ == "__main__":
    main()
