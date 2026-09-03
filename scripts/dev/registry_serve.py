#!/usr/bin/env python3
"""
registry_serve.py — Local development registry for the Sounio Package Manager.

Simulates a read-only future registry catalog on localhost:8080.
Public publishing and registry attestation are outside R0-R2.

Usage:
    python3 scripts/dev/registry_serve.py [--port 8080] [--db data/registry.json]

Endpoints:
    GET  /api/v1/search?q=<query>
    GET  /api/v1/packages/<name>/<version|latest>
    GET  /api/v1/packages/<name>/boundary-report
    POST /api/v1/packages         - disabled until registry attestation exists
    GET  /health
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
        "ring":              "scientific-package",
        "evidence_status":   "passes-gate",
        "context_of_use":    "epistemic measurement primitives for research software",
        "visibility":        "public",
        "review_state":      "draft",
        "sha256":            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "source":            "preview+local://epistemic-core/0.1.0",
    },
    {
        "name":              "epistemic-stats",
        "version":           "0.1.0",
        "description":       "Bayesian inference, hypothesis testing, credibility intervals",
        "ring":              "scientific-package-candidate",
        "evidence_status":   "implemented",
        "context_of_use":    "statistical research software pending package inventory",
        "visibility":        "protected",
        "review_state":      "draft",
        "sha256":            "beef0011beef0011beef0011beef0011beef0011beef0011beef0011beef0011",
        "source":            "preview+local://epistemic-stats/0.1.0",
    },
    {
        "name":              "darwin-pbpk",
        "version":           "0.4.2",
        "description":       "Physiologically-based pharmacokinetic models with epistemic uncertainty",
        "ring":              "research",
        "evidence_status":   "implemented",
        "context_of_use":    "PBPK model research pending model-specific qualification",
        "visibility":        "protected",
        "review_state":      "draft",
        "sha256":            "dead0000dead0000dead0000dead0000dead0000dead0000dead0000dead0000",
        "source":            "preview+local://darwin-pbpk/0.4.2",
    },
    {
        "name":              "snn-fractal",
        "version":           "0.2.1",
        "description":       "Spiking neural networks with fractal dynamics",
        "ring":              "research",
        "evidence_status":   "implemented",
        "context_of_use":    "spiking-neural-network research",
        "visibility":        "protected",
        "review_state":      "draft",
        "sha256":            "cafe0000cafe0000cafe0000cafe0000cafe0000cafe0000cafe0000cafe0000",
        "source":            "preview+local://snn-fractal/0.2.1",
    },
]

# Working copy for optional read-only catalog extensions.
_db: list[dict] = list(SEED_PACKAGES)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _public_package(pkg: dict) -> dict:
    return {
        "name": pkg.get("name", ""),
        "version": pkg.get("version", ""),
        "description": pkg.get("description", ""),
        "ring": pkg.get("ring", "unclassified"),
        "evidence_status": pkg.get("evidence_status", "unknown"),
        "context_of_use": pkg.get("context_of_use", "undeclared"),
        "visibility": pkg.get("visibility", "protected"),
        "review_state": pkg.get("review_state", "unreviewed"),
        "sha256": pkg.get("sha256", ""),
        "source": pkg.get("source", ""),
    }


def _boundary_report(pkg: dict) -> dict:
    result = _public_package(pkg)
    result["schema"] = "sounio.registry-boundary-report.preview.v1"
    result["boundary_receipt_sha256"] = pkg.get("boundary_receipt_sha256", "")
    result["limitations"] = [
        "preview_catalog_only",
        "does_not_assert_scientific_truth",
        "does_not_assert_clinical_or_regulatory_authority",
        "does_not_assert_publication",
    ]
    return result


def _search(query: str) -> list[dict]:
    q = query.lower()
    results = []
    for p in _db:
        if q and q not in p["name"].lower() and q not in p["description"].lower():
            continue
        results.append(_public_package(p))
    results.sort(key=lambda x: (x["name"], x["version"]))
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
            q = qs.get("q", [""])[0]
            results = _search(q)
            self._send_json(200, {
                "total":    len(results),
                "packages": results,
                "limitations": ["legacy_score_and_regulatory_filters_are_ignored"],
            })
            return

        # -- Package metadata or boundary report ----------------------------
        parts = path.split("/")
        # /api/v1/packages/<name>/<version_or_report>
        if len(parts) >= 5 and parts[1] == "api" and parts[2] == "v1" and parts[3] == "packages":
            name = parts[4]
            sub  = parts[5] if len(parts) > 5 else "latest"

            if sub == "epistemic-report":
                self._send_error(410, "epistemic-report was removed; use boundary-report")
                return

            if sub == "boundary-report":
                pkg = _get_package(name, "latest")
                if not pkg:
                    self._send_error(404, f"Package '{name}' not found")
                    return
                self._send_json(200, _boundary_report(pkg))
                return

            pkg = _get_package(name, sub)
            if not pkg:
                self._send_error(404, f"Package '{name}@{sub}' not found")
                return
            self._send_json(200, _public_package(pkg))
            return

        self._send_error(404, f"Unknown path: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path != "/api/v1/packages":
            self._send_error(404, f"Unknown path: {path}")
            return

        self._send_error(
            501,
            "publishing is disabled until a separately specified registry attestation gate exists",
        )

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
