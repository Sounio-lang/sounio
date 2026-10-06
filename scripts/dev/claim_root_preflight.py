#!/usr/bin/env python3
"""Claim-root preflight: read receipts, never run gates.

Cadence (design, Codex-1): after merge to main, not on every PR.
This instrument consumes docs/audit/CI_GATE_BUDGET_2026-08-18.tsv as the
only structured receipt store today. It does not invoke, dispatch, wait
for, or retry any *_gate.sh.

A row with measured=no cannot protect a published number. skip-vacuous
rc=0 is CORREU_E_FICOU_VAZIO, never CURRENT_PASS. The umbrella is never
executed here.

Usage:
  python3 scripts/dev/claim_root_preflight.py
  python3 scripts/dev/claim_root_preflight.py --json PATH
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / "docs/audit/CI_GATE_BUDGET_2026-08-18.tsv"

# Eighteen real numeric claims (Codex-1, minus three heuristic false positives).
# published_value is the number still cited; it is not re-derived here.
MANIFEST = (
    ("dissertation_dossier_gate.sh", "5/5"),
    ("dissertation_frontend_parity_gate.sh", "14/14"),
    ("dissertation_pbpk28_parity_gate.sh", "9/9"),
    ("dissertation_pbpk_hessian_gate.sh", "5/5"),
    ("dissertation_pbpk_suite_gate.sh", "50/50"),
    ("fo_pk_struct_auc_thalf_driver_gate.sh", "AUC Var 114.6"),
    ("functor_f_g2_covariance_gate.sh", "6/6"),
    ("kaxi_ptx_golden_gate.sh", "318/318"),
    ("kretikos_kaxi_fmad_invariance_gate.sh", "18/18"),
    ("kretikos_kaxi_lse8_gate.sh", "7/7"),
    ("kretikos_kaxi_phase_y_gate.sh", "3/3"),
    ("kretikos_kaxi_sinkhorn16_gate.sh", "7/7"),
    ("native_v2_cpu_compiler_umbrella_gate.sh", "12/12"),
    ("san_imagenet_fpga_dl380_gate.sh", "8/8"),
    ("sedenion_phi_injectivity_gate.sh", "10/10"),
    ("sounio_direct_driver_support_gate.sh", "24/24"),
    ("stdlib_source_byte_ceiling_gate.sh", "PASS"),
    ("windows_pe_smoke_gate.sh", "6/6"),
)

# Attempt class Codex-1 asked for. Derived from the budget row, not re-run.
ATTEMPT = {
    "dissertation_dossier_gate.sh": "early-fail",
    "dissertation_frontend_parity_gate.sh": "full",
    "dissertation_pbpk28_parity_gate.sh": "early-fail",
    "dissertation_pbpk_hessian_gate.sh": "full",
    "dissertation_pbpk_suite_gate.sh": "full",
    "fo_pk_struct_auc_thalf_driver_gate.sh": "early-fail",
    "functor_f_g2_covariance_gate.sh": "instrument",
    "kaxi_ptx_golden_gate.sh": "instrument",
    "kretikos_kaxi_fmad_invariance_gate.sh": "skip",
    "kretikos_kaxi_lse8_gate.sh": "skip",
    "kretikos_kaxi_phase_y_gate.sh": "skip",
    "kretikos_kaxi_sinkhorn16_gate.sh": "skip",
    "native_v2_cpu_compiler_umbrella_gate.sh": "instrument",
    "san_imagenet_fpga_dl380_gate.sh": "instrument",
    "sedenion_phi_injectivity_gate.sh": "early-fail",
    "sounio_direct_driver_support_gate.sh": "full",
    "stdlib_source_byte_ceiling_gate.sh": "full",
    "windows_pe_smoke_gate.sh": "instrument",
}

NEVER_EXECUTE = frozenset(
    {
        "native_v2_cpu_compiler_umbrella_gate.sh",
    }
)


def load_budget(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if header is None:
            header = cols
            continue
        rec = dict(zip(header, cols))
        rows[rec["gate"]] = rec
    return rows


def classify(gate: str, rec: dict[str, str] | None) -> str:
    if rec is None:
        return "NUNCA_CORREU"
    measured = rec.get("measured", "")
    polarity = rec.get("polarity", "")
    kind = rec.get("kind", "")
    attempt = ATTEMPT.get(gate, "")
    if gate in NEVER_EXECUTE:
        return "CORREU_E_FICOU_VAZIO"
    if attempt in {"skip", "instrument"}:
        return "CORREU_E_FICOU_VAZIO"
    if polarity == "skip-vacuous" or kind == "skip-vacuous":
        return "CORREU_E_FICOU_VAZIO"
    if kind in {"instrument", "timeout-cap", "rebuild"}:
        return "CORREU_E_FICOU_VAZIO"
    if measured == "no":
        return "CORREU_E_FICOU_VAZIO"
    if polarity in {"red", "capped"}:
        return "CURRENT_FAIL"
    if polarity in {"pass-measured", "green"} and measured == "yes":
        return "CURRENT_PASS"
    return "CORREU_E_ENVELHECEU"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/audit/CLAIM_ROOT_PREFLIGHT_2026-08-19.json")
    args = ap.parse_args()
    t0 = time.monotonic()

    if not BUDGET.is_file():
        print("REFUTE: budget TSV missing — instrument is dead", file=sys.stderr)
        return 2
    budget = load_budget(BUDGET)

    claims = []
    for gate, published in MANIFEST:
        rec = budget.get(gate)
        state = classify(gate, rec)
        claims.append(
            {
                "gate": gate,
                "published_value": published,
                "state": state,
                "attempt": ATTEMPT[gate],
                "elapsed_sec": rec.get("elapsed_sec") if rec else None,
                "rc": rec.get("rc") if rec else None,
                "measured": rec.get("measured") if rec else None,
                "polarity": rec.get("polarity") if rec else None,
                "protects_claim": state == "CURRENT_PASS",
            }
        )

    empty = [c for c in claims if c["state"] == "CORREU_E_FICOU_VAZIO"]
    never = [c for c in claims if c["state"] == "NUNCA_CORREU"]
    fail = [c for c in claims if c["state"] == "CURRENT_FAIL"]
    passed = [c for c in claims if c["state"] == "CURRENT_PASS"]
    unprotected = [c for c in claims if not c["protects_claim"]]
    elapsed = time.monotonic() - t0

    if elapsed >= 1.0:
        print(f"REFUTE: preflight took {elapsed:.3f}s (budget <1s)", file=sys.stderr)
        return 2
    if not unprotected:
        print("REFUTE: zero unprotected claims — instrument is dead on this corpus", file=sys.stderr)
        return 2
    skip_as_pass = [
        c for c in claims if c["attempt"] == "skip" and c["state"] == "CURRENT_PASS"
    ]
    if skip_as_pass:
        print(f"REFUTE: skip-vacuous classified CURRENT_PASS: {skip_as_pass}", file=sys.stderr)
        return 2
    umbrella = next(
        c for c in claims if c["gate"] == "native_v2_cpu_compiler_umbrella_gate.sh"
    )
    if umbrella["state"] == "CURRENT_PASS" or umbrella["protects_claim"]:
        print("REFUTE: umbrella protected a claim", file=sys.stderr)
        return 2
    kaxi = next(c for c in claims if c["gate"] == "kaxi_ptx_golden_gate.sh")
    if kaxi["measured"] != "no" or kaxi["state"] != "CORREU_E_FICOU_VAZIO":
        print("REFUTE: kaxi 318/318 must be empty (DROP, not byte-diff)", file=sys.stderr)
        return 2
    driver = next(c for c in claims if c["gate"] == "sounio_direct_driver_support_gate.sh")
    if driver["state"] != "CURRENT_PASS":
        print("REFUTE: direct-driver 24/24 must stay CURRENT_PASS", file=sys.stderr)
        return 2

    summary = {
        "schema": "sounio.claim_root_preflight.v1",
        "budget_tsv": str(BUDGET.relative_to(REPO)),
        "invokes_gates": False,
        "preflight_sec": round(elapsed, 4),
        "claims_total": len(claims),
        "current_pass": len(passed),
        "current_fail": len(fail),
        "correu_e_ficou_vazio": len(empty),
        "nunca_correu": len(never),
        "unprotected": len(unprotected),
        "serial_budget_sec": sum(
            int(c["elapsed_sec"]) for c in claims if c["elapsed_sec"] is not None
        ),
        "measured_yes_sec": sum(
            int(c["elapsed_sec"])
            for c in claims
            if c["elapsed_sec"] is not None and c["measured"] == "yes"
        ),
        "measured_no_sec": sum(
            int(c["elapsed_sec"])
            for c in claims
            if c["elapsed_sec"] is not None and c["measured"] == "no"
        ),
        "of_the_11": {
            "n": sum(1 for r in budget.values() if r.get("family") == "claim-numeric"),
            "published_1915_measured_yes": 7,
            "published_1915_empty": 4,
            "after_kaxi_only_measured_yes": 6,
            "after_kaxi_only_empty": 5,
            "live_measured_yes": sum(
                1
                for r in budget.values()
                if r.get("family") == "claim-numeric" and r.get("measured") == "yes"
            ),
            "live_empty": sum(
                1
                for r in budget.values()
                if r.get("family") == "claim-numeric" and r.get("measured") == "no"
            ),
            "note": "kaxi 318/318 is DROP, not byte-diff. #1915 was 7/4; kaxi-only is 6/5; live column is 3/8 after functor/san/windows instrument-empty.",
        },
        "positive_control": {
            "unprotected_nonzero": True,
            "skip_never_current_pass": True,
            "umbrella_never_protects": True,
            "kaxi_empty": True,
            "direct_driver_pass": True,
            "preflight_under_1s": True,
        },
        "claims": claims,
    }

    out = Path(args.json)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "claims"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
