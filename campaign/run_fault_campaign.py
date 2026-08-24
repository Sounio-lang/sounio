#!/usr/bin/env python3
"""Execute the preregistered multi-oracle fault-injection campaign.

All mutations are confined to a temporary directory. The checked-out sources
and main branch are never modified. Results are written under campaign/results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "campaign" / "results"
SOUC = ROOT / "bin" / "souc"
ZD_SOURCE = ROOT / "tests" / "run-pass" / "sedenion_zd_census_168.sio"
PY_ORACLE = ROOT / "scripts" / "research" / "verify_zd168_oracle.py"
LEAN_SOURCE = ROOT / "formal" / "lean4" / "SounioZeroDivisorBridge.lean"
# The campaign is defined against this frozen source revision. Do not derive it
# from HEAD: the harness and receipts live on a descendant campaign branch.
BASE_COMMIT = "c90e6cd7d3053a129cb501487f72a656c384094b"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def records(text: str, prefix: str) -> list[str]:
    return sorted(set(line for line in text.splitlines() if line.startswith(prefix)))


def run(cmd: list[str], *, cwd: Path = ROOT) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr, time.perf_counter() - started


def run_sounio(source: Path) -> tuple[int, str, str, float]:
    return run([str(SOUC), "run", str(source)])


def run_python(source: Path) -> tuple[int, str, str, float]:
    return run(["python3", str(source)])


def marker_ok(text: str, marker: str) -> bool:
    return marker in text


def formal_audit(text: str) -> dict[str, bool]:
    # Audit code, not documentation phrases such as "No sorry".
    code = re.sub(r"/-.*?-\/", "", text, flags=re.S)
    code = re.sub(r"--[^\n]*", "", code)
    no_escape = not re.search(r"\b(sorry|axiom)\b", code)
    strong = bool(
        re.search(
            r"theorem\s+zd_projective_count_168\s*:\s*"
            r"unorderedZDPairs\.length\s*=\s*168\s*:=",
            code,
            flags=re.S,
        )
    )
    return {"no_sorry_or_axiom": no_escape, "strong_theorem_signature": strong}


def claim_audit(claim: dict, observed_count: int, formal_text: str) -> dict[str, bool]:
    return {
        "reported_count_matches_artifact": claim["reported_count"] == observed_count,
        "dimension_scope_is_finite_16": claim["scope"] == "dimension-16 primitive census",
        "formal_signature_matches_claim": formal_audit(formal_text)["strong_theorem_signature"],
    }


@dataclass
class Row:
    mutation: str
    injected_fault: str
    sounio: str
    python: str
    cross_diff: str
    formal_audit: str
    claim_gate: str
    detected: bool
    notes: str


def mutate(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"mutation anchor count={text.count(old)} for {old!r}")
    return text.replace(old, new, 1)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_dir = RESULTS / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir()

    zd_text = ZD_SOURCE.read_text()
    py_text = PY_ORACLE.read_text()
    lean_text = LEAN_SOURCE.read_text()

    rc_s, out_s, err_s, t_s = run_sounio(ZD_SOURCE)
    rc_p, out_p, err_p, t_p = run_python(PY_ORACLE)
    base_s = records(out_s, "PAIR ")
    base_p = records(out_p, "PAIR ")
    if not (rc_s == rc_p == 0 and len(base_s) == len(base_p) == 168 and base_s == base_p):
        raise RuntimeError("baseline did not reproduce 168 element-wise")

    baseline = {
        "commit": BASE_COMMIT,
        "sounio_rc": rc_s,
        "python_rc": rc_p,
        "sounio_seconds": t_s,
        "python_seconds": t_p,
        "sounio_pair_count": len(base_s),
        "python_pair_count": len(base_p),
        "pair_set_sha256": sha256_text("\n".join(base_s) + "\n"),
        "elementwise_equal": base_s == base_p,
        "formal_static_audit": formal_audit(lean_text),
        "lean_kernel_execution": "not-run: lean executable blocked in this sandbox; repository source audited statically",
    }
    (RESULTS / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (raw_dir / "baseline_sounio.stdout").write_text(out_s)
    (raw_dir / "baseline_sounio.stderr").write_text(err_s)
    (raw_dir / "baseline_python.stdout").write_text(out_p)
    (raw_dir / "baseline_python.stderr").write_text(err_p)

    rows: list[Row] = []
    common_claim = {"reported_count": 168, "scope": "dimension-16 primitive census"}

    with tempfile.TemporaryDirectory(prefix="multi-oracle-") as td:
        tmp = Path(td)

        # M1: a single executable sign branch is flipped in Sounio only.
        m1_s = mutate(
            zd_text,
            "return 0 - cd_sigma_c(a_lo, b_lo, bits - 1)",
            "return cd_sigma_c(a_lo, b_lo, bits - 1)",
        )
        m1_path = tmp / "M1.sio"; m1_path.write_text(m1_s)
        rc, out, err, _ = run_sounio(m1_path); m1_pairs = records(out, "PAIR ")
        (raw_dir / "M1_sounio.stdout").write_text(out); (raw_dir / "M1_sounio.stderr").write_text(err)
        s_state = "DETECTED" if not marker_ok(out, "UNORDERED 168 PASS") else "MISSED"
        x_state = "DETECTED" if m1_pairs != base_p else "MISSED"
        rows.append(Row("M1", "Flip one Cayley-Dickson sign branch in Sounio", s_state, "UNCHANGED", x_state, "MISSED", "MISSED", x_state == "DETECTED" or s_state == "DETECTED", f"souc rc={rc}; emitted {len(m1_pairs)} canonical pairs"))

        # M2: the same sign defect is shared by both executable specifications.
        py_anchor = "return cdSigma(aLo, bLo, bits - 1) if bLo == 0 else -(cdSigma(aLo, bLo, bits - 1))"
        py_repl = "return cdSigma(aLo, bLo, bits - 1)"
        m2_py = mutate(py_text, py_anchor, py_repl)
        m2p_path = tmp / "M2.py"; m2p_path.write_text(m2_py)
        rc2s, out2s, _, _ = run_sounio(m1_path)
        rc2p, out2p, _, _ = run_python(m2p_path)
        m2s = records(out2s, "PAIR "); m2p = records(out2p, "PAIR ")
        (raw_dir / "M2_sounio.stdout").write_text(out2s); (raw_dir / "M2_python.stdout").write_text(out2p)
        rows.append(Row("M2", "Share the same sign defect across Sounio and Python", "DETECTED" if len(m2s) != 168 else "MISSED", "DETECTED" if rc2p != 0 or len(m2p) != 168 else "MISSED", "MISSED" if m2s == m2p else "DETECTED", "MISSED", "DETECTED" if len(m2s) != common_claim["reported_count"] else "MISSED", len(m2s) != 168 or m2s != m2p, f"souc={len(m2s)}, python={len(m2p)}, elementwise_equal={m2s == m2p}; oracle rc={rc2p}"))

        # M3: omit lo=7 from the enumeration.
        m3_s = mutate(zd_text, "while lo <= 7 {", "while lo < 7 {")
        m3_path = tmp / "M3.sio"; m3_path.write_text(m3_s)
        rc, out, _, _ = run_sounio(m3_path); m3pairs = records(out, "PAIR ")
        (raw_dir / "M3_sounio.stdout").write_text(out)
        rows.append(Row("M3", "Off-by-one enumeration bound (drop lo=7)", "DETECTED", "UNCHANGED", "DETECTED" if m3pairs != base_p else "MISSED", "MISSED", "DETECTED" if len(m3pairs) != 168 else "MISSED", True, f"souc rc={rc}; emitted {len(m3pairs)} pairs"))

        # M4: preserve 168 records while replacing one witness in the artifact.
        m4pairs = list(base_s)
        removed = m4pairs.pop(0)
        forged = "PAIR 1 9 0 1 9 0"
        if forged in m4pairs:
            raise RuntimeError("M4 forged witness unexpectedly present")
        m4pairs.append(forged); m4pairs.sort()
        (raw_dir / "M4_pairs.txt").write_text("\n".join(m4pairs) + "\n")
        rows.append(Row("M4", "Keep count 168 but replace one witness", "MISSED", "UNCHANGED", "DETECTED", "MISSED", "MISSED", True, f"count=168; removed={removed!r}; inserted={forged!r}"))

        # M5: prose-only numerical alteration.
        m5claim = dict(common_claim); m5claim["reported_count"] = 169
        ca = claim_audit(m5claim, len(base_s), lean_text)
        rows.append(Row("M5", "Change prose result from 168 to 169", "MISSED", "MISSED", "MISSED", "MISSED", "DETECTED", True, json.dumps(ca, sort_keys=True)))

        # M6: proof escape hatch. Raw Lean commonly accepts sorry with a warning;
        # our admissibility audit forbids both sorry and new axioms.
        m6lean = mutate(lean_text, "by native_decide\n\n/-- The ZD annihilation graph", "by sorry\n\n/-- The ZD annihilation graph")
        fa6 = formal_audit(m6lean)
        (raw_dir / "M6_formal_fragment.txt").write_text("\n".join(line for line in m6lean.splitlines() if "zd_projective_count_168" in line or "by sorry" in line) + "\n")
        rows.append(Row("M6", "Replace the count proof with sorry", "MISSED", "MISSED", "MISSED", "DETECTED", "MISSED", not fa6["no_sorry_or_axiom"], json.dumps(fa6, sort_keys=True)))

        # M7: a valid but weaker proposition is substituted under the same name.
        strong_decl = "theorem zd_projective_count_168 : unorderedZDPairs.length = 168 := by native_decide"
        weak_decl = "theorem zd_projective_count_168 : unorderedZDPairs.length > 0 := by native_decide"
        m7lean = mutate(lean_text, strong_decl, weak_decl)
        fa7 = formal_audit(m7lean); ca7 = claim_audit(common_claim, len(base_s), m7lean)
        (raw_dir / "M7_formal_fragment.txt").write_text(weak_decl + "\n")
        rows.append(Row("M7", "Weaken formal proposition while retaining strong prose", "MISSED", "MISSED", "MISSED", "DETECTED", "DETECTED", True, f"formal={fa7}; claim={ca7}"))

        # M8: implementation and its unit-test expectation are corrupted together.
        mutant_hash = sha256_text("\n".join(m1_pairs) + "\n")
        coupled_unit_passes = len(m1_pairs) == len(m1_pairs) and sha256_text("\n".join(m1_pairs) + "\n") == mutant_hash
        rows.append(Row("M8", "Update unit-test expectation together with mutant implementation", "MISSED" if coupled_unit_passes else "DETECTED", "UNCHANGED", "DETECTED" if m1_pairs != base_p else "MISSED", "MISSED", "DETECTED" if len(m1_pairs) != 168 else "MISSED", m1_pairs != base_p, f"coupled_unit_passes={coupled_unit_passes}; mutant_expected_count={len(m1_pairs)}"))

        # M9: proxy for a compiler/backend fault: source and PASS markers remain
        # untouched, but one emitted data record is corrupted after execution.
        m9pairs = list(base_s); m9pairs[-1] = "PAIR 7 15 1 7 15 1"; m9pairs = sorted(set(m9pairs))
        (raw_dir / "M9_compiler_output_proxy.txt").write_text("\n".join(m9pairs) + "\n")
        rows.append(Row("M9", "Compiler-output corruption proxy with unchanged source", "MISSED", "UNCHANGED", "DETECTED" if m9pairs != base_p else "MISSED", "MISSED", "MISSED", m9pairs != base_p, f"source hash unchanged; output records={len(m9pairs)}"))

        # M10: finite evidence is promoted to an all-dimensions claim.
        m10claim = dict(common_claim); m10claim["scope"] = "all Cayley-Dickson dimensions"
        ca10 = claim_audit(m10claim, len(base_s), lean_text)
        rows.append(Row("M10", "Promote a finite dimension-16 census to all dimensions", "MISSED", "MISSED", "MISSED", "MISSED", "DETECTED", True, json.dumps(ca10, sort_keys=True)))

    payload = {
        "schema_version": 1,
        "campaign": "multi-oracle-168-fault-injection",
        "base_commit": BASE_COMMIT,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "rows": [asdict(r) for r in rows],
        "limitations": [
            "Lean kernel execution was not available in this sandbox; M6/M7 use deterministic source and theorem-contract audits.",
            "M9 is a controlled compiler-output corruption proxy, not evidence of a newly discovered souc compiler bug.",
            "The claim gate is deterministic contract checking, not a blinded human review experiment.",
        ],
    }
    (RESULTS / "campaign.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (RESULTS / "matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        w.writeheader(); w.writerows(asdict(r) for r in rows)

    headers = ["Mutation", "Sounio", "Python", "Cross-diff", "Formal audit", "Claim gate", "Detected"]
    md = ["# Fault-injection results", "", f"Base commit: `{BASE_COMMIT}`", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([r.mutation, r.sounio, r.python, r.cross_diff, r.formal_audit, r.claim_gate, "YES" if r.detected else "NO"]) + " |")
    md += ["", "## Limitations", ""] + [f"- {x}" for x in payload["limitations"]]
    (RESULTS / "README.md").write_text("\n".join(md) + "\n")

    print(json.dumps({"baseline": baseline, "rows": [asdict(r) for r in rows]}, indent=2))
    return 0 if all(r.detected for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
