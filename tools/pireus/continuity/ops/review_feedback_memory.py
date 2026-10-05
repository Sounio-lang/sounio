#!/usr/bin/env python3
"""Read-only comparison of archived memory observations; no runtime changes."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CLOSURE_SHA = "928d6ac9f19c7a2252464366cefd7044c3d48bbc3ef7aa4719be5ff1f7a1cb57"
GUARD = 33 * 1024**3

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def review(root):
    closure_raw = (root / "attempt-closure.json").read_bytes()
    if digest(closure_raw) != CLOSURE_SHA:
        raise ValueError("closure identity mismatch")
    closure = json.loads(closure_raw)
    for name, sha in closure["evidence_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path.read_bytes()) != sha:
            raise ValueError("archive artifact mismatch: " + name)
    out = dict(schema="pireus-feedback-memory-review-v1", closure_sha256=CLOSURE_SHA,
               scope="observational two sequential jobs; censored second arm",
               root_cause_established=False, new_profile_qualified=False, arms={})
    for arm, folder in (("without-feedback", "without-feedback-11969"),
                        ("with-feedback", "with-feedback-11970")):
        rows = []
        for line_number, line in enumerate((root / folder / "launch.log").read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "stage" in row:
                rows.append(dict(row, source_line=line_number))
        ranks = {}
        for rank in ("0", "1"):
            events = [r for r in rows if r.get("rank") == rank]
            ext = [r for r in events if r["stage"] == "OFFLINE_EXTEND_END"]
            ready = [r for r in events if r["stage"] == "OFFLINE_MODEL_READY"]
            samples = [r for r in events if r["stage"] in
                       ("MEMORY_GUARD_SAMPLE", "MEMORY_GUARD_STOP", "MEMORY_GUARD_CHILD_EXIT")]
            minimum = min(r["minimum_bytes"] for r in samples)
            after_first = [r for r in ext if r["index"] >= 1]
            saved = [r for r in events if r["stage"] == "OFFLINE_PROPOSAL_SAVED"]
            ranks[rank] = dict(ready_available_bytes=ready[0]["available_bytes"],
                guard_minimum_bytes=minimum, guard_margin_bytes=minimum-GUARD,
                saved_requests=len(saved), saved_output_tokens=sum(r["output_tokens"] for r in saved),
                extend_snapshots=[dict(index=r["index"], source_line=r["source_line"],
                    available_bytes=r["available_bytes"], allocated_bytes=r["cuda_allocated_bytes"],
                    reserved_bytes=r["cuda_reserved_bytes"], rss_bytes=r["process_memory"]["Rss"],
                    pss_anon_bytes=r["process_memory"]["Pss_Anon"],
                    pss_file_bytes=r["process_memory"]["Pss_File"]) for r in ext],
                post_first_allocated_range=[min(r["cuda_allocated_bytes"] for r in after_first),
                                            max(r["cuda_allocated_bytes"] for r in after_first)],
                post_first_reserved_range=[min(r["cuda_reserved_bytes"] for r in after_first),
                                           max(r["cuda_reserved_bytes"] for r in after_first)])
        out["arms"][arm] = ranks
    delta = {}
    for rank in ("0", "1"):
        a = out["arms"]["without-feedback"][rank]
        b = out["arms"]["with-feedback"][rank]
        paired = []
        for x, y in zip(a["extend_snapshots"], b["extend_snapshots"]):
            if x["index"] != y["index"]:
                raise ValueError("unpaired snapshot index")
            paired.append(dict(index=x["index"], **{
                k: y[k]-x[k] for k in ("available_bytes", "allocated_bytes", "reserved_bytes",
                                      "rss_bytes", "pss_anon_bytes", "pss_file_bytes")}))
        delta[rank] = paired
    out["with_minus_without_at_extend_end"] = delta
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=BASE/"validation/feedback-smoke-inference-20260909")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = review(args.root)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True)+"\n")
