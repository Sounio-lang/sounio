#!/usr/bin/env python3
"""Reproduce a bounded review of immutable diagnostic11971/11972 custody."""
import argparse
import json
from pathlib import Path
from feedback_smoke import HERE, digest

ARCHIVE = HERE/"validation/lifecycle-diagnostic-inference-20260909"
PINS = {
 "without-feedback-11971": "59cad9080273692e40b6519cbfa99ddfbc238e6867a6f56cf6ccf9f5b7c3f7b0",
 "with-feedback-11972": "4353c1ec3bb21036d7025de953d888ad47c4bdf14f5c81ef04a0a31629cb5af9"}
GUARD, FLOOR = 33*1024**3, 32*1024**3

def load_arm(root, name, sha):
    p = root/name/"collected"
    raw = (p/"collection.json").read_bytes()
    if digest(raw) != sha: raise ValueError("collection identity mismatch")
    c = json.loads(raw)
    for path, expected in c["files_sha256"].items():
        f = p/path
        if not f.resolve().is_relative_to(p.resolve()) or digest(f.read_bytes()) != expected:
            raise ValueError("collected artifact mismatch: "+path)
    for path in c["missing"]:
        if (p/"worker-receipts"/path).exists():
            raise ValueError("previously missing evidence appeared")
    logs = []
    for line in (p/"launch.log").read_bytes().splitlines():
        try: row = json.loads(line)
        except ValueError: continue
        if isinstance(row, dict): logs.append(row)
    return p, c, logs

def review(root=ARCHIVE):
    # Validate both entire collections before any projection.
    arms = {name: load_arm(root, name, sha) for name,sha in PINS.items()}
    results = {}
    for name,(p,c,logs) in arms.items():
        job = c["job"]
        expected_count = 8 if job == "11971" else 1
        accounting = [r.split("|") for r in (p/"accounting.txt").read_text().splitlines()]
        main = [r for r in accounting if r[0] == job]
        expected = ["COMPLETED","0:0"] if job == "11971" else ["FAILED","75:0"]
        if len(main) != 1 or main[0][2:4] != expected:
            raise ValueError("terminal accounting mismatch")
        rank_summaries = []
        for rank in (0,1):
            rows = [json.loads(x) for x in (p/f"worker-receipts/lifecycle-{job}-{rank}.jsonl").read_bytes().splitlines()]
            if (not rows or rows[0]["stage"] != "OBSERVER_START"
                or len({r["pid"] for r in rows}) != 1
                or any(r["job"] != job or r["rank"] != str(rank)
                       or r["schema"] != "pireus-lifecycle-observation-v1" for r in rows)):
                raise ValueError("journal identity mismatch")
            ended = rows[-1]["stage"] == "OBSERVER_END"
            if ended != (job == "11971"): raise ValueError("journal tail mismatch")
            released = [r for r in rows if r["stage"] == "REFERENCES_RELEASED"]
            if [r["index"] for r in released] != list(range(expected_count)):
                raise ValueError("release coverage mismatch")
            for i in range(expected_count):
                a=(p/f"worker-receipts/rank-0-{i:03d}.json").read_bytes()
                b=(p/f"worker-receipts/rank-1-{i:03d}.json").read_bytes()
                response=json.loads(a)
                if a != b or response["job"] != job or response["index"] != i:
                    raise ValueError("paired saved response mismatch")
                hits=[r for r in logs if r.get("stage")=="OFFLINE_PROPOSAL_SAVED" and r.get("rank")==str(rank) and r.get("index")==i]
                if len(hits)!=1 or hits[0]["job"]!=job or hits[0]["response_sha256"]!=digest(a):
                    raise ValueError("saved response log mismatch")
            stages = ("DECODE_ENTRY","CLEANUP_BEFORE","CLEANUP_AFTER","REFERENCES_RELEASED")
            matched = {s:next(r for r in rows if r["stage"]==s and r["index"]==0) for s in stages}
            before,after,release = (matched[s] for s in stages[1:])
            samples = [r for r in rows if r["stage"]=="HOST_SAMPLE"]
            last_hook = [r for r in rows if "cuda" in r][-1]
            rank_summaries.append(dict(rank=rank,journal_rows=len(rows),observer_end=ended,
                completed_release_cycles=len(released),first_request_hooks=matched,
                post_release_cuda=[dict(index=r["index"],**r["cuda"]) for r in released],
                cleanup_allocated_delta=after["cuda"]["allocated"]-before["cuda"]["allocated"],
                references_allocated_delta=release["cuda"]["allocated"]-after["cuda"]["allocated"],
                host_min_sampled_bytes=min(r["host"]["MemAvailable"] for r in rows if r["host"]["MemAvailable"] is not None),
                max_host_sample_gap_ns=max(r["sample_gap_ns"] for r in samples),
                final_hook=last_hook,final_observation=rows[-1],
                last_host_after_last_hook_ns=rows[-1]["monotonic_ns"]-last_hook["monotonic_ns"],
                observation_errors=sum(bool(r.get("host_error") or r.get("process_error") or r.get("owned_child_error")) for r in rows),
                hook_duration_ns_sum=sum(r["observation_duration_ns"] for r in rows if "cuda" in r)))
        stops = [r for r in logs if r.get("stage")=="MEMORY_GUARD_STOP"]
        if job == "11972":
            if len(stops)!=1 or stops[0]["rank"]!="0" or stops[0]["job"]!=job:
                raise ValueError("expected single rank0 memory stop")
            stop=stops[0]
            if not FLOOR < stop["available_bytes"] < GUARD or stop["protected_floor_bytes"]!=FLOOR:
                raise ValueError("guard/floor evidence mismatch")
            for rank in (0,1):
                if (p/f"worker-receipts/rank-{rank}-complete.json").exists():
                    raise ValueError("unexpected completion")
        results[name]=dict(job=job,accounting_state=main[0][2],exit_code=main[0][3],
            completed_requests=expected_count,
            output_tokens=sum(len(json.loads((p/f"worker-receipts/rank-0-{i:03d}.json").read_bytes())["output_ids"]) for i in range(expected_count)),
            memory_guard_stop=stops,missing_files=sorted(c["missing"]),ranks=rank_summaries)
    left,right=(results[n] for n in PINS)
    delta=[]
    for rank in (0,1):
        l=left["ranks"][rank]["first_request_hooks"]["REFERENCES_RELEASED"]
        r=right["ranks"][rank]["first_request_hooks"]["REFERENCES_RELEASED"]
        delta.append(dict(rank=rank,cuda_reserved_bytes=r["cuda"]["reserved"]-l["cuda"]["reserved"],
            cuda_allocated_bytes=r["cuda"]["allocated"]-l["cuda"]["allocated"],
            process_pss_bytes=r["process"]["Pss"]-l["process"]["Pss"]))
    return dict(schema="pireus-lifecycle-diagnostic-review-v1",collections_sha256=PINS,
        attempt_closed=True,both_arms_attempted=True,inference_completed=False,
        paired_diagnostic_complete=False,pilot_acceptance=False,
        root_cause_identified=False,instrumentation_overhead_qualified=False,
        arms=results,matched_first_release_feedback_minus_control=delta,
        boundaries=["sequential jobs and unequal prompts confound causal attribution",
            "host and CUDA/process views are not additive",
            "CUDA peaks are process lifetime, not decode-specific",
            "failed journal has no observer end or synchronized guardian-stop snapshot",
            "hook duration excludes unmeasured scheduling/GIL and other overhead"])

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=review()
    with a.output.open("x") as out:out.write(json.dumps(result,indent=2)+"\n")
