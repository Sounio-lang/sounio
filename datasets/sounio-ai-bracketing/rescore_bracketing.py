#!/usr/bin/env python3
"""Rescore an SBMP probe run with an LLM judge (independent provider
recommended). Reads a results JSONL from probe_bracketing.py and asks the
judge, per item:

  1. does response_left  express answer_left  (YES/NO)
  2. does response_left  express answer_right (YES/NO)
  3. does response_right express answer_left  (YES/NO)
  4. does response_right express answer_right (YES/NO)
  5. do the two responses give semantically DIFFERENT answers to the probe
     question (YES/NO)   <- the real associator, not wording

Usage:
  python rescore_bracketing.py results/grok-4.3_run1.jsonl \
      --judge-model glm-5.2 --base-url ... --api-key-env ZAI_API_KEY
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def call_model(base_url, api_key, model, prompt, max_tokens=8, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=json.dumps({"model": model,
                                 "messages": [{"role": "user", "content": prompt}],
                                 # generous budget: some endpoints (glm) spend
                                 # tokens on reasoning_content before content
                                 "max_tokens": max(max_tokens, 512),
                                 "temperature": 0.0}).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
            msg = body["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            if not text:
                # reasoning-only reply: take the last line of reasoning
                text = (msg.get("reasoning_content") or "").strip().splitlines()
                text = text[-1] if text else ""
            return text.strip().upper().startswith("YES")
        except Exception as e:  # network/timeout/rate-limit: back off and retry
            last = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"judge call failed after {retries} attempts: {last}")


def match_q(response, answer):
    return ("You are scoring whether two short answers express the same "
            "content. Answer YES or NO only.\n\nANSWER A: " + response +
            "\n\nANSWER B: " + answer + "\n\nSame content?")


def diff_q(probe, r_left, r_right):
    return ("Two answers to the same question about a dialogue are given. "
            "Do they give semantically DIFFERENT answers (different facts, "
            "referents, or conclusions — not just different wording)? "
            "Answer YES or NO only.\n\nQUESTION: " + probe +
            "\n\nANSWER 1: " + r_left + "\n\nANSWER 2: " + r_right +
            "\n\nDifferent content?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results JSONL from probe_bracketing.py")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        sys.exit(f"missing API key env {args.api_key_env}")

    rows = [json.loads(l) for l in open(args.run)]
    if rows and "summary" in rows[-1]:
        rows.pop()

    out = Path(args.out) if args.out else Path(args.run).with_suffix(".judged.jsonl")
    done_ids = set()
    if out.exists():  # resume: skip already-judged items
        for l in out.open():
            try:
                d = json.loads(l)
                if "judge" in d:
                    done_ids.add(d["id"])
            except json.JSONDecodeError:
                pass
        print(f"resume: {len(done_ids)} items already judged")

    out_rows = []
    for it in rows:
        if it["id"] in done_ids:
            continue
        ll = call_model(args.base_url, api_key, args.judge_model,
                        match_q(it["response_left"], it["answer_left"]))
        time.sleep(0.2)
        lr = call_model(args.base_url, api_key, args.judge_model,
                        match_q(it["response_left"], it["answer_right"]))
        time.sleep(0.2)
        rl = call_model(args.base_url, api_key, args.judge_model,
                        match_q(it["response_right"], it["answer_left"]))
        time.sleep(0.2)
        rr = call_model(args.base_url, api_key, args.judge_model,
                        match_q(it["response_right"], it["answer_right"]))
        time.sleep(0.2)
        sem_diff = call_model(args.base_url, api_key, args.judge_model,
                              diff_q(it["probe_question"],
                                     it["response_left"], it["response_right"]))
        time.sleep(0.2)
        # directional reading taken under each induction
        read_left = "left" if (ll and not lr) else ("right" if (lr and not ll)
                                                    else ("both" if ll and lr else "neither"))
        read_right = "left" if (rl and not rr) else ("right" if (rr and not rl)
                                                     else ("both" if rl and rr else "neither"))
        rec = {**it, "judge": {
            "left_resp_matches": read_left, "right_resp_matches": read_right,
            "semantic_flip": sem_diff}}
        out_rows.append(rec)
        with out.open("a") as f:  # incremental: survive a SIGKILL mid-run
            f.write(json.dumps(rec) + "\n")
        print(f"{it['id']}: L-induction→{read_left} R-induction→{read_right} "
              f"sem_flip={sem_diff} gold={it['gold_human']}", flush=True)

    # final summary over ALL judged rows (resumed + new)
    all_rows = []
    for l in out.open():
        d = json.loads(l)
        if "judge" in d:
            all_rows.append(d)
    out_rows = all_rows
    n = len(out_rows)
    if n < len(rows):
        print(f"partial: {n}/{len(rows)} judged — rerun to resume")
        return
    sem_flips = sum(r["judge"]["semantic_flip"] for r in out_rows)
    directional = sum(
        r["judge"]["left_resp_matches"] == "left"
        and r["judge"]["right_resp_matches"] == "right" for r in out_rows)
    by_gold = {}
    for r in out_rows:
        g = r["gold_human"]
        by_gold.setdefault(g, {"n": 0, "sem_flip": 0, "directional": 0})
        by_gold[g]["n"] += 1
        by_gold[g]["sem_flip"] += int(r["judge"]["semantic_flip"])
        by_gold[g]["directional"] += int(
            r["judge"]["left_resp_matches"] == "left"
            and r["judge"]["right_resp_matches"] == "right")
    summary = {
        "judge": args.judge_model,
        "n_items": n,
        "semantic_flip_rate": sem_flips / max(n, 1),
        "directional_clean_rate": directional / max(n, 1),
        "by_gold": {g: {**v,
                        "sem_flip_rate": v["sem_flip"] / v["n"],
                        "directional_rate": v["directional"] / v["n"]}
                    for g, v in by_gold.items()},
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.out) if args.out else Path(args.run).with_suffix(".judged.jsonl")
    with out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
