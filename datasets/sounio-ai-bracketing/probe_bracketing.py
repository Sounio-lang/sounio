#!/usr/bin/env python3
"""SBMP probe harness — induce both bracketings of each 3-turn item in a
target LLM and measure the empirical associator (answer-flip rate).

Protocol (declared in README.md): each item is presented twice with identical
turns; only the grouping framing differs. The probe question is then asked.
The strong non-associativity hypothesis predicts answer flips on items where
gold_human != "both"; bracket-blindness predicts ~zero flips.

Usage:
  python probe_bracketing.py --model grok-4.3 \
      --base-url https://api.x.ai/v1 --api-key-env XAI_API_KEY \
      [--limit 10] [--out results/run.jsonl]

A --judge-model may be given for LLM-judged answer matching; otherwise a
normalised substring heuristic is used. The harness never edits the dataset.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATEGORIES = ["repair", "anaphora", "ellipsis", "scope", "qud"]

LEFT_FRAME = (
    "In this dialogue, speaker B's turn (turn 2) responds directly to "
    "speaker A's first turn, forming one exchange. Speaker A's final turn "
    "then continues from that exchange as a whole."
)
RIGHT_FRAME = (
    "In this dialogue, speaker A's final turn continues directly from "
    "speaker B's turn (turn 2) — turns 2 and 3 form one exchange, which "
    "then relates back to turn 1."
)

PROMPT = """{frame}

Turn 1 (A): {u1}
Turn 2 (B): {u2}
Turn 3 (A): {u3}

Answer briefly and directly: {probe}"""

SUMMARIZE_LEFT = (
    "Summarize this exchange in one short sentence, keeping who did/said "
    "what.\n\nTurn 1 (A): {u1}\nTurn 2 (B): {u2}")
SUMMARIZE_RIGHT = (
    "Summarize this exchange in one short sentence, keeping who did/said "
    "what.\n\nTurn 1 (B): {u2}\nTurn 2 (A): {u3}")

# summary induction: the pair is composed into a literal summary, then the
# probe is answered with the summary in context (μ is operationalised as the
# summarisation step, not as framing text).
SUMMARY_LEFT_PROMPT = """Earlier in this dialogue, this exchange happened:
Turn 1 (A): {u1}
Turn 2 (B): {u2}
Summary of that exchange: {summary}

Speaker A then says: {u3}

Answer briefly and directly: {probe}"""
SUMMARY_RIGHT_PROMPT = """This dialogue began with:
Turn 1 (A): {u1}

Then the following exchange happened:
Turn 2 (B): {u2}
Turn 3 (A): {u3}
Summary of that later exchange: {summary}

Answer briefly and directly: {probe}"""


def load_items():
    items = []
    for cat in CATEGORIES:
        path = HERE / f"{cat}.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def call_model(base_url, api_key, model, prompt, max_tokens=128, retries=8):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max(max_tokens, 256),
                    "temperature": 0.0,
                }).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                body = json.loads(r.read())
            msg = body["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            if not text:  # glm-style reasoning-only reply
                rc = (msg.get("reasoning_content") or "").strip().splitlines()
                text = rc[-1] if rc else ""
            return text.strip()
        except Exception as e:
            last = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"model call failed after {retries} attempts: {last}")


def norm(s):
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c == " ").split())


def heuristic_match(response, answer):
    """Normalised containment in either direction (short answers)."""
    r, a = norm(response), norm(answer)
    return bool(a) and (a in r or r in a)


def judge_match(base_url, api_key, judge_model, response, answer):
    prompt = (
        "Does the RESPONSE express the same content as the REFERENCE answer? "
        "Reply with exactly YES or NO.\n\nRESPONSE: " + response +
        "\n\nREFERENCE: " + answer)
    out = call_model(base_url, api_key, judge_model, prompt, max_tokens=8)
    return out.strip().upper().startswith("YES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", required=True)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--induction", choices=["framing", "summary"],
                    default="framing",
                    help="framing = grouping stated in text; summary = the "
                         "pair is composed into a literal summary first")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts, call no API")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        sys.exit(f"missing API key env {args.api_key_env}")

    items = load_items()[: args.limit]
    if args.judge_model:
        def matcher(resp, ans):
            return judge_match(args.base_url, api_key, args.judge_model,
                               resp, ans)
    else:
        matcher = heuristic_match

    results, flips = [], 0
    scored = 0
    for it in items:
        if args.dry_run:
            p_left = PROMPT.format(frame=LEFT_FRAME, u1=it["u1"], u2=it["u2"],
                                   u3=it["u3"], probe=it["probe_question"])
            print(f"== {it['id']} LEFT PROMPT ==\n{p_left}\n")
            continue
        if args.induction == "summary":
            s_left = call_model(args.base_url, api_key, args.model,
                                SUMMARIZE_LEFT.format(u1=it["u1"], u2=it["u2"]))
            time.sleep(0.3)
            s_right = call_model(args.base_url, api_key, args.model,
                                 SUMMARIZE_RIGHT.format(u2=it["u2"], u3=it["u3"]))
            time.sleep(0.3)
            p_left = SUMMARY_LEFT_PROMPT.format(
                u1=it["u1"], u2=it["u2"], summary=s_left, u3=it["u3"],
                probe=it["probe_question"])
            p_right = SUMMARY_RIGHT_PROMPT.format(
                u1=it["u1"], u2=it["u2"], u3=it["u3"], summary=s_right,
                probe=it["probe_question"])
        else:
            p_left = PROMPT.format(frame=LEFT_FRAME, u1=it["u1"], u2=it["u2"],
                                   u3=it["u3"], probe=it["probe_question"])
            p_right = PROMPT.format(frame=RIGHT_FRAME, u1=it["u1"], u2=it["u2"],
                                    u3=it["u3"], probe=it["probe_question"])
        if args.dry_run:
            print(f"== {it['id']} LEFT PROMPT ==\n{p_left}\n")
            continue
        r_left = call_model(args.base_url, api_key, args.model, p_left)
        time.sleep(0.3)
        r_right = call_model(args.base_url, api_key, args.model, p_right)
        time.sleep(0.3)
        m_l = matcher(r_left, it["answer_left"])
        m_r = matcher(r_right, it["answer_right"])
        flipped = norm(r_left) != norm(r_right)
        # a "clean" flip: each induction matches its own reading's answer
        clean = m_l and m_r
        flips += int(flipped)
        scored += int(clean)
        results.append({**it, "response_left": r_left,
                        "response_right": r_right, "flip": flipped,
                        "clean_flip": clean})
        print(f"{it['id']}: flip={flipped} clean={clean} "
              f"gold={it['gold_human']}")

    if args.dry_run:
        return
    n = len(results)
    by_gold = {}
    for r in results:
        by_gold.setdefault(r["gold_human"], []).append(r)
    summary = {
        "model": args.model,
        "n_items": n,
        "flip_rate": flips / max(n, 1),
        "clean_flip_rate": scored / max(n, 1),
        "by_gold": {g: {"n": len(v),
                        "flip_rate": sum(x["flip"] for x in v) / len(v),
                        "clean_flip_rate": sum(x["clean_flip"] for x in v) / len(v)}
                    for g, v in by_gold.items()},
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
        print(f"written: {out}")


if __name__ == "__main__":
    main()
