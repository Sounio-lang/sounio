#!/usr/bin/env python3
"""Verify paired token custody and physical cache budget before execution freeze."""
import argparse
import json
from pathlib import Path
from feedback_smoke import HERE, digest, encoded, requests
from build_control_feedback import require

REQUESTS = HERE / "validation/feedback-smoke-requests-20260908"


def qualify(root):
    manifest = json.loads((REQUESTS / "request-manifest.json").read_bytes())
    for name, expected in manifest["dependencies"].items():
        require(digest((REQUESTS / name).read_bytes()) == expected, "request dependency mismatch")
    for name, expected in manifest["source_dependencies"].items():
        require(digest((HERE / name).read_bytes()) == expected, "request source mismatch")
    arms, _ = requests((REQUESTS / "context.json").read_bytes())
    expected_items = []
    for arm in manifest["arm_order"]:
        for i, body in enumerate(arms[arm]):
            require((REQUESTS / arm / f"{i:03}.request.json").read_bytes() == encoded(body),
                    "request reconstruction mismatch")
            expected_items.append(dict(index=len(expected_items), messages=body["messages"],
                                       max_tokens=body["max_tokens"]))
    bundle = json.loads((root / "encode-bundle.json").read_bytes())
    require(bundle["items"] == expected_items, "token bundle differs from staged requests")
    rows = [json.loads((root / f"rank-{rank}.json").read_bytes()) for rank in range(2)]
    logs = []
    for line in (root / "launch.log").read_text().splitlines():
        try:
            logs.append(json.loads(line))
        except ValueError:
            pass
    for rank, receipt in enumerate(rows):
        require(receipt["rank"] == str(rank) and receipt["mode"] == "encode", "rank/mode mismatch")
        require(receipt["revision"] == bundle["revision"], "revision mismatch")
        require(receipt["input_sha256"] == digest((root / "encode-bundle.json").read_bytes()),
                "token input mismatch")
        require(receipt["helper_sha256"] == digest((HERE / "runtime/tokenizer_transport.py").read_bytes()),
                "tokenizer helper mismatch")
        witnesses = [x for x in logs if x.get("stage") == "TOKENIZER_TRANSPORT_PASS"
                     and x.get("job") == receipt["job"] and x.get("rank") == str(rank)]
        require(len(witnesses) == 1 and witnesses[0]["count"] == 16 and
                witnesses[0]["receipt_sha256"] == digest((root / f"rank-{rank}.json").read_bytes()),
                "tokenizer log custody mismatch")
    require({k:v for k,v in rows[0].items() if k != "rank"} ==
            {k:v for k,v in rows[1].items() if k != "rank"}, "paired tokenizer mismatch")
    require([x["index"] for x in rows[0]["items"]] == list(range(16)), "token coverage mismatch")
    counts = [len(x["input_ids"]) for x in rows[0]["items"]]
    require(all(0 < n and n + 4096 <= 6144 for n in counts), "physical cache budget exceeded")
    require((root / "exit-code").read_text().strip() == "0", "launcher not successful")
    accounting = (root / "accounting.txt").read_text().strip().split("|")
    require(len(accounting) == 7 and accounting[:4] ==
            [rows[0]["job"], "pireus-inkling-tokenize", "COMPLETED", "0:0"],
            "durable accounting mismatch")
    require(set(accounting[4].split(",")) ==
            {"gpuorangefs-multi-spark-3c59", "gpuorangefs-multi-spark-8e54"},
            "accounting pair mismatch")
    return dict(schema="pireus-feedback-token-qualification-v1", job=rows[0]["job"],
                request_count=16, input_tokens=counts, output_ceiling=4096,
                max_total_tokens=6144, token_budget_verified=True, paired_receipts_equal=True,
                request_manifest_sha256=digest((REQUESTS / "request-manifest.json").read_bytes()),
                source_files={name:digest((root/name).read_bytes()) for name in
                              ("encode-bundle.json", "rank-0.json", "rank-1.json", "launch.log",
                               "accounting.txt", "exit-code", "mapping.json", "source.json")},
                inference_completed=False, pilot_acceptance=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = qualify(args.root)
    with (args.root / "qualification.json").open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
