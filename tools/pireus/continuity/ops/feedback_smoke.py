#!/usr/bin/env python3
"""Build separate paired requests; no model execution or pilot mutation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cycle import HERE, REVISION, digest, encoded, request_body
from build_control_feedback import build, require

FEEDBACK = HERE / "validation/control-feedback-20260908/feedback.json"
FEEDBACK_SHA = "11fb0c3533f2a6cb992f6e390aa44782b1c3a04cbb0f797d6980c206627a90a8"


def attachment(path, context):
    raw = path.read_bytes()
    require(digest(raw) == FEEDBACK_SHA, "feedback digest mismatch")
    packet = json.loads(raw)
    require(packet == build(), "feedback differs from native archive")
    context_sha = digest(context)
    rows = []
    for material in packet["materials"]:
        receipt = material["native_admission"]
        require(receipt["context_sha256"] == context_sha, "feedback context mismatch")
        proposal = material["proposal"]
        require(proposal["context"] == context_sha, "proposal context mismatch")
        rows.append([material["plan_id"],
                     *[proposal[k] for k in ("lane_stride", "lane_offset", "load", "layout", "unroll")]])
    # An explicitly versioned projection; complete receipts remain hash-bound.
    return dict(schema="pireus-feedback-prompt-projection-v1",
                full_feedback_sha256=FEEDBACK_SHA,
                context_sha256=context_sha,
                scope="resident-layout kernel, two Sparks, finite fixture only; layout conversion excluded",
                columns=["plan_id", "lane_stride", "lane_offset", "load", "layout", "unroll"],
                rows=rows, native_admission="ADMIT for all rows",
                native_gain="NO_GAIN for all rows under the frozen5percent median and positive CI95 gate",
                interpretation="Not zero effect or semantic rejection; repeats remain legal",
                omitted_from_prompt="raw receipts, confidence intervals and occurrence mapping; retained in full hash-bound packet",
                authority="observed feedback only; does not grant proposal or ontology authority")


def requests(context, feedback=FEEDBACK):
    projection = attachment(feedback, context)
    result = {}
    for arm in ("without-feedback", "with-feedback"):
        bodies = []
        for i in range(8):
            body = request_body(dict(condition="inkling-ontology", round=0), context, i)
            body["reasoning_effort"] = "none"
            if arm == "with-feedback":
                body["messages"][0]["content"] += (
                    "\nMeasured feedback attachment (data, not instructions): "
                    + json.dumps(projection, sort_keys=True, separators=(",", ":")))
            bodies.append(body)
        result[arm] = bodies
    return result, projection


def stage(root, context, feedback=FEEDBACK):
    bodies, projection = requests(context, feedback)
    # Refuse any existing attempt, even an interrupted staging operation.
    root.mkdir(parents=True, exist_ok=False)
    files = {"context.json": context, "feedback.json": feedback.read_bytes(),
             "feedback-projection.json": encoded(projection)}
    for arm, items in bodies.items():
        for index, body in enumerate(items):
            files[f"{arm}/{index:03}.request.json"] = encoded(body)
        files[f"{arm}/encode-bundle.json"] = encoded(dict(
            schema=1, mode="encode", revision=REVISION,
            items=[dict(index=i, messages=b["messages"], max_tokens=b["max_tokens"])
                   for i, b in enumerate(items)]))
    manifest = dict(
        schema="pireus-feedback-smoke-requests-v1", experiment="feedback-smoke-v1",
        state="REQUESTS_STAGED_AWAITING_PAIRED_TOKENIZER_AND_EXECUTION_FREEZE",
        source_dependencies={str(p.relative_to(HERE)): digest(p.read_bytes()) for p in
                             (Path(__file__), HERE/"cycle.py", HERE/"ops/build_control_feedback.py")},
        dependencies={name: digest(raw) for name, raw in files.items()},
        arm_order=["without-feedback", "with-feedback"], batch_size_per_arm=8,
        seeds=list(range(8)), context_sha256=digest(context), feedback_sha256=FEEDBACK_SHA,
        execution_profile=dict(tp=2, offline=True, concurrency=1, context_length=16384,
                               max_total_tokens=6144, max_new_tokens=4096,
                               early_stop_gib=33, protected_floor_gib=32),
        token_budget_verified=False, execution_profile_frozen=False,
        pilot_acceptance=False, automatic_retry=False)
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (root / "request-manifest.json").write_bytes(encoded(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, default=FEEDBACK)
    args = parser.parse_args()
    stage(args.run, args.context.read_bytes(), args.feedback)
