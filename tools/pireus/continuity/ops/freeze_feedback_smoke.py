#!/usr/bin/env python3
"""Freeze exact inference inputs/runtime; CI acceptance is a separate receipt."""
import argparse
import json
from pathlib import Path
import subprocess
from feedback_smoke import HERE, encoded, digest
from qualify_feedback_tokens import qualify, REQUESTS
from build_control_feedback import require

TOKENS = HERE / "validation/feedback-smoke-tokenizer-11965"
RUNTIME_SHA = "fa4fb62f1f8a5d57a40374ef4ad839718caa171b902fb115cc85247191ca8cc0"


def freeze(root):
    evidence = qualify(TOKENS)
    require(evidence == json.loads((TOKENS / "qualification.json").read_bytes()),
            "token qualification differs")
    runtime = {p.name:p.read_bytes() for p in sorted((HERE / "runtime").iterdir()) if p.is_file()}
    require(digest(runtime["offline_generate.py"]) == RUNTIME_SHA, "uninstrumented runtime mismatch")
    repo = HERE.parents[2]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(HERE)], cwd=repo, text=True)
    require(not dirty.strip(), "commit continuity source before freezing")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    request_manifest = json.loads((REQUESTS / "request-manifest.json").read_bytes())
    tokens = json.loads((TOKENS / "rank-0.json").read_bytes())["items"]
    files = {}
    for arm_id, arm in enumerate(request_manifest["arm_order"]):
        items = []
        for i in range(8):
            body = json.loads((REQUESTS / arm / f"{i:03}.request.json").read_bytes())
            token = tokens[arm_id * 8 + i]
            item = dict(index=i, input_ids=token["input_ids"], stop_token_ids=token["stop_token_ids"],
                        temperature=body["temperature"], seed=body["seed"],
                        max_new_tokens=body["max_tokens"])
            items.append(item)
            files[f"{arm}/{i:03}.token.request.json"] = encoded(item)
        arm_manifest = dict(experiment="feedback-smoke-v1", arm=arm, batch_size=8,
                            source_commit=source, request_manifest_sha256=digest(
                                (REQUESTS / "request-manifest.json").read_bytes()),
                            qualification_sha256=digest((TOKENS / "qualification.json").read_bytes()))
        files[f"{arm}/manifest.json"] = encoded(arm_manifest)
        files[f"{arm}/offline-bundle.json"] = encoded(dict(
            schema=1, mode="offline-generate", revision=json.loads(runtime["runtime-lock.json"])["model"]["revision"],
            manifest_sha256=digest(encoded(arm_manifest)), items=items))
    freeze_spec = dict(
        schema="pireus-feedback-execution-freeze-v1", experiment="feedback-smoke-v1",
        source_commit=source, arm_order=request_manifest["arm_order"], batch_size_per_arm=8,
        request_manifest_sha256=digest((REQUESTS/"request-manifest.json").read_bytes()),
        token_qualification_sha256=digest((TOKENS/"qualification.json").read_bytes()),
        runtime_sha256={name:digest(raw) for name, raw in runtime.items()},
        files_sha256={name:digest(raw) for name, raw in files.items()},
        runtime_snapshot_strategy="git show source_commit:tools/pireus/continuity/runtime/PATH; verify each hash",
        required_execution_profile=dict(
            scope="frozen-offline-canary", tp_size=2, context_length=16384,
            max_total_tokens=6144, actual_full_tokens=6144, actual_swa_tokens=896,
            swa_full_tokens_ratio=0.15, page_size=128, max_running_requests=1,
            max_new_tokens=4096, native_host_floor_gib=32, early_stop_gib=33,
            embedding_placement="file-backed-cpu", lm_head_placement="file-backed-gpu-tiles",
            lm_head_tile_rows=4096, lm_head_hidden_rows=1, inductor_compile_threads=1,
            collective_backend="existing-pynccl", http_serving=False),
        model_image_lock=json.loads(runtime["runtime-lock.json"]),
        required_source_checks=["CI Decision", "transport-and-archive",
                                "Archived script custody (no runtime replay)"],
        ci_acceptance=False, execution_authorized_by_freeze=False,
        execution_profile_frozen=True, inference_completed=False,
        pilot_acceptance=False, automatic_retry=False,
        original_pilot_preservation="1/9 cells,32/288; failed11956 not resumed")
    # Source, profile and inputs are fixed. A later CI receipt must match source_commit.
    root.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    (root / "execution-freeze.json").write_bytes(encoded(freeze_spec))
    return freeze_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.run)
