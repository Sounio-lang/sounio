#!/usr/bin/env python3
"""Custody-only SGLang token-ID transport; admission remains native Sounio."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import urllib.request
import urllib.parse
from cycle import HERE, REVISION, atomic, digest, encoded, event, request_body, verify

def save(root, name, value, stage):
    path = root / name
    atomic(path, encoded(value))
    event(root, stage, path)
    return path

def pair(root, mode, receipts, manifest):
    bundle = root / (mode + "-bundle.json")
    expected = digest(bundle.read_bytes())
    values = [json.loads(p.read_bytes()) for p in receipts]
    if len(values) != 2 or {v["rank"] for v in values} != {"0", "1"}:
        raise ValueError("both tokenizer ranks required")
    if len({v["job"] for v in values}) != 1:
        raise ValueError("different tokenizer jobs")
    for v in values:
        if (v["revision"] != REVISION or v["mode"] != mode or v["input_sha256"] != expected
            or v["helper_sha256"] != digest((HERE / "runtime/tokenizer_transport.py").read_bytes())):
            raise ValueError("tokenizer custody mismatch")
        if [x["index"] for x in v["items"]] != list(range(manifest["budget"])):
            raise ValueError("tokenizer item set mismatch")
    a, b = [{k: v for k, v in x.items() if k not in ("rank",)} for x in values]
    if a != b:
        raise ValueError("two-node tokenizer results differ")
    for value in values:
        save(root, mode + "-rank-" + value["rank"] + ".json", value, mode + "-receipt")
    return values[0]

def pack_encode(root, manifest):
    if manifest["condition"] == "deterministic":
        raise ValueError("token transport requires a real model condition")
    context = (root / "context.json").read_bytes()
    items = []
    for index in range(manifest["budget"]):
        body = request_body(manifest, context, index)
        body["reasoning_effort"] = "none"
        save(root, "%03d.request.json" % index, body, "request-specification")
        items.append(dict(index=index, messages=body["messages"], max_tokens=body["max_tokens"]))
    save(root, "encode-bundle.json", dict(schema=1, mode="encode", revision=REVISION, items=items), "encode-bundle")

def job_is_owned(job):
    if not job or not job.isdecimal():
        raise ValueError("explicit serving job required")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(x.split("=", 1) for x in raw.split() if "=" in x)
    if fields.get("JobState") != "RUNNING" or fields.get("JobName") != "pireus-inkling-serve-token-ids":
        raise ValueError("expected running token-ID serving job")
    if fields.get("NumNodes") != "2" or "spark" not in fields.get("NodeList", ""):
        raise ValueError("serving job is not on the pair")
    return raw

def issue_once(root, request, response, endpoint, body):
    if request.exists():
        raise RuntimeError("ambiguous interrupted generation; never automatically replay: " + str(request))
    save(root, request.name, body, "token-request")
    req = urllib.request.Request(endpoint.rstrip("/") + "/generate",
                                 data=encoded(body), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as reply:
        raw = reply.read()
    atomic(response, raw)

def generate(root, manifest, endpoint, job):
    if manifest.get("transport") != "sglang-token-ids":
        raise ValueError("HTTP generation cannot service an offline manifest")
    pair_receipts = [root / ("encode-rank-" + str(rank) + ".json") for rank in (0, 1)]
    receipt = pair(root, "encode", pair_receipts, manifest)
    if not endpoint:
        raise ValueError("endpoint required")
    address = urllib.parse.urlparse(endpoint)
    pods = json.loads(subprocess.check_output(
        ["kubectl", "-n", "slurm-pilot", "get", "pods", "-o", "json"], text=True))["items"]
    rank0 = [p for p in pods if p["spec"].get("nodeName") == "spark-3c59"
             and p["metadata"]["name"].startswith("slurm-pilot-worker-spark-")
             and p["status"].get("phase") == "Running"]
    if (len(rank0) != 1 or address.scheme != "http" or address.port != 30000
        or address.hostname != rank0[0]["status"]["podIP"]):
        raise ValueError("endpoint must be the current rank-zero worker port")
    from runtime.preflight import check_pair
    preflight = check_pair()
    state = job_is_owned(job)
    with urllib.request.urlopen(endpoint.rstrip("/") + "/get_model_info", timeout=30) as reply:
        info_raw = reply.read()
    info = json.loads(info_raw)
    if REVISION not in str(info.get("model_path", "")):
        raise ValueError("endpoint checkpoint mismatch")
    save(root, "serving-binding.json",
         dict(job=job, endpoint=endpoint, model_info=info, job_observation=state,
              host_preflight=preflight), "serving-binding")
    for item in receipt["items"]:
        i = item["index"]
        logical = json.loads((root / ("%03d.request.json" % i)).read_text())
        request = root / ("%03d.token.request.json" % i)
        response = root / ("%03d.token.response.json" % i)
        if not response.exists():
            job_is_owned(job)
            body = dict(input_ids=item["input_ids"], sampling_params=dict(
                max_new_tokens=logical["max_tokens"], temperature=logical["temperature"],
                sampling_seed=logical["seed"], stop_token_ids=item["stop_token_ids"]),
                stream=False)
            issue_once(root, request, response, endpoint, body)
        event(root, "token-response", response)
        result = json.loads(response.read_bytes())
        ids = result.get("output_ids")
        if not isinstance(ids, list) or not all(type(t) is int and t >= 0 for t in ids):
            raise ValueError("missing output token IDs; preserve original response")
        print(json.dumps(dict(stage="TOKEN_RESPONSE_SAVED", index=i, output_tokens=len(ids))), flush=True)

def pack_decode(root, manifest):
    items = []
    for i in range(manifest["budget"]):
        p = root / ("%03d.token.response.json" % i)
        raw = p.read_bytes()
        value = json.loads(raw)
        items.append(dict(index=i, output_ids=value["output_ids"], token_response_sha256=digest(raw)))
    save(root, "decode-bundle.json", dict(schema=1, mode="decode", revision=REVISION, items=items), "decode-bundle")

def finalize(root, manifest, receipts):
    receipt = pair(root, "decode", receipts, manifest)
    for item in receipt["items"]:
        i = item["index"]
        if item["token_response_sha256"] != digest((root / ("%03d.token.response.json" % i)).read_bytes()):
            raise ValueError("decoded response identity mismatch")
        if not isinstance(item["text"], str):
            raise ValueError("non-textual decode")
        save(root, "%03d.decoded.json" % i, item, "decoded-response")
        proposal = root / ("%03d.proposal.json" % i)
        # Exact tokenizer output: no JSON repair, extraction, or semantic filtering.
        atomic(proposal, item["text"].encode())
        event(root, "generate", proposal)

def pack_offline(root, manifest):
    if manifest.get("transport") != "sglang-offline-token-ids" or manifest["budget"] not in (8, 32):
        raise ValueError("offline path requires its own frozen 8- or 32-proposal manifest")
    enc = pair(root, "encode", [root / ("encode-rank-" + str(i) + ".json") for i in (0, 1)], manifest)
    items = []
    for item in enc["items"]:
        logical = json.loads((root / ("%03d.request.json" % item["index"])).read_bytes())
        value = dict(index=item["index"], input_ids=item["input_ids"],
                     stop_token_ids=item["stop_token_ids"], temperature=logical["temperature"],
                     seed=logical["seed"], max_new_tokens=logical["max_tokens"])
        items.append(value)
        save(root, "%03d.token.request.json" % item["index"], value, "offline-token-request")
    save(root, "offline-bundle.json", dict(schema=1, mode="offline-generate", revision=REVISION,
         manifest_sha256=digest((root / "manifest.json").read_bytes()), items=items), "offline-bundle")

def accept_offline(root, manifest, worker_dir):
    if manifest.get("transport") != "sglang-offline-token-ids" or worker_dir is None:
        raise ValueError("offline worker evidence required")
    receipts = [json.loads((worker_dir / ("rank-" + str(i) + "-complete.json")).read_bytes()) for i in (0, 1)]
    expected = digest((root / "offline-bundle.json").read_bytes())
    for rank, receipt in enumerate(receipts):
        if (receipt["rank"] != str(rank) or receipt["input_sha256"] != expected
            or receipt["revision"] != REVISION
            or receipt["helper_sha256"] != digest((HERE / "runtime/offline_generate.py").read_bytes())
            or receipt["model_loaded"] is not True or len(receipt["results"]) != manifest["budget"]):
            raise ValueError("offline completion receipt identity")
    for receipt in receipts:
        profile = receipt.get("execution_profile", {})
        expected_profile = dict(schema=1, scope=("frozen-offline-pilot-batch" if manifest["budget"] == 32 else "frozen-offline-canary"),
            transport="sglang-offline-token-ids", tp_size=2, jit_cache_storage="local-ssd", inductor_compile_threads=1, embedding_placement="file-backed-cpu", lm_head_placement="file-backed-gpu-tiles", lm_head_tile_rows=4096, lm_head_hidden_rows=1, lm_head_numerical_scope="qualified-controls-only", collective_backend="existing-pynccl", context_length=16384,
            max_total_tokens=6144, actual_full_tokens=6144, swa_full_tokens_ratio=0.15,
            page_size=128, max_running_requests=1, max_new_tokens=4096,
            native_host_floor_gib=32, early_stop_gib=33,
            http_serving=False, general_16k_inference_accepted=False)
        if (any(profile.get(k) != v for k,v in expected_profile.items())
            or type(profile.get("actual_swa_tokens")) is not int
            or profile["actual_swa_tokens"] < 639):
            raise ValueError("offline execution profile identity")
    embedding_lock = json.loads((HERE/"runtime/embedding-offload-lock.json").read_bytes())
    for rank, receipt in enumerate(receipts):
        storage = receipt.get("embedding_storage", {})
        if (embedding_lock["revision"] != REVISION
            or storage.get("rank") != str(rank) or storage.get("job") != receipt["job"]
            or storage.get("placement") != "file-backed-cpu"
            or storage.get("dtype") != embedding_lock["dtype"]
            or storage.get("shape") != embedding_lock["shape"]
            or storage.get("bytes") != embedding_lock["bytes"]
            or storage.get("source_gpu_sha256") != embedding_lock["rank_sha256"][str(rank)]
            or storage.get("file_sha256") != embedding_lock["rank_sha256"][str(rank)]
            or storage.get("checkpoint_precision_changed") is not False
            or storage.get("helper_sha256") != digest((HERE/"runtime/offload_embedding.py").read_bytes())):
            raise ValueError("offline embedding storage identity")
    lm_head_lock = json.loads((HERE/"runtime/lm-head-offload-lock.json").read_bytes())
    for rank, receipt in enumerate(receipts):
        storage = receipt.get("lm_head_storage", {})
        if (lm_head_lock["revision"] != REVISION
            or storage.get("rank") != str(rank) or storage.get("job") != receipt["job"]
            or storage.get("placement") != "file-backed-gpu-tiles"
            or storage.get("tile_rows") != 4096 or storage.get("hidden_rows") != 1
            or storage.get("dtype") != lm_head_lock["dtype"]
            or storage.get("shape") != lm_head_lock["shape"]
            or storage.get("bytes") != lm_head_lock["bytes"]
            or storage.get("source_gpu_sha256") != lm_head_lock["rank_sha256"][str(rank)]
            or storage.get("file_sha256") != lm_head_lock["rank_sha256"][str(rank)]
            or storage.get("checkpoint_precision_changed") is not False
            or storage.get("helper_sha256") != digest((HERE/"runtime/offload_lm_head.py").read_bytes())):
            raise ValueError("offline LM-head storage identity")
    comparable = [{k: v for k, v in r.items() if k not in ("rank", "embedding_storage", "lm_head_storage")} for r in receipts]
    if comparable[0] != comparable[1]:
        raise ValueError("offline two-rank receipt disagreement")
    for i in range(manifest["budget"]):
        raw = [(worker_dir / ("rank-%d-%03d.json" % (rank, i))).read_bytes() for rank in (0, 1)]
        if raw[0] != raw[1]:
            raise ValueError("offline two-rank token response disagreement")
        response = json.loads(raw[0])
        if (response["index"] != i or response["job"] != receipts[0]["job"]
            or response["input_sha256"] != expected or response["revision"] != REVISION
            or response["transport"] != "sglang-offline-token-ids"
            or response.get("execution_profile") != receipts[0]["execution_profile"]):
            raise ValueError("offline token response identity")
        for receipt in receipts:
            matches = [x for x in receipt["results"] if x["index"] == i]
            if len(matches) != 1 or matches[0]["response_sha256"] != digest(raw[0]):
                raise ValueError("offline response hash mismatch")
        path = root / ("%03d.token.response.json" % i)
        atomic(path, raw[0])
        event(root, "offline-token-response", path)
    for rank, receipt in enumerate(receipts):
        save(root, "offline-rank-" + str(rank) + "-complete.json", receipt, "offline-completion")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["pack-encode", "accept-encode", "pack-offline", "accept-offline", "generate", "pack-decode", "finalize"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--receipts", type=Path, nargs=2)
    ap.add_argument("--worker-dir", type=Path)
    ap.add_argument("--endpoint")
    ap.add_argument("--serving-job")
    args = ap.parse_args()
    with (args.run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify(args.run)
        if manifest.get("transport") not in ("sglang-token-ids", "sglang-offline-token-ids"):
            raise ValueError("wrong frozen transport")
        if args.command == "pack-encode":
            pack_encode(args.run, manifest)
        elif args.command == "accept-encode":
            pair(args.run, "encode", args.receipts or [], manifest)
        elif args.command == "pack-offline":
            pack_offline(args.run, manifest)
        elif args.command == "accept-offline":
            accept_offline(args.run, manifest, args.worker_dir)
        elif args.command == "generate":
            generate(args.run, manifest, args.endpoint, args.serving_job)
        elif args.command == "pack-decode":
            pack_decode(args.run, manifest)
        else:
            finalize(args.run, manifest, args.receipts or [])

if __name__ == "__main__":
    main()
