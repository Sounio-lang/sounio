#!/usr/bin/env python3
"""One-GPU material observer. Output directories are keyed by UUID and SM, not hostname."""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
import numpy as np
import torch
from gpu_device_key import device_identity, marker_name, observation_dir
from gpu_probe import Driver, hex_frame
from paired_timing import BLOCKS, GRAPH_WARMUP, REPLAYS, VECTORS, row
from run_inputs import data_root, require_probe_inputs, require_timing_inputs


def version_text(value):
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")
    return str(value)


def visible_identity(expect_sm):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("this step must see exactly one GPU")
    props = torch.cuda.get_device_properties(0)
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = getattr(props, "name", None)
    try:
        return device_identity(props, name, expect_sm)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def write_marker(root, phase, identity, job):
    path = root / marker_name(phase, identity["sm"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "phase": phase, "job": job, "key_scope": "per-gpu", **identity,
    }, indent=2) + "\n")


def probe(root, identity, job):
    manifest, fixtures, _digest, _ptx = require_probe_inputs(root)
    rows = [json.loads(line) for line in fixtures.splitlines()]
    arrays = {key: np.array([r[key] for r in rows], dtype=np.int64) for key in ("a_bits", "b_bits", "output_bits")}
    driver = Driver()
    output = root / observation_dir("probe", identity["gpu_uuid"], identity["sm"] // 10, identity["sm"] % 10)
    output.mkdir(exist_ok=True)
    (output / "reference.hex").write_text(hex_frame(arrays["output_bits"]))
    for candidate in manifest["candidates"]:
        path = root / candidate["ptx"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["ptx_sha256"]:
            raise ValueError("ptx digest")
        module, function = driver.load(path)
        tensors = []
        for key in ("a_bits", "b_bits"):
            value = arrays[key].view(np.float64)
            if candidate["layout"] == 1:
                value = value.T
            tensors.append(torch.from_numpy(np.ascontiguousarray(value)).to("cuda"))
        out = torch.empty_like(tensors[0])
        driver.launch(function, *tensors, out, len(rows))
        torch.cuda.synchronize()
        raw = out.cpu().numpy()
        if candidate["layout"] == 1:
            raw = raw.T
        bits = np.ascontiguousarray(raw).view(np.int64)
        destination = output / (candidate["id"] + ".hex")
        destination.write_text(hex_frame(bits))
        print(json.dumps(dict(
            stage="MATERIAL_OBSERVATION", candidate=candidate["id"],
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            components=int(bits.size), gpu_uuid=identity["gpu_uuid_hex"],
            compute_capability=identity["compute_capability"], sm=identity["sm"],
            key_scope="per-gpu", job=job)), flush=True)
        driver.check(driver.lib.cuModuleUnload(module))
    (output / "inventory.json").write_text(json.dumps(dict(
        job=job, node=socket.gethostname(), cuda=version_text(torch.version.cuda),
        observation_only=True, key_scope="per-gpu", **identity), indent=2) + "\n")


def timing(root, identity, job):
    manifest, fixtures, _digest, _ptx = require_timing_inputs(root)
    if manifest["blocks"] != BLOCKS or manifest["vectors"] != VECTORS or manifest["replays"] != REPLAYS:
        raise ValueError("protocol")
    rows = [json.loads(line) for line in fixtures.splitlines()][256:288]
    inputs = {k: np.tile(np.array([r[k] for r in rows], dtype=np.int64), (512, 1)).view(np.float64)
              for k in ("a_bits", "b_bits")}
    driver = Driver()
    graphs = {}
    retained = []
    for item in manifest["controls"] + manifest["candidates"]:
        path = root / item["ptx"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["ptx_sha256"]:
            raise ValueError("ptx digest")
        module, function = driver.load(path)
        tensors = [torch.from_numpy(np.ascontiguousarray(v.T if item["layout"] else v)).to("cuda")
                   for v in inputs.values()]
        out = torch.empty_like(tensors[0])
        driver.launch(function, *tensors, out, VECTORS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPLAYS):
                driver.launch(function, *tensors, out, VECTORS)
        for _ in range(GRAPH_WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        graphs[item["id"]] = graph
        retained.append((module, function, tensors, out))
    output = root / observation_dir("timing", identity["gpu_uuid"], identity["sm"] // 10, identity["sm"] % 10)
    output.mkdir(exist_ok=True)
    for candidate in manifest["candidates"]:
        for control in manifest["controls"]:
            target = output / (candidate["id"] + "." + control["id"] + ".tsv")
            if target.exists():
                raise ValueError("timing trial exists; use a distinct frozen run")
            cg = graphs[candidate["id"]]
            bg = graphs[control["id"]]
            for _ in range(GRAPH_WARMUP):
                cg.replay()
                bg.replay()
            torch.cuda.synchronize()
            with target.open("w") as log:
                for block in range(BLOCKS):
                    durations = {}
                    order = [("control", bg), ("candidate", cg)] if block % 2 else [("candidate", cg), ("control", bg)]
                    for label, graph in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        durations[label] = round(start.elapsed_time(end) * 1000000)
                    log.write(row(block, durations["candidate"], durations["control"]))
                    log.flush()
                    os.fsync(log.fileno())
            print(json.dumps(dict(
                stage="PAIRED_TIMING_OBSERVED", candidate=candidate["id"], control=control["id"],
                gpu_uuid=identity["gpu_uuid_hex"], compute_capability=identity["compute_capability"],
                sm=identity["sm"], key_scope="per-gpu", job=job,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(), blocks=BLOCKS)), flush=True)
    (output / "inventory.json").write_text(json.dumps(dict(
        job=job, node=socket.gethostname(), cuda=version_text(torch.version.cuda), torch=str(torch.__version__),
        timing="CUDA events around 32 captured kernel launches", vectors=VECTORS,
        layout_conversion_included=False, observation_only=True, key_scope="per-gpu",
        **identity), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("probe", "timing"), required=True)
    parser.add_argument("--expect-sm", type=int, choices=(75, 86), required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Slurm allocation required")
    identity = visible_identity(args.expect_sm)
    job = os.environ["SLURM_JOB_ID"]
    data = data_root(args.run, args.expect_sm)
    if args.phase == "probe":
        probe(data, identity, job)
    else:
        timing(data, identity, job)
    write_marker(args.run, args.phase, identity, job)


if __name__ == "__main__":
    main()
