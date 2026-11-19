#!/usr/bin/env python3
"""Stage a pinned ARM64 OCI layout without executing image contents."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

REPOSITORY = "lmsysorg/sglang"
MANIFEST = "sha256:bbedab8cbf2d209b00f48f1e96ef4e9b638b98771477fa14e0e70d62679f383b"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()

def token():
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPOSITORY}:pull"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["token"]

def stage(root, desc):
    algo, digest = desc["digest"].split(":")
    if algo != "sha256" or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("unsupported digest")
    out = root / "blobs" / "sha256" / digest
    if out.is_file():
        if out.stat().st_size != desc["size"] or sha256(out) != digest:
            raise ValueError("completed blob failed integrity")
        return
    part = out.with_suffix(".partial")
    offset = part.stat().st_size if part.exists() else 0
    if offset > desc["size"]:
        raise ValueError("oversized partial blob")
    if offset < desc["size"]:
        headers = {"Authorization": "Bearer " + token()}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{REPOSITORY}/blobs/sha256:{digest}", headers=headers)
        with urllib.request.urlopen(req, timeout=180) as r:
            if offset and (r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("resume range was not honored")
            with part.open("ab" if offset else "wb") as stream:
                for block in iter(lambda: r.read(8 * 1024**2), b""):
                    stream.write(block)
                    offset += len(block)
                    if offset > desc["size"]:
                        raise ValueError("oversized response")
                stream.flush()
                os.fsync(stream.fileno())
    if part.stat().st_size != desc["size"] or sha256(part) != digest:
        raise ValueError("blob checksum mismatch")
    os.replace(part, out)
    print(f"STAGED {digest} {desc['size']}", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args()
    root = args.destination
    (root / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
    raw = Path(__file__).with_name("inkling-oci-manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST.split(":")[1]:
        raise ValueError("manifest is not pinned ARM64 descriptor")
    manifest = json.loads(raw)
    descriptors = [manifest["config"], *manifest["layers"]]
    if shutil.disk_usage(root).free < sum(x["size"] for x in descriptors) + 16 * 1024**3:
        raise SystemExit("insufficient disk space including reserve")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda d: stage(root, d), descriptors))
    config = json.loads((root / "blobs" / "sha256" / manifest["config"]["digest"].split(":")[1]).read_text())
    if config["architecture"] != "arm64" or config["os"] != "linux":
        raise ValueError("image architecture mismatch")
    (root / "blobs" / "sha256" / MANIFEST.split(":")[1]).write_bytes(raw)
    (root / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}\n')
    index = {"schemaVersion": 2, "manifests": [{
        "mediaType": manifest["mediaType"], "digest": MANIFEST, "size": len(raw),
        "platform": {"architecture": "arm64", "os": "linux"},
        "annotations": {"org.opencontainers.image.ref.name": "inkling-spark"}}]}
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print("OCI_LAYOUT_COMPLETE " + str(root), flush=True)

if __name__ == "__main__":
    main()
