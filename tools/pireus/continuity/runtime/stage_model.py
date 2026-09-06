#!/usr/bin/env python3
"""Stage an immutable public HF snapshot. Transport/custody only; no semantics."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

MODEL = "thinkingmachines/Inkling-Small-NVFP4"
REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"

def digest_file(path, git_blob=False):
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    lfs = entry.get("lfs")
    return digest_file(path, not bool(lfs)) == (lfs["sha256"] if lfs else entry["blobId"])

def fetch(root, entry):
    name = entry["rfilename"]
    # Repository paths remain data, never shell fragments.
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe manifest path")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if verify(path, entry):
        print(f"VERIFIED {name}", flush=True)
        return {"path": name, "size": entry["size"], "sha256": digest_file(path)}
    if path.exists():
        raise ValueError(f"existing completed file failed integrity: {name}")
    partial = path.with_name(path.name + ".partial")
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["size"]:
                raise ValueError(f"oversized partial: {name}")
            if offset < entry["size"]:
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=180) as response:
                    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise ValueError("server did not honor resume range")
                    with partial.open("ab" if offset else "wb") as out:
                        next_progress = offset + 1024**3
                        while True:
                            block = response.read(8 * 1024 * 1024)
                            if not block:
                                break
                            out.write(block)
                            offset += len(block)
                            if offset > entry["size"]:
                                raise ValueError("response exceeds frozen size")
                            if offset >= next_progress:
                                print(f"PROGRESS {name} {offset}/{entry['size']}", flush=True)
                                next_progress += 1024**3
                        out.flush()
                        os.fsync(out.fileno())
            if not verify(partial, entry):
                raise ValueError(f"snapshot hash/size mismatch: {name}")
            os.replace(partial, path)
            result = {"path": name, "size": entry["size"], "sha256": digest_file(path)}
            print(f"STAGED {name} {result['sha256']}", flush=True)
            return result
        except ValueError:
            raise
        except Exception as exc:
            print(f"RETRY {name} attempt={attempt+1} error={type(exc).__name__}", flush=True)
            if attempt == 3:
                raise
            time.sleep(min(30, 2 ** attempt))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("inkling-files.json").read_text())
    entries = [x for x in manifest if not args.metadata_only or x["size"] < 100_000_000]
    root = args.destination
    root.mkdir(parents=True, exist_ok=True)
    outstanding = sum(max(0, x["size"] - (root / x["rfilename"]).stat().st_size) if (root / x["rfilename"]).is_file() else x["size"] for x in entries)
    if shutil.disk_usage(root).free < outstanding + 16 * 1024**3:
        raise SystemExit("insufficient free space including 16 GiB reserve")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda item: fetch(root, item), entries))
    receipt = {"schema": "pireus-model-staging-v1", "model": MODEL, "revision": REVISION,
               "complete_snapshot": not args.metadata_only, "semantic_authority": False,
               "files": sorted(receipts, key=lambda x: x["path"])}
    target = root / ("metadata-staging.json" if args.metadata_only else "snapshot-staging.json")
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, target)
    print(f"STAGING_COMPLETE receipt={target}", flush=True)

if __name__ == "__main__":
    main()
