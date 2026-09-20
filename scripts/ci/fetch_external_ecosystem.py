#!/usr/bin/env python3
"""Fetch checksum-pinned ecosystem artifacts; never fall back to in-tree copies."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def fetch(destination, suffix=None):
    lock = Path(__file__).resolve().parents[2] / 'ecosystem/external-releases.json'
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for asset in json.loads(lock.read_text())['assets']:
        name = asset['name']
        if suffix is not None and not name.endswith(suffix):
            continue
        if Path(name).name != name:
            raise ValueError('Asset must have a flat filename')
        target = destination / name
        def valid(path):
            return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
        if not valid(target):
            with urllib.request.urlopen(asset['url'], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != asset['sha256']:
                raise ValueError('Checksum mismatch: ' + name)
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as staged:
                staged.write(data)
                staged_path = Path(staged.name)
            try:
                staged_path.replace(target)
            finally:
                staged_path.unlink(missing_ok=True)
        paths.append(str(target))
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--suffix', choices=['.whl', '.vsix'])
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination.resolve(), args.suffix)))
