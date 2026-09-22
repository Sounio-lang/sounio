#!/usr/bin/env python3
"""Export website source inputs from a Git commit, or verify an existing export.

This exports evidence as recorded at that commit, not a fresh validation claim.
No compiler is provided: bin/souc is non-executable source data for status pages.
Consumers must pin the archive SHA-256 separately from the embedded inventory.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

ROOTS = (
    'docs', 'scripts/docs/governance_registry.mjs', 'README.md', 'CHANGELOG.md',
    'bin/souc', 'bootstrap/stage0.c',
    'artifacts/stdlib/stdlib_reliability_status.v1.json',
    'artifacts/stdlib/stdlib_science_pipeline_status.v1.json',
    'artifacts/stdlib/stdlib_hyper_execution_status.v1.json',
    'artifacts/omega/native_backend_v2_gate.v1.json',
    'artifacts/omega/selfhost_verification_report.v1.json',
    'artifacts/omega/lsp_smoke_status.v1.json',
    'artifacts/omega/gpu_runtime_attest_gate.v1.json',
    'artifacts/omega/bootstrap_full_gate_status.v1.json',
    'website/public/assets/generated/render', 'examples/render',
)
DIR_ROOTS = ('docs', 'website/public/assets/generated/render', 'examples/render')
FILE_ROOTS = tuple(root for root in ROOTS if root not in DIR_ROOTS)
MANIFEST = 'website-inputs-manifest.json'
# These tracked links contain a historical machine's absolute path. Map only
# this explicit inventory to blobs at the selected commit; never follow disk links.
PROOF_NAMES = ('SounioCausality', 'SounioEffects', 'SounioEpistemic', 'SounioFormal',
               'SounioLinear', 'SounioPreservation', 'SounioProgress', 'SounioRowPoly',
               'SounioSemantics', 'SounioSubstitution', 'SounioTyping', 'SounioUnits')
PROOF_LINKS = {f'docs/papers/main/epistemic-types/proofs/{name}.lean':
               f'formal/lean4/{name}.lean' for name in PROOF_NAMES}



def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name:
        raise ValueError(f'Invalid archive path: {name!r}')
    return name


def selected(name):
    return name in FILE_ROOTS or any(name.startswith(root + '/') for root in DIR_ROOTS)


def missing_inputs(names):
    missing = [root for root in FILE_ROOTS if root not in names]
    missing += [root for root in DIR_ROOTS if not any(n.startswith(root + '/') for n in names)]
    missing += [name for name in PROOF_LINKS if name not in names]
    return missing


def add_file(archive, name, data):
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = 0o644
    entry.mtime = 0
    archive.addfile(entry, io.BytesIO(data))


def export(repo, revision, output):
    commit = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}'],
        text=True).strip()
    raw = subprocess.check_output(['git', '-C', str(repo), 'archive', '--format=tar', commit, *ROOTS])
    files = {}
    origins = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as source:
        for entry in source:
            if entry.isdir():
                continue
            name = safe_path(entry.name)
            if not selected(name) or name in files:
                raise ValueError(f'Unsupported or duplicate input: {name}')
            if entry.issym() and name in PROOF_LINKS:
                origin = PROOF_LINKS[name]
                if entry.linkname != '/home/demetrios/work/sounio/' + origin:
                    raise ValueError(f'Unexpected historical proof link: {name}')
                listing = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', commit, '--', origin], text=True)
                if not listing.startswith('100644 blob '):
                    raise ValueError(f'Proof target is not a regular committed file: {origin}')
                files[name] = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + origin])
                origins[name] = origin
            elif entry.isfile() and name not in PROOF_LINKS:
                files[name] = source.extractfile(entry).read()
                origins[name] = name
            else:
                raise ValueError(f'Unsupported input member: {name}')
    missing = missing_inputs(files)
    if missing:
        raise ValueError(f'Missing selected inputs: {missing}')
    manifest = {
        'schema': 1, 'source_commit': commit,
        'purpose': 'Recorded website inputs; no compiler executable or fresh validation claim',
        'selection': list(ROOTS),
        'files': [{'path': n, 'source_path': origins[n], 'sha256': digest(data), 'bytes': len(data)} for n, data in sorted(files.items())],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + '.', delete=False) as dest:
            temporary = Path(dest.name)
            with gzip.GzipFile(fileobj=dest, mode='wb', mtime=0, filename='') as gz:
                with tarfile.open(fileobj=gz, mode='w') as archive:
                    for name, data in sorted(files.items()):
                        add_file(archive, name, data)
                    add_file(archive, MANIFEST, (json.dumps(manifest, indent=2) + '\n').encode())
        verify(temporary)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'archive': str(output), 'sha256': digest(output.read_bytes()),
            'source_commit': commit, 'files': len(files)}


def verify(path, expected_sha256=None):
    data = Path(path).read_bytes()
    actual = digest(data)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('Archive SHA-256 differs from pinned digest')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for entry in archive:
            name = safe_path(entry.name)
            if not entry.isfile() or name in files or entry.mode != 0o644:
                raise ValueError(f'Unsupported member, duplicate or mode: {name}')
            files[name] = archive.extractfile(entry).read()
    manifest = json.loads(files.pop(MANIFEST))
    if manifest.get('schema') != 1 or manifest.get('selection') != list(ROOTS):
        raise ValueError('Unknown website input schema or selection')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('source_commit', '')):
        raise ValueError('Invalid source commit')
    expected = set()
    for row in manifest['files']:
        name = safe_path(row['path'])
        if name in expected or not selected(name):
            raise ValueError(f'Duplicate or unselected inventory entry: {name}')
        if row.get('source_path') != PROOF_LINKS.get(name, name):
            raise ValueError(f'Unexpected inventory source: {name}')
        expected.add(name)
        payload = files.get(name)
        if payload is None or len(payload) != row['bytes'] or digest(payload) != row['sha256']:
            raise ValueError(f'Inventory mismatch: {name}')
    if set(files) != expected:
        raise ValueError('Archive contains files outside its inventory')
    missing = missing_inputs(expected)
    if missing:
        raise ValueError(f'Missing selected inputs: {missing}')
    return {'sha256': actual, 'source_commit': manifest['source_commit'], 'files': len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    create.add_argument('--ref', default='HEAD')
    create.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('archive', type=Path)
    check.add_argument('--sha256', required=True, help='Digest from the pinned release, not the archive itself')
    args = parser.parse_args()
    try:
        result = export(args.repo, args.ref, args.output) if args.command == 'create' else verify(args.archive, args.sha256)
    except (ValueError, KeyError, TypeError, OSError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'website inputs: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
