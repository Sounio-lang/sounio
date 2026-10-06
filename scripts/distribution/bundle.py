#!/usr/bin/env python3
"""Build/verify a relocatable Madaros distribution from a committed source tree.

The source/binary relationship is a publisher declaration, not independently
proved by hashing. Pass a reviewed build receipt; verify the archive's published
SHA256 before unpacking it. No compiler capability is inferred from packaging.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

SOURCE_PATHS = ('bin/souc', 'bin/madaros', 'scripts/lib/', 'tools/science_boundary/',
                'tools/lsp/', 'tools/fmt/', 'tools/repl.sh', 'stdlib/', 'LICENSE',
                'CITATION.cff', 'docs/compiler/KNOWN_LIMITATIONS.md')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def inventory(root):
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'symlink not allowed: {p}')
        if p.is_file() and p != root / 'distribution.json':
            rows.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                             sha256=digest(p), executable=bool(p.stat().st_mode & 0o111)))
    return rows


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'distribution.json').read_text())
    if manifest.get('schema') != 'sounio.compiler-distribution.v1':
        raise ValueError('unsupported distribution schema')
    if inventory(root) != manifest['files']:
        raise ValueError('distribution inventory mismatch (content, mode, missing or extra file)')
    binary = root / 'runtime/bin/madaros-linux-x86_64'
    if digest(binary) != manifest['compiler']['sha256']:
        raise ValueError('compiler hash mismatch')
    print(json.dumps(dict(status='verified', source_commit=manifest['source_commit'],
                          compiler_sha256=digest(binary))))
    return manifest


def build(repo, source, binary, expected, receipt, version, destination):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', version):
        raise ValueError('version must be a simple archive component')
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('output already exists')
    source = git(repo, 'rev-parse', '--verify', source + '^{commit}').decode().strip()
    if digest(binary) != expected:
        raise ValueError('binary SHA256 differs from reviewed build receipt')
    with binary.open('rb') as f:
        header = f.read(20)
    if header[:6] != b'\x7fELF\x02\x01' or header[18:20] != b'\x3e\x00':
        raise ValueError('requires a little-endian Linux x86_64 ELF64 candidate')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sounio-dist-', dir=destination.parent) as tmp:
        staging = Path(tmp)
        root = staging / ('sounio-' + version + '-linux-x86_64')
        root.mkdir()
        selected = []
        tree = git(repo, 'ls-tree', '-rz', source).split(b'\0')
        for entry in tree:
            if not entry:
                continue
            meta, rawpath = entry.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            path = rawpath.decode()
            if not any(path == p or (p.endswith('/') and path.startswith(p)) for p in SOURCE_PATHS):
                continue
            if kind != 'blob' or mode not in ('100644', '100755'):
                raise ValueError('unsupported source entry: ' + path)
            target = root / 'runtime' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(repo, 'cat-file', 'blob', oid))
            target.chmod(0o755 if mode == '100755' else 0o644)
            selected.append(path)
        for required in ('bin/souc', 'bin/madaros', 'LICENSE', 'tools/science_boundary/attestor.py'):
            if required not in selected:
                raise ValueError('missing canonical runtime input: ' + required)
        if not any(p.startswith('stdlib/') for p in selected):
            raise ValueError('source has no stdlib')
        raw = root / 'runtime/bin/madaros-linux-x86_64'
        shutil.copyfile(binary, raw)
        raw.chmod(0o755)
        if digest(raw) != expected:
            raise ValueError('binary changed during packaging')
        (root / 'bin').mkdir()
        for command in ('souc', 'madaros'):
            wrapper = root / 'bin' / command
            wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                               'DIST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
                               'export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$DIST_ROOT/runtime/stdlib}"\n'
                               f'exec "$DIST_ROOT/runtime/bin/{command}" "$@"\n')
            wrapper.chmod(0o755)
        shutil.copyfile(Path(__file__), root / 'verify-distribution.py')
        (root / 'verify-distribution.py').chmod(0o644)
        (root / 'README.txt').write_text(
            'Sounio/Madaros Linux x86_64 distribution\n'
            'Add this directory/bin to PATH. No source checkout is needed.\n'
            'Requires Linux x86_64, Bash, GNU coreutils (timeout), and Python 3.11+.\n'
            'LSP additionally requires jq. No bootstrap seed is shipped.\n'
            'Verify: python3 verify-distribution.py --verify .\n'
            'Use: souc check input.sio; souc compile input.sio -o program; souc run input.sio\n'
            'The canonical launcher reserves up to 512 MiB stack.\n'
            'Explicit compiler/stdlib environment overrides still take precedence.\n'
            'Bundled helper availability is not proof of a tested capability.\n'
            'See distribution.json for provenance and capability evidence status.\n')
        manifest = dict(schema='sounio.compiler-distribution.v1', version=version,
                        platform='linux-x86_64', source_commit=source,
                        source_tree=git(repo, 'rev-parse', source + '^{tree}').decode().strip(),
                        stdlib=dict(source_commit=source,
                                    git_tree=git(repo, 'rev-parse', source + ':stdlib').decode().strip(),
                                    path='runtime/stdlib'),
                        compiler=dict(path='runtime/bin/madaros-linux-x86_64', sha256=expected,
                                      build_receipt=receipt, provenance='publisher-declared-build-receipt'),
                        capabilities={key:'requires-external-consumer-validation' for key in
                                      ('check', 'compile', 'run', 'stdlib_import', 'lsp', 'format', 'pkg')},
                        files=inventory(root))
        (root / 'distribution.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        verify(root)
        archive = staging / 'bundle.tar.gz'
        with archive.open('wb') as f, gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for path in [root, *sorted(root.rglob('*'))]:
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(staging).as_posix())
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
                    if path.is_file():
                        with path.open('rb') as inp:
                            tar.addfile(info, inp)
                    else:
                        tar.addfile(info)
        # Atomic no-clobber publication on the same filesystem.
        os.link(archive, destination)
    print(json.dumps(dict(archive=str(destination), sha256=digest(destination), source_commit=source)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--source')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--binary-sha256')
    parser.add_argument('--build-receipt')
    parser.add_argument('--version')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.verify)
        else:
            if not all((args.source, args.binary, args.binary_sha256, args.build_receipt, args.version, args.out)):
                parser.error('build requires --source --binary --binary-sha256 --build-receipt --version --out')
            build(args.repo, args.source, args.binary, args.binary_sha256,
                  args.build_receipt, args.version, args.out)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'distribution refused: {error}\n')


if __name__ == '__main__':
    main()
