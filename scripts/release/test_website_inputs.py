"""Boundary tests for release integrity; no compiler or network required."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import export_website_inputs as inputs


class WebsiteInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {name if '.' in name.rsplit('/', 1)[-1] else name + '/fixture.txt': b'content\n'
                      for name in inputs.ROOTS}

    def source(self, files):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as archive:
            for name, payload in files.items():
                inputs.add_file(archive, name, payload)
        return data.getvalue()

    def create(self, name='inputs.tar.gz', files=None):
        target = self.root / name
        with patch.object(inputs.subprocess, 'check_output', side_effect=['a' * 40 + '\n', self.source(self.files if files is None else files)]):
            receipt = inputs.export(self.root, 'HEAD', target)
        return target, receipt

    def mutate(self, target, edit):
        with tarfile.open(target, 'r:gz') as archive:
            files = {entry.name: archive.extractfile(entry).read() for entry in archive}
        edit(files)
        with tarfile.open(target, 'w:gz') as archive:
            for name, payload in files.items():
                inputs.add_file(archive, name, payload)

    def proof_source(self, destination):
        data = io.BytesIO()
        name = next(iter(inputs.PROOF_LINKS))
        with tarfile.open(fileobj=data, mode='w') as archive:
            for path, payload in self.files.items():
                inputs.add_file(archive, path, payload)
            link = tarfile.TarInfo(name)
            link.type = tarfile.SYMTYPE
            link.linkname = destination
            archive.addfile(link)
        return data.getvalue()

    def test_historical_link_uses_committed_blob(self):
        name, origin = next(iter(inputs.PROOF_LINKS.items()))
        raw = self.proof_source('/home/demetrios/work/sounio/' + origin)
        responses = ['a' * 40, raw, '100644 blob ' + 'b' * 40 + '\t' + origin, b'theorem proof']
        target = self.root / 'proof.tar.gz'
        with patch.object(inputs.subprocess, 'check_output', side_effect=responses):
            inputs.export(self.root, 'HEAD', target)
        with tarfile.open(target) as archive:
            self.assertTrue(archive.getmember(name).isfile())
            self.assertEqual(archive.extractfile(name).read(), b'theorem proof')
            manifest = json.load(archive.extractfile(inputs.MANIFEST))
            row = next(row for row in manifest['files'] if row['path'] == name)
            self.assertEqual(row['source_path'], origin)

    def test_changed_historical_destination_rejected(self):
        with patch.object(inputs.subprocess, 'check_output', side_effect=['a' * 40, self.proof_source('/etc/passwd')]):
            with self.assertRaisesRegex(ValueError, 'Unexpected historical proof link'):
                inputs.export(self.root, 'HEAD', self.root / 'bad.tar.gz')

    def test_reproducible_and_pinned(self):
        first, receipt = self.create()
        second, _ = self.create('second.tar.gz')
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(inputs.verify(first, receipt['sha256'])['files'], len(self.files))
        with self.assertRaisesRegex(ValueError, 'pinned digest'):
            inputs.verify(first, '0' * 64)

    def test_changed_payload_rejected(self):
        target, _ = self.create()
        self.mutate(target, lambda files: files.update({'README.md': b'tampered'}))
        with self.assertRaisesRegex(ValueError, 'Inventory mismatch'):
            inputs.verify(target)

    def test_extra_member_rejected(self):
        target, _ = self.create()
        self.mutate(target, lambda files: files.update({'docs/unlisted.txt': b'extra'}))
        with self.assertRaisesRegex(ValueError, 'outside its inventory'):
            inputs.verify(target)

    def test_duplicate_manifest_entry_rejected(self):
        target, _ = self.create()
        def duplicate(files):
            manifest = json.loads(files[inputs.MANIFEST])
            manifest['files'].append(manifest['files'][0])
            files[inputs.MANIFEST] = json.dumps(manifest).encode()
        self.mutate(target, duplicate)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            inputs.verify(target)

    def test_missing_source_preserves_previous_output(self):
        target, _ = self.create()
        previous = target.read_bytes()
        incomplete = dict(self.files)
        del incomplete['README.md']
        with self.assertRaisesRegex(ValueError, 'Missing selected inputs'):
            self.create(files=incomplete)
        self.assertEqual(target.read_bytes(), previous)

    def test_traversal_rejected(self):
        target, _ = self.create()
        self.mutate(target, lambda files: files.update({'../outside': b'bad'}))
        with self.assertRaisesRegex(ValueError, 'Invalid archive path'):
            inputs.verify(target)

    def test_symlink_and_duplicate_members_rejected(self):
        for kind in ('symlink', 'duplicate'):
            with self.subTest(kind=kind):
                target = self.root / (kind + '.tar.gz')
                with tarfile.open(target, 'w:gz') as archive:
                    if kind == 'symlink':
                        entry = tarfile.TarInfo('README.md')
                        entry.type = tarfile.SYMTYPE
                        entry.linkname = '/etc/passwd'
                        entry.mode = 0o644
                        archive.addfile(entry)
                    else:
                        inputs.add_file(archive, 'README.md', b'a')
                        inputs.add_file(archive, 'README.md', b'b')
                with self.assertRaisesRegex(ValueError, 'Unsupported member'):
                    inputs.verify(target)


if __name__ == '__main__':
    unittest.main()
