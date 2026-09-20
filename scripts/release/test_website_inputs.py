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
        self.files = {name + '/fixture.txt' if name in inputs.DIR_ROOTS else name: b'content\n'
                      for name in inputs.ROOTS}

    def source(self, files=None, omit=None, bad_destination=None):
        files = self.files if files is None else files
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as archive:
            for name, payload in files.items():
                inputs.add_file(archive, name, payload)
            for name, origin in inputs.PROOF_LINKS.items():
                if name == omit or name in files:
                    continue
                link = tarfile.TarInfo(name)
                link.type = tarfile.SYMTYPE
                link.linkname = bad_destination or '/home/demetrios/work/sounio/' + origin
                archive.addfile(link)
        return data.getvalue()

    def create(self, name='inputs.tar.gz', files=None, source=None):
        target = self.root / name
        raw = self.source(files) if source is None else source
        def git(command, **kwargs):
            operation = command[3]
            if operation == 'rev-parse':
                return 'a' * 40
            if operation == 'archive':
                return raw
            if operation == 'ls-tree':
                return '100644 blob ' + 'b' * 40 + '\t' + command[-1]
            if operation == 'show':
                return ('proof:' + command[-1].split(':', 1)[1]).encode()
            self.fail('Unexpected Git operation: ' + operation)
        with patch.object(inputs.subprocess, 'check_output', side_effect=git):
            receipt = inputs.export(self.root, 'HEAD', target)
        return target, receipt

    def mutate(self, target, edit):
        with tarfile.open(target, 'r:gz') as archive:
            files = {entry.name: archive.extractfile(entry).read() for entry in archive}
        edit(files)
        with tarfile.open(target, 'w:gz') as archive:
            for name, payload in files.items():
                inputs.add_file(archive, name, payload)

    def test_reproducible_and_pinned(self):
        first, receipt = self.create()
        second, _ = self.create('second.tar.gz')
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(inputs.verify(first, receipt['sha256'])['files'], len(self.files) + 12)
        with self.assertRaisesRegex(ValueError, 'pinned digest'):
            inputs.verify(first, '0' * 64)

    def test_historical_links_use_committed_blobs(self):
        target, _ = self.create()
        with tarfile.open(target) as archive:
            manifest = json.load(archive.extractfile(inputs.MANIFEST))
            for name, origin in inputs.PROOF_LINKS.items():
                self.assertTrue(archive.getmember(name).isfile())
                self.assertEqual(archive.extractfile(name).read(), ('proof:' + origin).encode())
                row = next(row for row in manifest['files'] if row['path'] == name)
                self.assertEqual(row['source_path'], origin)

    def test_changed_historical_destination_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected historical proof link'):
            self.create(source=self.source(bad_destination='/etc/passwd'))

    def test_regular_file_cannot_replace_proof_link(self):
        files = dict(self.files)
        files[next(iter(inputs.PROOF_LINKS))] = b'replacement'
        with self.assertRaisesRegex(ValueError, 'Unsupported input member'):
            self.create(files=files)

    def test_missing_proof_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing selected inputs'):
            self.create(source=self.source(omit=next(iter(inputs.PROOF_LINKS))))

    def test_proof_cannot_claim_self_origin(self):
        target, _ = self.create()
        def edit(files):
            m = json.loads(files[inputs.MANIFEST])
            row = next(row for row in m['files'] if row['path'] in inputs.PROOF_LINKS)
            row['source_path'] = row['path']
            files[inputs.MANIFEST] = json.dumps(m).encode()
        self.mutate(target, edit)
        with self.assertRaisesRegex(ValueError, 'Unexpected inventory source'):
            inputs.verify(target)

    def test_consistent_inventory_cannot_omit_required_proof(self):
        target, _ = self.create()
        def edit(files):
            name = next(iter(inputs.PROOF_LINKS))
            del files[name]
            m = json.loads(files[inputs.MANIFEST])
            m['files'] = [row for row in m['files'] if row['path'] != name]
            files[inputs.MANIFEST] = json.dumps(m).encode()
        self.mutate(target, edit)
        with self.assertRaisesRegex(ValueError, 'Missing selected inputs'):
            inputs.verify(target)

    def test_file_and_directory_root_shapes(self):
        for old, replacement in [('README.md', 'README.md/payload'), ('docs/fixture.txt', 'docs')]:
            with self.subTest(replacement=replacement):
                files = dict(self.files)
                files[replacement] = files.pop(old)
                with self.assertRaisesRegex(ValueError, 'Unsupported or duplicate input'):
                    self.create(files=files)
                target, _ = self.create()
                def edit(payloads):
                    payloads[replacement] = payloads.pop(old)
                    m = json.loads(payloads[inputs.MANIFEST])
                    row = next(row for row in m['files'] if row['path'] == old)
                    row['path'] = replacement
                    row['source_path'] = replacement
                    payloads[inputs.MANIFEST] = json.dumps(m).encode()
                self.mutate(target, edit)
                with self.assertRaisesRegex(ValueError, 'unselected inventory entry'):
                    inputs.verify(target)

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
            m = json.loads(files[inputs.MANIFEST])
            m['files'].append(m['files'][0])
            files[inputs.MANIFEST] = json.dumps(m).encode()
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
