"""Distribution contract tests; ELF fixture is packaging-only, never executed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/distribution/bundle.py'
spec = importlib.util.spec_from_file_location('bundle', SCRIPT)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        for path in ('bin/souc', 'bin/madaros', 'LICENSE', 'stdlib/cmp/lib.sio',
                     'tools/science_boundary/attestor.py'):
            p = self.repo / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('original\n')
            if path.startswith('bin/'):
                p.chmod(0o755)
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Distribution Test',
                        '-c', 'user.email=distribution@example.invalid',
                        'commit', '-qm', 'fixture'], check=True)
        self.binary = self.root / 'elf'
        header = bytearray(20)
        header[:6] = b'\x7fELF\x02\x01'
        header[18:20] = b'\x3e\x00'
        self.binary.write_bytes(header)
        self.sha = bundle.digest(self.binary)

    def build(self, name='bundle.tar.gz', sha=None):
        output = self.root / name
        bundle.build(self.repo, 'HEAD', self.binary, sha or self.sha,
                     'fixture://packaging-only', 'test', output)
        return output

    def unpack(self):
        archive = self.build()
        target = self.root / 'extracted'
        target.mkdir()
        with tarfile.open(archive) as t:
            t.extractall(target, filter='data')
        return target / 'sounio-test-linux-x86_64'

    def test_reproducible_and_commit_pinned(self):
        a = self.build('a.tar.gz')
        (self.repo / 'stdlib/cmp/lib.sio').write_text('dirty content must not ship\n')
        b = self.build('b.tar.gz')
        self.assertEqual(bundle.digest(a), bundle.digest(b))
        with tarfile.open(a) as tar:
            self.assertEqual(tar.extractfile('sounio-test-linux-x86_64/runtime/stdlib/cmp/lib.sio').read(), b'original\n')

    def test_wrong_hash_and_occupied_output_refused(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.build(sha='0'*64)
        self.assertFalse((self.root / 'bundle.tar.gz').exists())
        p = self.build()
        original = bundle.digest(p)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.build()
        self.assertEqual(bundle.digest(p), original)

    def test_relocation_tampering_and_extra_files(self):
        root = self.unpack()
        moved = self.root / 'relocated with spaces'
        root.rename(moved)
        bundle.verify(moved)
        file = moved / 'runtime/stdlib/cmp/lib.sio'
        file.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            bundle.verify(moved)
        file.write_text('original\n')
        (moved / 'unexpected').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            bundle.verify(moved)

    def test_symlink_and_architecture_refused(self):
        p = self.repo / 'stdlib/alias'
        p.symlink_to('cmp/lib.sio')
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Distribution Test',
                        '-c', 'user.email=distribution@example.invalid', 'commit', '-qm', 'symlink'], check=True)
        with self.assertRaisesRegex(ValueError, 'unsupported source entry'):
            self.build()
        self.binary.write_bytes(b'not an ELF')
        with self.assertRaisesRegex(ValueError, 'ELF64'):
            self.build(sha=bundle.digest(self.binary))


if __name__ == '__main__':
    unittest.main()
