#!/usr/bin/env python3
"""Provenance adversarial tests; synthetic ELF bytes are never executed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('canonical_madaros_artifact.py')


class Artifact(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, GIT_AUTHOR_NAME='CI Test', GIT_AUTHOR_EMAIL='ci@example.invalid',
                        GIT_COMMITTER_NAME='CI Test', GIT_COMMITTER_EMAIL='ci@example.invalid',
                        GITHUB_REPOSITORY='Sounio-lang/sounio', GITHUB_RUN_ID='123',
                        GITHUB_RUN_ATTEMPT='1')
        self.env.pop('GITHUB_OUTPUT', None)
        for path in ('bin/souc-linux-x86_64', 'scripts/ci/build_modular_madaros.sh'):
            p = self.root / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('test input')
        for args in (('init', '-q'), ('add', '.'), ('commit', '-qm', 'fixture')):
            subprocess.run(['git', *args], cwd=self.root, env=self.env, check=True, capture_output=True)
        self.env['GITHUB_SHA'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        self.art = self.root / 'artifact'
        self.art.mkdir()
        self.binary = self.art / 'madaros'
        self.binary.write_bytes(b'\x7fELF\x02\x01' + b'\0'*12 + b'\x3e\x00' + b'test-only')
        self.digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.assertEqual(self.run_tool('seal').returncode, 0)

    def run_tool(self, mode, digest=None):
        args = [sys.executable, str(SCRIPT), mode, str(self.art)]
        if mode == 'verify': args += ['--expected-sha256', digest or self.digest]
        return subprocess.run(args, cwd=self.root, env=self.env, capture_output=True, text=True)

    def test_accept_and_restore_execution_bit(self):
        self.binary.chmod(0o644)
        self.assertEqual(self.run_tool('verify').returncode, 0)
        self.assertTrue(self.binary.stat().st_mode & 0o111)

    def test_partial_rerun(self):
        self.env['GITHUB_RUN_ATTEMPT'] = '2'
        self.assertEqual(self.run_tool('verify').returncode, 0)

    def test_corruption_and_rewritten_receipt(self):
        self.binary.write_bytes(self.binary.read_bytes() + b'corrupt')
        self.assertNotEqual(self.run_tool('verify').returncode, 0)
        receipt = self.art / 'provenance.json'
        data = json.loads(receipt.read_text())
        data['binary_sha256'] = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        receipt.write_text(json.dumps(data))
        self.assertNotEqual(self.run_tool('verify').returncode, 0)

    def test_wrong_checkout_and_other_run(self):
        for field in ('GITHUB_SHA', 'GITHUB_REPOSITORY', 'GITHUB_RUN_ID'):
            old = self.env[field]
            self.env[field] = 'wrong'
            result = self.run_tool('verify')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CANONICAL_MADAROS_FAIL', result.stderr)
            self.env[field] = old

    def test_every_provenance_field(self):
        receipt = self.art / 'provenance.json'
        data = json.loads(receipt.read_text())
        for field in data:
            bad = dict(data); del bad[field]
            receipt.write_text(json.dumps(bad))
            self.assertNotEqual(self.run_tool('verify').returncode, 0, field)
        receipt.write_text(json.dumps(data))
        self.assertEqual(self.run_tool('verify').returncode, 0)

    def test_missing_and_malformed_receipt(self):
        receipt = self.art / 'provenance.json'
        receipt.write_text('{')
        self.assertNotEqual(self.run_tool('verify').returncode, 0)
        receipt.unlink()
        self.assertNotEqual(self.run_tool('verify').returncode, 0)

    def test_dirty_inputs(self):
        (self.root / 'bin/souc-linux-x86_64').write_text('mutated')
        self.assertNotEqual(self.run_tool('verify').returncode, 0)
        self.assertNotEqual(self.run_tool('seal').returncode, 0)

    def test_wrong_architecture_wrapper_missing_binary(self):
        original = self.binary.read_bytes()
        for data in (b'#!/bin/sh\nexit 0\n', b'', original[:18] + b'\xb7\0' + original[20:]):
            self.binary.write_bytes(data)
            self.assertNotEqual(self.run_tool('verify').returncode, 0)
        self.binary.unlink()
        self.assertNotEqual(self.run_tool('verify').returncode, 0)


if __name__ == '__main__':
    unittest.main()
