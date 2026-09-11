import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import host_decode_attempt as old
from host_decode_attempt_v2 import context
class SourceRevisionTests(unittest.TestCase):
 def test_parent_is_unchanged_and_runtime_matches(self):
  a=context()
  self.assertNotEqual(a.protocol()['source_commit'],old.protocol()['source_commit'])
  for field in ['runtime_sha256','files_sha256']:
   self.assertEqual(a.protocol()[field],old.protocol()[field])
  self.assertNotEqual(a.PROTOCOL,old.PROTOCOL)
 def test_tampered_protocol_is_refused(self):
  a=context()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'protocol.json';p.write_text('{}')
   with patch.object(a,'PROTOCOL',p),self.assertRaises(ValueError):a.verify(Path(d))
 def test_old_source_checks_cannot_qualify_new_source(self):
  a=context();spec=a.specification()
  checks=[dict(name=n,head_sha=old.protocol()['source_commit'],id=i,status='completed',conclusion='success') for i,n in enumerate(spec['required_source_checks'])]
  with self.assertRaises(ValueError):a.source_checks(spec,checks)
def load_tests(loader,tests,pattern):
 path=Path(__file__).with_name('test_host_decode_attempt.py')
 spec=importlib.util.spec_from_file_location('host_decode_v2_inherited_controls',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.a=context()
 tests.addTests(loader.loadTestsFromModule(module))
 return tests
if __name__=='__main__':unittest.main()
