"""Exercise the actual wrapper artifact function with controlled compiler processes."""
import json,os,pathlib,subprocess,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCE=(ROOT/'bin/madaros').read_text()
FUNCTION=SOURCE.split('_compile_source_to_artifact() {',1)[1].split('\n}\n',1)[0]
FUNCTION='_compile_source_to_artifact() {'+FUNCTION+'\n}\n'
class TimeoutTests(unittest.TestCase):
 def invoke(self,value=None,mode='ok',backend='native',real_timeout=False):
  with tempfile.TemporaryDirectory() as folder:
   p=pathlib.Path(folder)
   raw=p/'raw'
   raw.write_text('#!/usr/bin/env python3\nimport os,sys,time,pathlib\na=sys.argv[1:];o=pathlib.Path(a[a.index("-o")+1]);o.write_text("partial")\nif os.environ["MODE"]=="sleep":time.sleep(3)\nif os.environ["MODE"]=="fail":sys.exit(17)\n')
   raw.chmod(0o755)
   timeout=p/'timeout'
   timeout.write_text('#!/usr/bin/env python3\nimport sys,os,json,pathlib,subprocess\npathlib.Path(os.environ["ARGV_LOG"]).write_text(json.dumps(sys.argv[1:]))\nsys.exit(subprocess.call(sys.argv[2:]))\n')
   timeout.chmod(0o755)
   env=dict(os.environ,RAW_MADAROS=str(raw),MADAROS_VMEM_LIMIT_KB='33554432',MODE=mode,ARGV_LOG=str(p/'argv'),OUT=str(p/'out'))
   env.pop('MADAROS_NATIVE_COMPILE_TIMEOUT_SECONDS',None)
   if value is not None:env['MADAROS_NATIVE_COMPILE_TIMEOUT_SECONDS']=value
   if not real_timeout:env['PATH']=str(p)+os.pathsep+env['PATH']
   script=FUNCTION+'\nBUILD_BACKEND='+backend+'\nBUILD_GPU_TARGET=sm_121\nBUILD_GPU_BINARY_FORMAT=""\n_compile_source_to_artifact input.sio "$OUT"\n'
   r=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True)
   args=json.loads((p/'argv').read_text()) if (p/'argv').exists() else None
   return r,args,(p/'out').exists()
 def test_default(self):
  r,a,e=self.invoke();self.assertEqual(r.returncode,0);self.assertEqual(a[0],'300');self.assertTrue(e)
 def test_bounded_override(self):
  for value in ['1','600']:
   with self.subTest(value=value):
    r,a,e=self.invoke(value);self.assertEqual(r.returncode,0);self.assertEqual(a[0],value)
 def test_invalid_refuses_before_compiler(self):
  for value in ['0','601','-1','01','1s','1.5','unlimited','99999999999999999999','$(touch bad)']:
   with self.subTest(value=value):
    r,a,e=self.invoke(value);self.assertEqual(r.returncode,2);self.assertIsNone(a);self.assertFalse(e)
 def test_gpu_budget_unchanged(self):
  r,a,e=self.invoke('600',backend='gpu');self.assertEqual(r.returncode,0);self.assertEqual(a[0],'300')
 def test_failure_removes_partial_artifact(self):
  r,a,e=self.invoke('600',mode='fail');self.assertEqual(r.returncode,17);self.assertFalse(e)
 def test_real_timeout_removes_partial_artifact(self):
  r,a,e=self.invoke('1',mode='sleep',real_timeout=True);self.assertEqual(r.returncode,124);self.assertFalse(e)
 def test_gate_budgets(self):
  text=(ROOT/'scripts/ci/madaros_imported_capacity_gate.sh').read_text()
  for target,budget in [('BOUNDARY_MAIN','600'),('OVERFLOW_MAIN','300')]:
   lines=[line for line in text.splitlines() if 'bin/madaros" compile "$'+target+'"' in line]
   self.assertEqual(len(lines),1);self.assertIn('MADAROS_NATIVE_COMPILE_TIMEOUT_SECONDS='+budget+' ',lines[0])
if __name__=='__main__':unittest.main()
