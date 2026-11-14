import ast,hashlib,json,tempfile,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/"ci-recovery.py").read_text())
tree.body=[node for node in tree.body if not isinstance(node,ast.Try)]
space={}
exec(compile(tree,"ci-recovery.py","exec"),space)
SOURCE=space["SOURCE"]
cases=[]
def case(name,head=SOURCE,attempt=2,conclusion="success",existing=False,corrupt=False):
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);raw=b"frozen driver"
  (root/"driver.py").write_bytes(raw+(b"changed" if corrupt else b""))
  (root/"driver-contract.json").write_text(json.dumps({"driver_sha256":hashlib.sha256(raw).hexdigest()}))
  if existing:(root/"attempt").mkdir()
  calls=[];events=[];queries=[]
  def output(argv):
   queries.append(argv)
   return json.dumps({"head_sha":head,"run_attempt":attempt,"status":"completed","conclusion":conclusion}).encode()
  def call(argv,**kwargs):calls.append(argv);return 0
  scope=dict(space)
  # Function globals must reference the isolated fake root and subprocess.
  scope.update(ROOT=root,subprocess=types.SimpleNamespace(check_output=output,call=call,STDOUT=-2),event=lambda stage,**kw:events.append(stage))
  main=types.FunctionType(space["main"].__code__,scope)
  error=None
  try:main()
  except ValueError as exc:error=str(exc)
  accepted=not existing and not corrupt and head==SOURCE and attempt==2 and conclusion=="success"
  assert len(calls)==int(accepted),(name,calls,error)
  if existing:assert not queries
  if not accepted and conclusion=="success":assert error
  if conclusion=="failure":assert "STOP_SOURCE_CI_ATTEMPT_2" in events
  cases.append({"name":name,"pass":True,"driver_invocations":len(calls),"hardware_executed":False})
case("accepted exact-source attempt 2 invokes original driver once")
case("existing hardware attempt refuses before querying CI",existing=True)
case("changed source refuses",head="0"*40)
case("changed CI attempt refuses",attempt=3)
case("failed CI stops without invoking driver",conclusion="failure")
case("changed original driver refuses",corrupt=True)
for name,expected in json.loads((HERE/"source-ci-attempt-1/manifest.json").read_bytes())["files"].items():
 assert hashlib.sha256((HERE/"source-ci-attempt-1"/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({"schema":"pireus-ci-recovery-controls-v1","controls":cases,"archived_failure_hashes_pass":True,"hardware_qualified":False},indent=2))
