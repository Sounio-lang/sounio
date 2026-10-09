import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class ModelQualificationTests(unittest.TestCase):
 def test_lfs_and_git_bytes_then_corruption(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);snapshot=root/"snapshot";snapshot.mkdir()
   weight=b"frozen weight bytes";config=b'{"fixture":true}\n'
   (snapshot/"weights").write_bytes(weight);(snapshot/"config").write_bytes(config)
   entries=[dict(rfilename="weights",size=len(weight),lfs=dict(sha256=hashlib.sha256(weight).hexdigest())),
            dict(rfilename="config",size=len(config),blobId=hashlib.sha1(b"blob "+str(len(config)).encode()+b"\0"+config).hexdigest())]
   manifest=root/"manifest.json";manifest.write_text(json.dumps(entries))
   receipt=root/"receipt.json"
   command=[sys.executable,str(HERE/"qualify_model.py"),str(snapshot),str(manifest),"--receipt",str(receipt)]
   good=subprocess.run(command,capture_output=True,text=True)
   self.assertEqual(good.returncode,0,good.stderr)
   self.assertEqual(len(json.loads(receipt.read_text())["files"]),2)
   (snapshot/"weights").write_bytes(b"x"*len(weight))
   bad=subprocess.run(command,capture_output=True,text=True)
   self.assertNotEqual(bad.returncode,0)
   self.assertIn("sha256: weights",bad.stderr)
 def test_no_receipt_for_wrong_git_blob(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/"config").write_bytes(b"x")
   (root/"manifest.json").write_text(json.dumps([dict(rfilename="config",size=1,blobId="0"*40)]))
   receipt=root/"receipt.json"
   run=subprocess.run([sys.executable,str(HERE/"qualify_model.py"),str(root),str(root/"manifest.json"),"--receipt",str(receipt)],capture_output=True,text=True)
   self.assertNotEqual(run.returncode,0)
   self.assertFalse(receipt.exists())
if __name__=="__main__":unittest.main()
