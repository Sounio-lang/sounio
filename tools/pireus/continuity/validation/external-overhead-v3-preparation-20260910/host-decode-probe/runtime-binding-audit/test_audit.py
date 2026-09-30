import copy,unittest
from audit import validate
class BindingTests(unittest.TestCase):
 def setUp(self):
  self.start={'workers':[{'uid':'u','boot_id':'b'}],'runtime_before_sha256':[{'x':'h'}],'input_sha256':'i'}
  self.receipt={'stage':'OVERHEAD_RUNTIME_VERIFIED','job':'1','rank':'0','worker_uid':'u','boot_id':'b','runtime_sha256':{'x':'h'},'input_sha256':'i','monotonic_ns':10}
  self.life=[{'job':'1','rank':'0','monotonic_ns':11}]
 def test_valid(self):
  self.assertTrue(validate(self.receipt,self.start,'1',0,self.life)['runtime_binding_verified'])
 def test_identity_and_hash_sabotage(self):
  for key,value in [('stage','OTHER'),('job','2'),('rank','1'),('worker_uid','other'),('boot_id','other'),('runtime_sha256',{'x':'wrong'}),('input_sha256','wrong')]:
   with self.subTest(key=key):
    r=copy.deepcopy(self.receipt);r[key]=value
    with self.assertRaises(ValueError):validate(r,self.start,'1',0,self.life)
 def test_timestamp_sabotage(self):
  for value in [-1,True,'10',12]:
   with self.subTest(value=value):
    r=copy.deepcopy(self.receipt);r['monotonic_ns']=value
    with self.assertRaises(ValueError):validate(r,self.start,'1',0,self.life)
 def test_missing_lifecycle_stays_absent(self):
  result=validate(self.receipt,self.start,'1',0,[])
  self.assertFalse(result['lifecycle_present'])
if __name__=='__main__':unittest.main()
