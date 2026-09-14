import unittest
from verify_profile import verify
class ProfileGuard(unittest.TestCase):
 def setUp(self):
  self.env=dict(GITHUB_EVENT_NAME="workflow_dispatch",PIREUS_RUNNER_LABEL="cpu36-test",RUNNER_ENVIRONMENT="self-hosted",PIREUS_BASE_SHA="a"*40,GITHUB_SHA="b"*40)
 def test_valid_profile_is_not_qualification(self):
  self.assertFalse(verify(self.env,str(36*1024**3),"400000 100000",1000,False)["qualified"])
 def test_sabotaged_profiles_are_rejected(self):
  good=[self.env,str(36*1024**3),"400000 100000",1000,False]
  mutations=[(0,{**self.env,"GITHUB_EVENT_NAME":"pull_request"}),(0,{**self.env,"PIREUS_RUNNER_LABEL":""}),(0,{**self.env,"RUNNER_ENVIRONMENT":"github-hosted"}),(0,{**self.env,"PIREUS_BASE_SHA":"HEAD"}),(1,"max"),(1,str(16*1024**3)),(2,"200000 100000"),(3,0),(4,True)]
  for i,value in mutations:
   args=good.copy();args[i]=value
   with self.subTest(index=i,value=value),self.assertRaises(ValueError):verify(*args)
if __name__=="__main__":unittest.main()
