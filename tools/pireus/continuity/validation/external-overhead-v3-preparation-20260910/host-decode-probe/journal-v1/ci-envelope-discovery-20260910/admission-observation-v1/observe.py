import datetime,json,pathlib,subprocess,time
ROOT=pathlib.Path("/workspace/.cache/pireus-continuity/ci-runner-execute-v1-20260910")
REPO="/workspace/.wt/pireus-integration-20260906"
END=datetime.datetime.fromisoformat("2026-09-10T17:46:15+00:00")
def api(path):
 p=subprocess.run(["gh","api",path],cwd=REPO,capture_output=True,text=True,timeout=40)
 if p.returncode: return {"api_error":p.returncode}
 return json.loads(p.stdout)
with (ROOT/"admission-watch.jsonl").open("x") as out:
 while True:
  now=datetime.datetime.now(datetime.timezone.utc)
  v=api("repos/Sounio-lang/sounio/actions/jobs/102919731180")
  runner=api("orgs/Sounio-lang/actions/runners/6140")
  record={"observed_at":now.isoformat(),"job":{k:v.get(k) for k in ["id","status","conclusion","runner_id","runner_name","started_at","completed_at","api_error"]},"runner":{k:runner.get(k) for k in ["id","status","busy","api_error"]},"remaining_lifetime_seconds":max(0,(END-now).total_seconds()),"full_job_budget_fits":(END-now).total_seconds()>=9000,"automatic_retry":False}
  out.write(json.dumps(record)+"\n");out.flush()
  if v.get("status")=="completed" or now>=END:
   (ROOT/"admission-watch-terminal.json").write_text(json.dumps(record,indent=2)+"\n")
   break
  time.sleep(60)
