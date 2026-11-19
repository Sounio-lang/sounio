"""Compare preserved CI logs; observations are not compiler event timestamps."""
import datetime, hashlib, json, pathlib, re, subprocess
HERE=pathlib.Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/".git").exists())
FAIL=HERE.parent/"source-ci-attempt-2"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(log,job):
    text=log.read_text();lines=text.splitlines()
    def first(marker):
        return next((line for line in lines if marker in line),None)
    def stamp(line):
        return datetime.datetime.fromisoformat(line.split(" ",1)[0].replace("Z","+00:00"))
    start=first("GATE_OBSERVER_START label=gen2")
    merged=first("Merged IR: 13269 functions")
    reset=first("arena_reset_totals ok=0 skip=124 sites_reclaimed=0")
    end=first("GATE_OBSERVER_END label=gen2")
    shutdown=first("The runner has received a shutdown signal.")
    meta=json.loads(job.read_text())
    return dict(job=meta["id"],source=meta["head_sha"],conclusion=meta["conclusion"],
        log_sha256=sha(log),job_sha256=sha(job),
        first_observed_arena_reset=reset,first_observed_merged_ir=merged,
        gen2_start=start,gen2_end=end,shutdown=shutdown,
        seconds_to_first_observed_merge=(stamp(merged)-stamp(start)).total_seconds(),
        gen2_completed=end is not None,gen2_rc=int(re.search(r"command_rc=(\d+)",end)[1]) if end else None,
        reached_run=first("reached  run") is not None)
sources=["2416ffb47c8d405fe0241f8aa182992c33c8d0c1","342b3f4e36c78c48d70b3ca27b7ef69eb659001e"]
paths=["self-hosted","stdlib","scripts/ci","scripts/lib",".github/workflows/ci.yml"]
identities={}
for source in sources:
    identities[source]={p:subprocess.check_output(["git","rev-parse",source+":"+p],cwd=ROOT,text=True).strip() for p in paths}
result=dict(schema="pireus-ci-shutdown-comparison-v1",
    successful=extract(HERE/"successful-job.log",HERE/"successful-job.json"),
    interrupted=extract(FAIL/"current-source-job.log",FAIL/"current-source-job.json"),
    git_object_identities=identities,
    compared_source_paths_identical=identities[sources[0]]==identities[sources[1]],
    root_cause_established=False,oom_established=False,compiler_binary_identity_established=False,
    timing_benchmark=False,old_source_qualified=False,
    observation_limit="Log observer samples every 30 seconds; first observed marker is not the compiler event timestamp. Exit 143 and runner shutdown do not identify the signal sender or cause.")
print(json.dumps(result,indent=2))
