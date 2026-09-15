#!/usr/bin/env python3
"""Real process/lock tests using a temporary host root; never touches a GPU."""
import os,subprocess,tempfile,time,sys,re
from pathlib import Path
def main():
 source=Path(sys.argv[1]).read_text()
 mocks=r'''
grant_valid() { return 1; }
device_barrier_attached() {
  touch "$HOST_ROOT/ready"
  while [[ ! -e "$HOST_ROOT/release" ]]; do sleep .02; done
  return 0
}
enforce_fenced_compute_state() { printf 'mode=FENCED\n' > "$GRANT_FILE"; }
legacy_gpu_inventory_exact() { return 0; }
known_gpu_services_quiesced() { return 0; }
managed_gpu_restarts_blocked() { return 0; }
active_docker_gpu_claims_zero() { return 0; }
gpu_consumer_set_exact() { return 0; }
managed_gpu_cgroups_empty() { return 0; }
live_memory_floor_met() { return 0; }
protected_resources_unchanged() { return 0; }
test_commit() { printf 'mode=SLURM\n' > "$GRANT_FILE"; touch "$HOST_ROOT/committed"; }
test_hang() { sleep 90; }
test_docker_grace() {
  device_barrier_attach() { return 0; }
  device_barrier_relation_valid() { return 0; }
  disable_known_gpu_services() { return 0; }
  kill_kubernetes_compute_processes() { return 0; }
  fenced_compute_set_empty() { return 0; }
  docker_container_has_gpu_claim() { return 0; }
  docker_host() {
    case "$1" in
      ps) printf 'known-gpu-container\n' ;;
      update) [[ "$2" == --restart=no && "$3" == known-gpu-container ]] ;;
      inspect)
        if [[ -e "$HOST_ROOT/docker-stopped" ]]; then printf 'false\n'; else printf 'true\n'; fi
        ;;
      stop)
        [[ "$2" == -t && "$3" == 5 && "$4" == known-gpu-container ]]
        sleep "$3"
        touch "$HOST_ROOT/docker-stopped"
        ;;
      *) return 64 ;;
    esac
  }
  real_enforce_fenced_compute_state
  status_once
  [[ -e "$HOST_ROOT/docker-stopped" ]]
}
test_escape() { setsid sleep 15 >/dev/null 2>&1 & echo $! > "$HOST_ROOT/escaped-pid"; }
'''
 real_enforce=re.search(r"^enforce_fenced_compute_state\(\) \{.*?^\}",source,re.M|re.S).group(0).replace("enforce_fenced_compute_state()","real_enforce_fenced_compute_state()",1)
 mocks=real_enforce+"\n"+mocks
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory)
  script=root/"fence.sh"
  script.write_text(source.replace('if [[ "${PIREUS_HOST_FENCE_LIBRARY_MODE:-0}" == 1 ]]; then',mocks+'\nif [[ "${PIREUS_HOST_FENCE_LIBRARY_MODE:-0}" == 1 ]]; then'))
  env=os.environ.copy();env.update(NODE_NAME="spark-3c59",PIREUS_HOST_ROOT=str(root),PIREUS_HOST_FENCE_LIBRARY_MODE="1")
  grant=root/"var/lib/pireus-spark-pair/host-grant";grant.parent.mkdir(parents=True)
  def launch(command):
   return subprocess.Popen(["bash","-c",'source "$1"; '+command,"test",str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  def wait_file(path):
   end=time.monotonic()+5
   while not path.exists():
    assert time.monotonic()<end,"fixture synchronization timeout"
    time.sleep(.01)
  for locked in (False,True):
   for name in ("ready","release","committed"):
    (root/name).unlink(missing_ok=True)
   grant.write_text("mode=FENCED\n")
   prefix="with_host_transition_lock " if locked else ""
   watchdog=launch(prefix+"active_enforcement_cycle")
   wait_file(root/"ready")
   commit=launch(prefix+"test_commit")
   if locked:
    time.sleep(.2);assert not (root/"committed").exists(),"commit bypassed watchdog lock"
   else:wait_file(root/"committed")
   (root/"release").touch()
   for proc in (watchdog,commit):
    stdout,stderr=proc.communicate(timeout=8)
    assert proc.returncode==0,(stdout,stderr)
   assert grant.read_text()==("mode=SLURM\n" if locked else "mode=FENCED\n")
  lock=grant.parent/"host-transition.lock"
  # Contention has its own refusal code and must not execute the operation.
  holder=subprocess.Popen(["flock",str(lock),"sleep","12"])
  try:
   time.sleep(.2);start=time.monotonic()
   proc=launch("with_host_transition_lock test_commit");out,err=proc.communicate(timeout=13)
   assert proc.returncode==75,(proc.returncode,out,err)
   assert 9<=time.monotonic()-start<13
  finally:
   holder.wait(timeout=4)
  # A wedged operation loses the entire process group within the bounded timeout.
  start=time.monotonic()
  proc=launch("with_host_transition_lock test_hang");out,err=proc.communicate(timeout=49)
  assert proc.returncode==124,(proc.returncode,out,err)
  assert 44<=time.monotonic()-start<49
  proc=launch("with_host_transition_lock test_commit")
  out,err=proc.communicate(timeout=5);assert proc.returncode==0,(out,err)
  # A child in a new session survives its caller, but must not inherit the lock.
  proc=launch("with_host_transition_lock test_escape")
  out,err=proc.communicate(timeout=4);assert proc.returncode==0,(out,err)
  escaped=int((root/"escaped-pid").read_text())
  try:
   os.kill(escaped,0)
   start=time.monotonic();proc=launch("with_host_transition_lock test_commit")
   out,err=proc.communicate(timeout=4)
   assert proc.returncode==0 and time.monotonic()-start<3,(out,err)
   os.kill(escaped,0)
  finally:os.kill(escaped,15)
  # The emergency path can outwait an owner that exceeds normal lock patience.
  holder=subprocess.Popen(["flock",str(lock),"sleep","12"])
  try:
   time.sleep(.2);start=time.monotonic()
   proc=launch("with_host_transition_lock_wait 60 test_commit")
   out,err=proc.communicate(timeout=16)
   assert proc.returncode==0 and 10<time.monotonic()-start<16,(out,err)
  finally:holder.wait(timeout=4)
  # Execute the real fence/stop/status functions with a bounded simulated Docker daemon.
  start=time.monotonic();proc=launch("with_host_transition_lock test_docker_grace")
  out,err=proc.communicate(timeout=10);assert proc.returncode==0,(out,err)
  assert 5<=time.monotonic()-start<10
  assert "TimeoutStopSec=120" in source and "KillMode=control-group" in source
  assert "WatchdogSec=${SYSTEMD_WATCHDOG_SECONDS}s" in source
  # Exercise the actual daemon dispatch: failure cannot refresh its heartbeat.
  base_fixture=script.read_text()
  (root/"var/lib/pireus-spark-pair/activated").touch()
  for code in (1,0):
   (root/"ping").unlink(missing_ok=True)
   overrides='\ncapture_protected_baseline() { :; }\nwatchdog_notify_ready() { :; }\nwatchdog_ping() { touch "$HOST_ROOT/ping"; }\nactive_enforcement_cycle() { return '+str(code)+'; }\n'
   script.write_text(base_fixture.replace('if [[ "${PIREUS_HOST_FENCE_LIBRARY_MODE:-0}" == 1 ]]; then',overrides+'if [[ "${PIREUS_HOST_FENCE_LIBRARY_MODE:-0}" == 1 ]]; then'))
   daemon_env=env|{"PIREUS_HOST_FENCE_LIBRARY_MODE":"0"}
   result=subprocess.run(["timeout","1","bash",str(script),"enforce-notify"],env=daemon_env,capture_output=True,timeout=4)
   assert result.returncode==124,(result.returncode,result.stderr)
   assert (root/"ping").exists()==(code==0),"failed daemon cycle forged a heartbeat"
 print("PIREUS_HOST_TRANSITION_LOCK_PASS old_interleaving_reproduced=1 serialized_commit_survives=1 contention_refused=1 hang_bounded=1 lock_reusable=1 failed_cycle_heartbeat_refused=1 escaped_child_lock_released=1 emergency_outwaits_owner=1 real_fence_stop_status_bounded=1")
if __name__=="__main__":main()

