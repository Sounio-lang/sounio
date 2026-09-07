#!/usr/bin/env python3
"""Mock transport test: stage old bridge before exact new admission and host Pods."""
import subprocess,sys,tempfile
from pathlib import Path
HARNESS=r'''
set -euo pipefail
export SOUNIO_SPARK_PAIR_BACKEND_LIBRARY_MODE=1
source "$1"
work="$2";negative="$3"
POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
guard_mutation() { :; }
admission_current() { [[ -e "$work/current" ]]; }
stage_existing_host_fence_for_bootstrap() { printf 'stage\n' >> "$work/trace"; }
sync_admission_projection() { printf 'sync\n' >> "$work/trace"; }
wait_for() { "$2"; }
bootstrap_gpu_admission_denied() {
  printf 'deny-control\n' >> "$work/trace"
  [[ "$negative" == 0 ]]
}
kubectl() {
  if [[ "$1" == apply ]]; then
    if [[ "${@: -1}" == "-" ]]; then cat >/dev/null;return 0;fi
    [[ "$(cat "$work/trace")" == stage ]]
    printf 'apply-admission\n' >> "$work/trace"
    touch "$work/current"
  elif [[ "$1 $2" == "get nodes" ]]; then printf '{"items":[]}\n'
  else printf 'unexpected mock transport: %s\n' "$*" >&2;return 99
  fi
}
device_barrier_config_json() { printf 'host-manifest-boundary\n' >> "$work/trace";exit 77; }
install_host_fence --holder test --epoch 12 --receipt ignored
'''
def main():
 backend=Path(sys.argv[1]).resolve()
 for negative in (0,1):
  with tempfile.TemporaryDirectory() as directory:
   result=subprocess.run(["bash","-c",HARNESS,"test",str(backend),directory,str(negative)],capture_output=True,text=True,timeout=15)
   assert result.returncode==(42 if negative else 77),(result.returncode,result.stdout,result.stderr)
   trace=(Path(directory)/"trace").read_text().splitlines()
   assert trace==["stage","apply-admission","sync","deny-control"]+([] if negative else ["host-manifest-boundary"]),trace
 print("PIREUS_HOST_REVISION_ORDER_PASS stage_before_admission=1 deny_before_host_pods=1 failed_negative_control_refused=1 transport=mock")
if __name__=="__main__":main()

