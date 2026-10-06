#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-yaml build-essential ripgrep jq time unzip xz-utils
install -d -o 1000 -g 1000 /runner
curl --fail --location --retry 0 --output /tmp/runner.tar.gz https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-x64-2.337.0.tar.gz
printf '%s  %s\n' 70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613 /tmp/runner.tar.gz | sha256sum --check -
tar -xzf /tmp/runner.tar.gz -C /runner
bash /runner/bin/installdependencies.sh
chown -R 1000:1000 /runner
dpkg-query -W > /runner/packages.txt
python3 - <<'PY'
import json, pathlib
root=pathlib.Path("/sys/fs/cgroup")
memory=(root/"memory.max").read_text().strip()
cpu=(root/"cpu.max").read_text().strip()
assert memory == str(36*1024**3), ("unexpected cgroup memory",memory)
quota,period=map(int,cpu.split())
assert quota/period == 4, ("unexpected cgroup CPU",cpu)
assert not pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()
print(json.dumps(dict(stage="RUNNER_RESOURCE_PREFLIGHT",memory_max_bytes=int(memory),cpu_max=cpu,serviceaccount_token_present=False)))
PY
export HOME=/runner
cd /runner
if [[ "${PIREUS_RUNNER_MODE:-preflight}" == preflight ]]; then
  setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all --no-new-privs ./bin/Runner.Listener --version
  echo RUNNER_PREFLIGHT_COMPLETE_NO_REGISTRATION
  exit 0
fi
[[ "${PIREUS_RUNNER_MODE}" == execute ]] || exit 64
[[ -s /jit/encoded ]] || exit 65
JIT_CONFIG="$(cat /jit/encoded)"
exec setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all --no-new-privs ./run.sh --jitconfig "$JIT_CONFIG"
