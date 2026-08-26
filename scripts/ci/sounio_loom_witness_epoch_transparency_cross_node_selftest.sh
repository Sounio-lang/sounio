#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
NAMESPACE="${SOUNIO_SLURM_NAMESPACE:-slurm-pilot}"

fail() {
  printf 'loom-witness-epoch-transparency-cross-node: FAIL: %s\n' "$*" >&2
  exit 1
}

remote_runner() {
  local remote_root="$1"
  local loom="$remote_root/loom.exe"
  local token="loom-epoch-transparency-${SLURM_JOB_ID:-manual}-$$"
  local world="cross-node-epoch-transparency"
  local base_port=$((22000 + ($$ % 20000)))
  local operator_node client_node
  local -a nodes=() witness_nodes=() service_pids=()
  local -A ports=()

  export SOUNIO_LOOM_OPENSSL="$remote_root/openssl"
  [[ -x "$SOUNIO_LOOM_OPENSSL" ]] ||
    fail "staged OpenSSL is missing: $SOUNIO_LOOM_OPENSSL"

  mapfile -t nodes < <(
    sinfo -N -h -o '%N %T' |
      awk '$2 !~ /^(down|drain|fail|maint|unknown)/ {print $1}' |
      sort -u
  )
  [[ "${#nodes[@]}" -ge 5 ]] ||
    fail "need five distinct available Slurm nodes, found ${#nodes[@]}"

  operator_node="${nodes[0]}"
  witness_nodes=("${nodes[1]}" "${nodes[2]}" "${nodes[3]}" "${nodes[4]}")
  client_node="${nodes[4]}"

  local client_root="$remote_root/client-state"
  local config_root="$remote_root"
  local base_root="$client_root/base-root"
  local epoch1_root="$client_root/epoch1-root"
  local epoch2_root="$client_root/epoch2-root"
  local epoch_state="$client_root/epoch-state"
  local transparency_state="$client_root/transparency-state"
  local log_state="/tmp/$token-operator"

  cleanup_remote() {
    local pid node
    for pid in "${service_pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
    for pid in "${service_pids[@]:-}"; do
      wait "$pid" 2>/dev/null || true
    done
    for node in "${nodes[@]:0:5}"; do
      srun --quiet --partition=all --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --mem=512M --nodelist="$node" \
        bash -lc "rm -rf '/tmp/$token-operator' /tmp/$token-*-config /tmp/$token-e1* /tmp/$token-e2* /tmp/$token-tr*" \
        >/dev/null 2>&1 || true
    done
  }
  trap cleanup_remote EXIT

  digest() {
    printf '%s' "$1" | sha256sum | awk '{print $1}'
  }

  membership() {
    printf '%s/%s-membership.tsv\n' "$remote_root" "$1"
  }

  endpoints() {
    printf '%s/%s-endpoints.tsv\n' "$remote_root" "$1"
  }

  client_run() {
    srun --quiet --partition=all --nodes=1 --ntasks=1 --cpus-per-task=1 \
      --mem=512M \
      --nodelist="$client_node" "$@"
  }

  client_shell() {
    srun --quiet --partition=all --nodes=1 --ntasks=1 --cpus-per-task=1 \
      --mem=512M --nodelist="$client_node" \
      bash -lc "$1"
  }

  start_witness() {
    local group="$1" index="$2" node="$3" port="$4"
    local key="${group}${index}" log="$remote_root/$key.log"
    local service_config="/tmp/$token-$key-config"
    : > "$log"
    srun --quiet --partition=all --nodes=1 --ntasks=1 --cpus-per-task=1 \
      --mem=512M --nodelist="$node" \
      bash -lc "set -e; rm -rf '$service_config'; mkdir -p '$service_config'; cp '$config_tar' '$service_config/config.tar'; printf '%s  %s\\n' '$config_sha' '$service_config/config.tar' | sha256sum -c - >/dev/null; tar -xf '$service_config/config.tar' -C '$service_config'; exec env SOUNIO_LOOM_OPENSSL='$SOUNIO_LOOM_OPENSSL' '$loom' witness-serve --witness-state-dir '/tmp/$token-$key-state' --membership '$service_config/$group-membership.tsv' --witness '$key' --private-key '$service_config/$key-private.pem' --bind 0.0.0.0 --port '$port'" \
      >"$log" 2>&1 &
    service_pids+=("$!")
    ports[$key]="$port"
  }

  wait_ready() {
    local label="$1" pattern="$2" log="$3" pid="$4" attempt=0
    until grep -Eq "$pattern" "$log"; do
      kill -0 "$pid" 2>/dev/null || {
        sed -n '1,180p' "$log" >&2
        fail "$label exited before readiness"
      }
      attempt=$((attempt + 1))
      [[ "$attempt" -lt 400 ]] || fail "$label readiness timed out"
      sleep 0.05
    done
  }

  for group in e1 e2 tr; do
    openssl genpkey -algorithm ED25519 \
      -out "$remote_root/$group-anchor-private.pem" 2>/dev/null
    openssl pkey -in "$remote_root/$group-anchor-private.pem" -pubout \
      -out "$remote_root/$group-anchor-public.pem" 2>/dev/null
    for index in 1 2 3 4; do
      key="${group}${index}"
      openssl genpkey -algorithm ED25519 \
        -out "$remote_root/$key-private.pem" 2>/dev/null
      openssl pkey -in "$remote_root/$key-private.pem" -pubout \
        -out "$remote_root/$key-public.pem" 2>/dev/null
    done
    {
      printf 'schema\tloom-witness-membership-v1\n'
      printf 'anchor_public_key\t%s\n' "$group-anchor-public.pem"
      printf 'witness_id\tpublic_key\n'
      for index in 1 2 3 4; do
        key="${group}${index}"
        printf '%s\t%s\n' "$key" "$key-public.pem"
      done
    } >"$(membership "$group")"
  done

  openssl genpkey -algorithm ED25519 \
    -out "$remote_root/operator-private.pem" 2>/dev/null
  openssl pkey -in "$remote_root/operator-private.pem" -pubout \
    -out "$remote_root/operator-public.pem" 2>/dev/null
  openssl genpkey -algorithm ED25519 \
    -out "$remote_root/publisher-private.pem" 2>/dev/null
  openssl pkey -in "$remote_root/publisher-private.pem" -pubout \
    -out "$remote_root/publisher-public.pem" 2>/dev/null

  (
    cd "$remote_root"
    tar -cf config.tar -- ./*.pem ./*-membership.tsv
  )
  local config_tar="$remote_root/config.tar"
  local config_sha
  config_sha="$(sha256sum "$config_tar" | awk '{print $1}')"
  local group_index=0 service_index=0 group index key node port log
  for group in e1 e2 tr; do
    {
      printf 'witness_id\thost\tport\n'
      for index in 1 2 3 4; do
        node="${witness_nodes[index - 1]}"
        port=$((base_port + service_index))
        key="${group}${index}"
        start_witness "$group" "$index" "$node" "$port"
        printf '%s\t%s\t%s\n' "$key" "$node" "$port"
        service_index=$((service_index + 1))
      done
    } >"$(endpoints "$group")"
    group_index=$((group_index + 1))
  done

  local witness_pid_index=0
  for group in e1 e2 tr; do
    for index in 1 2 3 4; do
      key="${group}${index}"
      log="$remote_root/$key.log"
      wait_ready "$key" 'LOOM_WITNESS_READY schema=loom-witness-service-v1 ' \
        "$log" "${service_pids[witness_pid_index]}"
      witness_pid_index=$((witness_pid_index + 1))
    done
  done

  local operator_port=$((base_port + service_index))
  local operator_log="$remote_root/operator.log"
  local operator_config="/tmp/$token-operator-config"
  : > "$operator_log"
  srun --quiet --partition=all --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=512M --nodelist="$operator_node" \
    bash -lc "set -e; rm -rf '$operator_config'; mkdir -p '$operator_config'; cp '$config_tar' '$operator_config/config.tar'; printf '%s  %s\\n' '$config_sha' '$operator_config/config.tar' | sha256sum -c - >/dev/null; tar -xf '$operator_config/config.tar' -C '$operator_config'; exec env SOUNIO_LOOM_OPENSSL='$SOUNIO_LOOM_OPENSSL' '$loom' witness-epoch-log-serve --log-state-dir '$log_state' --operator log-operator --operator-public-key '$operator_config/operator-public.pem' --operator-private-key '$operator_config/operator-private.pem' --publisher-public-key '$operator_config/publisher-public.pem' --bind 0.0.0.0 --log-port '$operator_port'" \
    >"$operator_log" 2>&1 &
  service_pids+=("$!")
  wait_ready operator 'LOOM_EPOCH_TRANSPARENCY_LOG_READY ' "$operator_log" \
    "${service_pids[${#service_pids[@]} - 1]}"

  local observed_operator
  observed_operator="$(sed -n 's/.* operator_host=\([^ ]*\) .*/\1/p' "$operator_log")"
  [[ "$observed_operator" == "$operator_node" ]] ||
    fail "operator hostname mismatch: expected=$operator_node observed=$observed_operator"
  [[ "$observed_operator" != "$client_node" ]] ||
    fail "operator and client collapsed onto $client_node"

  client_run "$loom" world-create --state-dir "$base_root" --world "$world" \
    --agent codex --lane epoch-transparency-cross-node >/dev/null
  client_run "$loom" knowledge-observe --state-dir "$base_root" --world "$world" \
    --knowledge checkpoint --value shared --error 0 --uncertainty bounded \
    --confidence 1 --provenance "$(digest cross-node-checkpoint)" >/dev/null
  client_shell "cp -a '$base_root' '$epoch1_root' && cp -a '$base_root' '$epoch2_root'"

  client_run "$loom" witness-mesh-anchor --state-dir "$epoch1_root" \
    --world "$world" --membership "$config_root/e1-membership.tsv" \
    --endpoints "$config_root/e1-endpoints.tsv" \
    --anchor-private-key "$config_root/e1-anchor-private.pem" >/dev/null
  client_run "$loom" witness-mesh-anchor --state-dir "$epoch2_root" \
    --world "$world" --membership "$config_root/e2-membership.tsv" \
    --endpoints "$config_root/e2-endpoints.tsv" \
    --anchor-private-key "$config_root/e2-anchor-private.pem" >/dev/null
  client_run "$loom" witness-epoch-handoff --epoch-state-dir "$epoch_state" \
    --world "$world" --from-epoch 1 --to-epoch 2 \
    --old-state-dir "$epoch1_root" --old-membership "$config_root/e1-membership.tsv" \
    --old-endpoints "$config_root/e1-endpoints.tsv" --new-state-dir "$epoch2_root" \
    --new-membership "$config_root/e2-membership.tsv" \
    --new-endpoints "$config_root/e2-endpoints.tsv" \
    >/dev/null

  local publish_output verify_output
  publish_output="$(
    client_run "$loom" witness-epoch-transparency-publish \
      --epoch-state-dir "$epoch_state" \
      --transparency-state-dir "$transparency_state" --world "$world" \
      --log-host "$operator_node" --log-port "$operator_port" \
      --operator log-operator \
      --operator-public-key "$config_root/operator-public.pem" \
      --publisher-public-key "$config_root/publisher-public.pem" \
      --publisher-private-key "$config_root/publisher-private.pem" \
      --transparency-membership "$config_root/tr-membership.tsv" \
      --transparency-endpoints "$config_root/tr-endpoints.tsv" \
      --transparency-anchor-private-key "$config_root/tr-anchor-private.pem"
  )"
  grep -Eq 'epoch=2 tree_size=1 .*quorum=[34]/4 .*custody=EXTERNAL_HOST native_frame=9016' \
    <<<"$publish_output" || fail "cross-node publish failed: $publish_output"

  verify_output="$(
    client_run "$loom" witness-epoch-transparency-verify \
      --epoch-state-dir "$epoch_state" \
      --transparency-state-dir "$transparency_state" --world "$world" \
      --log-host "$operator_node" --log-port "$operator_port" \
      --operator log-operator \
      --operator-public-key "$config_root/operator-public.pem" \
      --transparency-membership "$config_root/tr-membership.tsv" \
      --transparency-endpoints "$config_root/tr-endpoints.tsv"
  )"
  grep -Eq 'epoch=2 tree_size=1 .*quorum=[34]/4 .*rollback=NOT_BELOW_LATEST_QUORUM_WITNESSED .*custody=EXTERNAL_HOST native_frame=9016' \
    <<<"$verify_output" || fail "cross-node verify failed: $verify_output"

  printf 'SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_CROSS_NODE_GATE_PASS=true schema=loom-witness-epoch-transparency-v0 frame=9016 operator_node=%s client_node=%s witness_nodes=%s,%s,%s,%s distinct_hosts=5 operator_host_separation=VERIFIED witness_distribution=4_HOSTS quorum=3/4 rollback_below_latest_quorum_witnessed=REFUSED custody=EXTERNAL_HOST runtime=OCaml+Sounio\n' \
    "$operator_node" "$client_node" "${witness_nodes[0]}" \
    "${witness_nodes[1]}" "${witness_nodes[2]}" "${witness_nodes[3]}"
  trap - EXIT
  cleanup_remote
}

if [[ "${1:-}" == "--remote-runner" ]]; then
  [[ "$#" -eq 2 ]] || fail 'remote runner requires its OrangeFS root'
  remote_runner "$2"
  exit 0
fi

command -v kubectl >/dev/null 2>&1 || fail 'kubectl is required for the Slurm login path'
"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

login_pod="${SOUNIO_SLURM_LOGIN_POD:-}"
if [[ -z "$login_pod" ]]; then
  login_pod="$(
    kubectl -n "$NAMESPACE" get pods --field-selector=status.phase=Running \
      -o name | sed 's#^pod/##' | rg 'slurm-pilot-login-slinky' | head -1
  )"
fi
[[ -n "$login_pod" ]] || fail "no running Slurm login pod in namespace $NAMESPACE"

remote_root="/orangefs/training/loom-witness-epoch-transparency-cross-node-$(date -u +%Y%m%dT%H%M%SZ)-$$"
cleanup_outer() {
  kubectl -n "$NAMESPACE" exec "$login_pod" -- rm -rf "$remote_root" \
    >/dev/null 2>&1 || true
}
trap cleanup_outer EXIT

kubectl -n "$NAMESPACE" exec "$login_pod" -- mkdir -p "$remote_root"
kubectl -n "$NAMESPACE" cp /usr/bin/openssl "$login_pod:$remote_root/openssl"
for artifact in loom.exe \
  sounio-loom-epistemic-runtime \
  sounio-loom-witness-mesh-v1-runtime \
  sounio-loom-witness-epoch-handoff-runtime \
  sounio-loom-witness-epoch-transparency-runtime; do
  source_path="$ROOT_DIR/tools/loom/_build/default/src/$artifact"
  [[ -x "$source_path" ]] || fail "built runtime artifact missing: $source_path"
  kubectl -n "$NAMESPACE" cp "$source_path" \
    "$login_pod:$remote_root/$artifact"
done
kubectl -n "$NAMESPACE" cp "$0" "$login_pod:$remote_root/cross-node-selftest.sh"
kubectl -n "$NAMESPACE" exec "$login_pod" -- \
  bash "$remote_root/cross-node-selftest.sh" --remote-runner "$remote_root"

trap - EXIT
cleanup_outer
