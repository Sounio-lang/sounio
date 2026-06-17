# Self-hosted in-cluster runner for the K-AXI PTX acceptance gate

The nightly acceptance gate (`.github/workflows/kaxi-ptx-acceptance.yml`) needs
to reach the on-prem SLURM cluster via `kubectl -n slurm-pilot exec …`. A
GitHub-*hosted* runner generally **cannot** reach the cluster's k8s API
(it's on a private Proxmox network), so the working setup is a **self-hosted
runner that lives inside the cluster** — exactly where `kubectl` already works.

Bonus: an in-cluster runner authenticates to k8s with an ambient **ServiceAccount
token**, so **no `SLURM_KUBECONFIG` secret is needed in GitHub** — the gate only
needs RBAC scoped to "exec into the login pod".

Everything below runs once, by a human with cluster + repo admin. Replace
`OWNER/REPO` with `Sounio-lang/sounio` and `NAMESPACE` with the namespace you
deploy into (using `slurm-pilot` keeps the RBAC a namespaced `Role`).

---

## What the runner must be able to do

- Reach the k8s API and `kubectl -n slurm-pilot`:
  - `get`/`list` pods (resolve the login pod),
  - `create` `pods/exec` (run `sbatch`/`squeue`/`cat`/`tar` in the login pod),
  - `cp` (implemented as `exec`, so `pods/exec` covers it).
- Have a toolchain for the gate's pre-submit steps: `gcc` + `-lm`/`-ldl`
  (build `kaxi_ptx_runner.c`), `git`, `curl`, `tar`, `kubectl`, and enough of a
  base to run the pinned `souc` release binary (glibc Linux x86-64).
- Egress to `github.com` / `dl.k8s.io` (checkout, runner self-update, pinned
  souc download, kubectl install if not baked in).

The gate does **not** request a GPU on the runner — the GPU work happens on the
SLURM worker. A small CPU pod is enough.

---

## Step 1 — RBAC: ServiceAccount + Role + RoleBinding

```yaml
# kaxi-ci-runner-rbac.yaml   (apply in the slurm-pilot namespace)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kaxi-ci-runner
  namespace: slurm-pilot
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: kaxi-ci-runner
  namespace: slurm-pilot
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: ["create", "get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kaxi-ci-runner
  namespace: slurm-pilot
subjects:
  - kind: ServiceAccount
    name: kaxi-ci-runner
    namespace: slurm-pilot
roleRef:
  kind: Role
  name: kaxi-ci-runner
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f kaxi-ci-runner-rbac.yaml
```

This is the least privilege the gate needs: exec into the login pod in one
namespace. No cluster-wide rights, no node access.

---

## Step 2 — GitHub credential for runner registration

Use a fine-grained PAT (or classic PAT with `repo`) that can manage the repo's
self-hosted runners; store it as a k8s secret. The `myoung34/github-runner`
image uses it to mint short-lived registration tokens automatically (survives
restarts, unlike a one-shot registration token).

```bash
kubectl -n slurm-pilot create secret generic kaxi-ci-runner-gh \
  --from-literal=ACCESS_TOKEN='github_pat_xxx'
```

(One-shot alternative, expires in ~1h — fine for a quick test, not for a
long-lived Deployment:
`gh api -X POST /repos/OWNER/REPO/actions/runners/registration-token --jq .token`,
then pass it as `RUNNER_TOKEN` instead of `ACCESS_TOKEN`.)

---

## Step 3 — Runner image with kubectl + gcc

The stock runner images don't ship `kubectl`/`gcc`. Bake a tiny image:

```dockerfile
# Dockerfile.kaxi-runner
FROM myoung34/github-runner:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git tar gzip \
    && curl -fsSLo /usr/local/bin/kubectl \
        "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && rm -rf /var/lib/apt/lists/*
```

```bash
docker build -f Dockerfile.kaxi-runner -t <your-registry>/kaxi-ci-runner:1 .
docker push <your-registry>/kaxi-ci-runner:1
```

(If you prefer not to build an image, install `kubectl` + `build-essential`
at pod start via the image's `PRE_RUNNER_SCRIPT_PATH` hook — but a baked image
is faster and reproducible.)

---

## Step 4 — Deploy the runner

```yaml
# kaxi-ci-runner.yaml   (apply in slurm-pilot)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kaxi-ci-runner
  namespace: slurm-pilot
spec:
  replicas: 1
  selector:
    matchLabels: { app: kaxi-ci-runner }
  template:
    metadata:
      labels: { app: kaxi-ci-runner }
    spec:
      serviceAccountName: kaxi-ci-runner   # ← ambient in-cluster kubeconfig
      containers:
        - name: runner
          image: <your-registry>/kaxi-ci-runner:1
          env:
            - name: REPO_URL
              value: "https://github.com/OWNER/REPO"
            - name: ACCESS_TOKEN
              valueFrom:
                secretKeyRef: { name: kaxi-ci-runner-gh, key: ACCESS_TOKEN }
            - name: RUNNER_NAME
              value: "kaxi-ci-runner"
            - name: LABELS
              value: "self-hosted,linux,x64,kaxi-slurm"   # ← unique label kaxi-slurm
            - name: RUNNER_SCOPE
              value: "repo"
            - name: RUNNER_WORKDIR
              value: "/tmp/runner-work"
            - name: DISABLE_AUTO_UPDATE
              value: "true"
          resources:
            requests: { cpu: "1", memory: "2Gi" }
            limits:   { cpu: "2", memory: "4Gi" }
```

```bash
kubectl apply -f kaxi-ci-runner.yaml
kubectl -n slurm-pilot logs deploy/kaxi-ci-runner -f   # watch it register
```

The runner uses the pod's mounted ServiceAccount token automatically — `kubectl`
inside the pod resolves the in-cluster API + the RBAC from Step 1 with **no
kubeconfig file**. (The workflow's "Write kubeconfig from secret" step is a
no-op when `SLURM_KUBECONFIG` is unset, which is what you want here.)

---

## Step 5 — Point the gate at the runner (pure config, no workflow edit)

In **repo → Settings → Secrets and variables → Actions → Variables**:

| Variable | Value | Effect |
|----------|-------|--------|
| `SOUNIO_ENABLE_KAXI_SLURM_GATE` | `1` | un-skips the nightly job |
| `KAXI_RUNNER_LABELS` | `kaxi-slurm` | routes the job to this runner (the workflow's `runs-on` reads this var; unset ⇒ `ubuntu-latest`) |

`kaxi-slurm` must match a label you gave the runner in Step 4 and be unique to
it, so the job can't accidentally land on another runner.

---

## Step 6 — Verify

1. The runner shows **Idle** under repo → Settings → Actions → Runners with the
   `kaxi-slurm` label.
2. Manually dispatch: **Actions → "K-AXI PTX Acceptance (nightly, SLURM/GPU)" →
   Run workflow**. Watch it: verify-reachability → emit PTX → build runner →
   submit + wait → `KAXI_JIT_ACCEPT_OK` / `ACCEPTANCE GATE: PASS`.
3. The nightly `cron` (05:00 UTC) then runs unattended.

---

## Notes / teardown

- **Security:** the runner has repo-scoped Actions access and namespaced
  `pods/exec` in `slurm-pilot` only. Treat the PAT and the pod as sensitive
  (anyone who can run workflows can exec in the login pod). Prefer a fine-grained
  PAT limited to this repo; rotate it periodically.
- **Cost/availability:** the job is nightly + manual only and never per-PR, so a
  single replica is fine. If the SLURM partition is busy, `submit_jit.sh --wait`
  honours `WAIT_TIMEOUT` (workflow sets 2400s) and `scancel`s on timeout.
- **Disable:** set `SOUNIO_ENABLE_KAXI_SLURM_GATE` to anything but `1` (or delete
  it) to skip the job; `kubectl -n slurm-pilot delete -f kaxi-ci-runner.yaml`
  to remove the runner.
