# Sounio Loom

Loom is Sounio's native OCaml session kernel and terminal multiplexer. A
recoverable guardian owns the PTY generation, child process, durable output, and
custody journal. The kernel owns semantic input authority and disposable client
connections. Clients or the entire kernel can disappear and recover without
changing the guardian, child, or generation identity.
Persisted descriptors are revalidated against process birth identity: an absent
kernel with a live Guardian is shown as `recoverable`, while loss of both is
shown as `lost` rather than being laundered as an active lane.

## Build

The local build requires OCaml, Dune, findlib, Cryptokit, and OpenSSL. On Debian or
Ubuntu the corresponding packages are:

```sh
sudo apt-get install ocaml-nox ocaml-dune ocaml-findlib libcryptokit-ocaml-dev openssl
scripts/dev/build_sounio_loom.sh
```

The stable `bin/sounio-loom` launcher selects the content-addressed shared
runtime when one is active. Set `SOUNIO_COORD_RUNTIME_MODE=local` to force the
source-worktree build for diagnosis.
`bin/loom` is the product-facing alias; `bin/sounio-loom` remains the compatible
launcher used by existing automation.

## Operate

```sh
bin/sounio-loom start --agent codex --lane experiment -- COMMAND ARG...
bin/sounio-loom recover --agent codex --lane experiment
bin/sounio-loom guardian-status --agent codex --lane experiment
bin/sounio-loom snapshot --agent codex --lane experiment --cursor 0 --meta
bin/sounio-loom fleet-enroll --slot codex-1 --kind codex --home "$HOME" --cwd "$PWD"
bin/sounio-loom fleet-enroll --slot codex-persistent --kind codex \
  --custody loom --home "$HOME" --cwd "$PWD" --prompt-file BOOTSTRAP.md
bin/sounio-loom fleet-reconcile
bin/sounio-loom fleet-reconcile --apply
bin/sounio-loom list
bin/sounio-loom tui
bin/sounio-loom serve --bind 127.0.0.1 --port 8787
bin/sounio-loom export-events-arrow --out loom-events.arrow
bin/sounio-loom verify-events-arrow --file loom-events.arrow
bin/sounio-loom beagle-serve --bind 127.0.0.1 --port 4372
bin/sounio-loom verify-continuity-receipt \
  --receipt PATH --public-key PUBLIC.pem --adapter PATH
bin/sounio-loom attest-continuity-receipt \
  --receipt PATH --subject-public-key SIGNER.pem \
  --observer-private-key OBSERVER-PRIVATE.pem \
  --observer-public-key OBSERVER-PUBLIC.pem \
  --out PATH --adapter PATH
bin/sounio-loom measure-continuity-generation \
  --state-dir PATH --pane-id ID --generation ID \
  --receipt PATH --subject-public-key SIGNER.pem \
  --observer-private-key OBSERVER-PRIVATE.pem \
  --observer-public-key OBSERVER-PUBLIC.pem \
  --out PATH --adapter PATH
bin/sounio-loom journal-authority-serve \
  --socket PATH --state-dir PATH \
  --private-key JOURNAL-PRIVATE.pem --public-key JOURNAL-PUBLIC.pem \
  --epoch 1
bin/sounio-loom journal-authority-status --socket PATH
bin/sounio-loom obligation-list
bin/sounio-loom obligation-tui
bin/sounio-loom obligation-serve --bind 127.0.0.1 --port 8788
bin/sounio-loom obligation-supervise --state-dir PATH
```

## Sovereign Change Kernel

With `SOUNIO_LOOM_SOVEREIGN_CHANGE_MEDIATED=1`, the provider sees both the
worktree and its Git common directory as read-only. `Write`, `Edit`, and
`apply_patch` are redirected to kernel-owned staging outside the worktree. A
single-use in-memory `ChangeGrant` binds the exact tool-call ID, mutation bytes,
worktree and index preimage, authenticated harness peer, and file set. The
kernel materializes only the expected postimage.

The hook accepts only `git commit -m MESSAGE` or `git commit --message MESSAGE`.
The kernel builds a temporary index from the authorized postimages, asks frozen
Sounio action 9044 for `COMMIT_ADMIT` and `CI_ADMIT`, and atomically advances the
branch together with a `refs/loom/change-receipts/<sha256>` receipt ref. Direct,
widened, dynamic, and replayed commit forms fail closed.

CI consumes the pinned receipt without executing or reinterpreting its policy:

```sh
scripts/ci/sounio_loom_sovereign_change_receipt_admit.sh \
  /absolute/change/worktree /absolute/commit.receipt
```

Only after receipt consumption does the separate claim stage ask the frozen
Sounio authority for `CLAIM_READY`. The operational gate exercises a disposable
Git repository, direct-write and direct-commit refusal, exact-tree admission,
receipt tamper, replay, and Python/Rust execution sentinels:

```sh
scripts/ci/sounio_loom_sovereign_change_kernel_operational_selftest.sh
```

## Subprocess Membrane

LOOM has frozen Sounio semantics for a process-tree effect membrane and a Linux
x86_64 OCaml/C diagnostic probe inside a hash-pinned Bubblewrap namespace. The
probe stops root and descendant execution, write-capable opens, and path
mutations for a hash-pinned Sounio decision. Its negative gate verifies that
direct Python, Python hidden behind a shell, a Rust-named executable,
out-of-scope writes, semantic writes, path mutation, inherited writable file
descriptors, and deadline-surviving descendants produce no forbidden external
effect. A sabotage control disables the filesystem observer and proves the
read-only kernel root still protects paths outside the exact writable scope.

This is deliberately not attached to general Bash/Exec, commit, or CI. The
current ptrace realization does not claim a closed syscall algebra or race-free
path mediation. See [SUBPROCESS_MEMBRANE_V1.md](SUBPROCESS_MEMBRANE_V1.md) for
the proved surface, nonclaims, and kernel attachment path.

## Resident Sounio Authority

Actions `9024` and `9025` are the Sounio-first transport and effect-closure
authorities for the resident decision process. Action `9024` binds every
generation to the frozen `9023` parent,
requires strict request/response sequencing and correlation, and makes timeout,
unhealthy transport, or poisoned-generation reuse explicit refusals. Its native
executable, adversarial gate, and content-addressed semantic freeze now exist.
Action `9025` defines absence as a positive, receipt-bound closure certificate:
all twelve effect families need an explicit material coverage mode, unknown
effects must be kernel-denied, and five independent sabotage witnesses must
remain complete. The frozen v2 runtime keeps actions `9023`, `9024`, and `9025`
resident in one stable Sounio process and matches their frozen single-shot
outputs byte for byte. The v1 runtime remains available for compatibility.
An OCaml supervisor now binds one random generation to the hash-pinned runtime,
PID and process birth identity; sequences requests, applies monotonic deadlines,
journals hash-bound receipts, and permanently poisons the generation on replay,
correlation failure, timeout, EOF, or process drift. It cannot alter the frozen
decision bundle. The resident runtime uses a bounded `read_byte()` framer because
the current `lean_single` `read_line()` builtin performs one bulk read rather
than newline framing.

The bounded performance gate measures authority transport separately from
durable audit persistence. On the recorded x86_64 run, 20 resident decisions
took 5,369 microseconds of transport versus 14,097 microseconds for 20
single-shot decisions (2.625x). The same resident run took 837,382
microseconds end to end because its current audit policy performs an `fsync` for
each REQUEST, EFFECT, and RESPONSE receipt. Both numbers are acceptance
evidence; the transport result is not presented as an end-to-end durability
speedup.

The Linux x86_64 diagnostic subprocess probe now opens one resident generation
before admitting the root process and keeps that same Sounio PID, generation,
and monotonic sequence through effect-closure validation, observed effects, and
final outcome. The closure check runs before the diagnostic child and currently
returns `DENY447` because material effect-family coverage is incomplete; the
diagnostic effect may still run only to collect evidence while product
attachment remains refused. Resident startup failure, identity drift, timeout,
EOF, replay, correlation failure, runtime hash drift, or either frozen-manifest
drift fails closed before child execution. Hostile same-UID peer isolation also
remains a hard, explicit blocker. This remains a diagnostic membrane: general
Exec/Bash, commit, and CI attachment are still explicitly false.

The frozen v3 resident adds action `9029` without changing the producer of any
expected result. The OCaml `InvocationCell` kernel validates the exact Sounio
freeze and resident-v3 manifest before spawn, then owns only lifecycle,
correlation, deadlines, receipts, and irreversible poisoning. Its operational
states are `UNPREPARED`, `PREPARED`, `EFFECT_STOPPED`, `CLOSED`, and
`POISONED`. Sounio `DENY481` and `DENY488` leave an unprepared cell without
promoting it; replay, operation mismatch, timeout, EOF, and typed abort poison
the generation and make reuse impossible. The OCaml freeze gate rejects any
encoded `ALLOW`/`DENY481`/`DENY488` oracle and rebuilds the binary twice to the
same digest. This is an operational kernel, not semantic or material authority.

Action `9026` now freezes the stronger kernel-principal contract behind that
blocker. Its C++20 material probe distinguishes a configured subordinate-ID
range, an installed helper, a successful helper exit, and an actually installed
kernel UID/GID map. On the recorded pod, ordinary user namespaces remain bound
to outer UID `1000`, the subordinate-map syscall is refused with `EPERM`, and
the outer account can regain root through passwordless sudo; Sounio therefore
returns `DENY455`. See [KERNEL_PRINCIPAL_V1.md](KERNEL_PRINCIPAL_V1.md).

Action `9027` freezes the principal-lease lifecycle after `9026`: `FREE`,
`RESERVED`, `MAPPED`, `LAUNCHED`, `DRAINING`, `QUARANTINED`, and only then
`FREE` again after affirmative process, namespace, and authority extinction.
The transitory C++20 host broker verifies the exact frozen Sounio executable,
keeps a hash-chained and per-record-synchronized lease journal, and quarantines
uncertain generations after recovery. Its host installer builds immutable
releases addressed by the manifest, broker, and complete operational bundle;
the live root-only probe proves systemd activation while deliberately keeping
`LAUNCH` and `RECYCLE` closed. The same immutable release now pins action `9029`;
its decision-only `ADMIT` operation reaches Sounio but cannot create a lease,
grant, pidfd, barrier, or process. A `HOST_ACTIVATION_PASS` therefore still reports
`material_broker=false` until namespace, cgroup, pidfd, attack, extinction, and
Sounio `ALLOW` gates all execute on the real host. See
[HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md](HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md).

Action `9028` removes the last ambient process-identity shortcut: a lane process
is not denotable inside Loom as a bare PID or numeric pidfd. Frozen Sounio admits
only a non-bearer `PrincipalCapsule` that binds the proposed `9026`/`9027`
launch, PID/start-time identity, namespace and cgroup vector, privilege posture,
broker-only pidfd custody, single-use grant fence, and provenance. A broker
restart may reacquire custody only into `QUARANTINED`; it cannot restore grants,
release a barrier, or resume execution. Capsule bytes name broker custody but
grant no authority on their own. The current Pod returns `DENY472`, and a bare
PID authority witness returns `DENY476`. See
[GARDEN_KERNEL_PRINCIPAL_CAPSULE_V1.md](GARDEN_KERNEL_PRINCIPAL_CAPSULE_V1.md).

Action `9029` freezes the `InvocationCell` semantic join. A valid
`PrincipalCapsule` cannot authorize an unclosed effect stream, and a valid
effect-closure result cannot be reassigned to another principal. Sounio binds
the exact `9028`, `9025`, and transitive `9023` manifests to one principal
generation, membrane generation, event sequence, command, worktree, claim
scope, deadline, and terminal outcome lineage. The four operations are
`PREPARE_ROOT`, `ADMIT_EFFECT`, `CLOSE_OUTCOME`, and irreversible
`ABORT_INCOMPLETE`. Ten single-rule source sabotages prove that each named
refusal is load-bearing; a deliberate Python-oracle frame returns `DENY488`.
The current Pod returns `DENY481` because neither parent has supplied an
`ALLOW` for the same material observation. Cell bytes remain non-authorizing,
and material one-shot custody, hostile same-UID isolation, Exec/Bash, commit,
and CI attachment remain closed. See
[GARDEN_KERNEL_INVOCATION_CELL_V1.md](GARDEN_KERNEL_INVOCATION_CELL_V1.md).

Action `9030` freezes the next authority boundary: the `ExecGrantCell` that
joins action `9029` with pre-execution action `9021` and outcome action `9022`.
The opaque handle is only a lookup coordinate. Sounio requires kernel-derived
peer identity, a kernel-distinct principal, exact Guardian ancestry, pre-write
shape validation, atomic single-use custody, fail-closed crash revocation, and
the same command and generation vector across issue, consume, close, or revoke.
Grant extinction is affirmative evidence, not a table miss: one receipt must
bind observed state absence after a terminal transition, retired generations,
and revoked barrier/descriptor/grant authority. Eleven causal source sabotages
make every rule falsifiable; the deliberate Python-oracle path returns
`DENY499` without reaching its executable sentinel. Current material returns
`DENY491`, and `material_grant`, hostile same-UID isolation, parity, Exec/Bash,
commit, and CI attachment all remain false. See
[GARDEN_KERNEL_EXEC_GRANT_CELL_V1.md](GARDEN_KERNEL_EXEC_GRANT_CELL_V1.md).

The frozen resident v4 adds action `9030` to the same long-lived Sounio process.
Its OCaml `ExecGrantCell` client owns only the operational lifecycle
`VACANT -> ISSUED -> OUTCOME_PENDING -> CLOSED | REVOKED | POISONED`; it loads
the exact action-9030 and resident-v4 manifests before spawn, preserves state on
a semantic denial, and poisons the generation on replay, correlation mismatch,
timeout, EOF, or resident loss. Two deterministic builds and a source scan prove
that OCaml contains no copied Sounio result table. This remains an isolated
operational probe: the existing product `EXEC_ISSUE`, `EXEC_CONSUME`, and
`EXEC_OUTCOME` route is not yet attached.

The frozen resident v5 adds Sounio action `9031` and the non-bearer
`PeerActivationCapsule`. LOOM now executes the frozen `current_material`
projection before creating any session directory, token, daemon, Guardian, or
provider process on real `start`, `provider-start`, `provider-open`, and
`recover` paths. The normal dark observation is `DENY502` and remains
nonauthorizing, so the established lifecycle continues; the positive `seal`
sabotage is instead treated as an unexpected `ALLOW` and refuses before any
session exists. Every observation records hashes for the Sounio semantics and
projection, command, cwd, identities, resident generation, and result. Policy
lookup is anchored to the source binary or its immutable installed capsule, not
to caller-controlled `--cwd`. The installed capsule keeps policy bytes separate
from its writable audit root.

This is a real launch-path observation, not material activation. It does not
yet attach the `ExecGrantCell` to arbitrary Exec/Bash, forbid a Python command
on the general `start` path, or enforce write, commit, and CI boundaries.
`production_activation`, `exec_attached`, `commit_attached`, and `ci_attached`
therefore remain false. See
[GARDEN_PRODUCT_LAUNCH_DARK_ATTACHMENT_V1.md](GARDEN_PRODUCT_LAUNCH_DARK_ATTACHMENT_V1.md).

The host `PrincipalCell` experiment now measures the missing hostile-principal
prerequisite on t560. Two simultaneous systemd `DynamicUser` cells received
distinct UID/GID values and cgroups; reciprocal `kill`, `/proc` memory and fd,
`ptrace`, `process_vm_readv`, `pidfd_send_signal`, and `pidfd_getfd` attacks were
refused even when the attacker inherited a root-opened pidfd for its peer. The
preregistered sabotage retained the same binary, hardening, and distinct
cgroups but assigned both cells the same DynamicUser. `kill(..., 0)` and
`pidfd_send_signal(..., 0)` then became usable in both directions, making the
kernel-distinct principal rule causal rather than correlational. The receipt
still reports `material_grant=false`, `grant_extinction=false`,
`same_uid_peer_isolation=false`, and every product attachment flag false. See
[HOST_EXEC_GRANT_PRINCIPAL_CELL_V1.md](HOST_EXEC_GRANT_PRINCIPAL_CELL_V1.md) and
[HOST_EXEC_GRANT_PRINCIPAL_CELL_SABOTAGE_V1.md](HOST_EXEC_GRANT_PRINCIPAL_CELL_SABOTAGE_V1.md).

The action `9029` OCaml operational kernel and its resident Sounio v3 route are
now frozen in `kernel_invocation_cell.runtime.v1`. The retained adversarial gate
exercises the full positive lifecycle, typed abort, current-material and
Python-oracle refusals, replay, operation mismatch, timeout, EOF, receipt
binding, and manifest/runtime tamper before spawn. It does not open material
invocation or attach the cell to general Exec/Bash, commit, or CI.

The transitory C++ broker now has a separately frozen action `9029` admission
adapter in `kernel_invocation_cell.material.v1`. The same broker binary routes a
positive fixture to Sounio `ALLOW` and the current Pod observation to Sounio
`DENY481`; it contains neither the `481` result nor its reason. Manifest,
authority, and multiline-frame drift fail closed, while the live broker still
refuses `LAUNCH` and `RECYCLE`. The host bundle exposes that adapter as
root-controller-only `ADMIT` and reports all six action `9027`/`9028`/`9029`
artifact hashes. This freezes `MATERIAL_PARITY`, not material
execution: no grant, pidfd, barrier, or process is created, and hostile
same-UID isolation remains false. See
[INVOCATION_CELL_MATERIAL_ADMISSION_V1.md](INVOCATION_CELL_MATERIAL_ADMISSION_V1.md).

The host bundle can now cross the Kubernetes/host boundary as a deterministic,
content-addressed promotion capsule. The source worktree rebuilds and freezes
the Sounio authorities once; the host verifies and installs those exact bytes
rather than recompiling semantics with a different checkout or toolchain. A
strict inner inventory remains effective even if a test recomputes the outer
archive hash, and a causal sabotage proves the pre-execution refusal of a
Python host-gate oracle. The Beagle hostPID transport does not remount the live
`ReadWriteOncePod` workspace PVC and carries no semantic authority. Host
promotion still leaves `PARITY_OPEN`, `CLAIM_READY`, `LAUNCH`, material
invocation, and same-UID peer isolation closed. See
[HOST_PROMOTION_CAPSULE_V1.md](HOST_PROMOTION_CAPSULE_V1.md).

The main Loom build reconstructs all frozen resident runtime generations with
the frozen Sounio toolchain, verifies each frozen digest, installs it under a
`sha256-<digest>` directory, and atomically switches the stable runtime symlink
under a filesystem lock. A live generation therefore keeps its original
executable inode while a concurrent build stages the same content-addressed
generation.

The gate includes two single-rule sabotages. Removing only the frozen-parent
rule admits an unchanged orphan request, and removing only strict progression
admits an unchanged replay. See
[GARDEN_RESIDENT_AUTHORITY_V1.md](GARDEN_RESIDENT_AUTHORITY_V1.md) for the
protocol, evidence stages, and nonclaims.

`serve` is read-only and binds to loopback by default. A non-loopback bind is
refused unless `--allow-remote` is explicit. The session directory and token are
local capabilities and must remain private to the owning user.

The Fusion cockpit keeps three authorities separate. `/api/fleet` overlays the
coordination runtime's lightweight `cockpit-snapshot` with Loom session
descriptors, so a lane can be live on the bus, reachable through an agentd or
tmux endpoint, and still have no Loom PTY custody. Only `loom_state=active` or
`loom_state=recoverable` means Loom owns durable terminal continuity. The
machine snapshot never exports endpoint addresses, sockets, token paths,
message bodies, or provider prompts. `/api/events` likewise emits only verified
journal metadata and refuses unverified histories.

## Spectral Data Plane

`/api/events.arrow` and `export-events-arrow` expose the verified semantic and
Guardian event histories as an Apache Arrow IPC stream. The stream is a derived,
read-only projection: the append-only journals remain the authority, and one
invalid journal refuses the complete projection instead of yielding a partial
or apparently verified table.

The `loom-spectral-events-v1` schema keeps proof material columnar alongside the
event data. Its 13 non-null columns include the lane and generation identity,
journal domain, sequence, UTC observation string, event kind, binary payload,
the previous and current event digests, the verified journal head, and the
explicit `verified` bit. SHA-256 values use Arrow `fixed_size_binary[32]`, not
hex strings, so analytical and WebGPU consumers receive a compact physical
representation without losing the hash-chain boundary.

The OCaml runtime constructs the projection through a small C FFI over the
vendored Apache nanoarrow 0.9.0 IPC writer. Python and Rust are not runtime or
build dependencies. The HTTP response uses
`application/vnd.apache.arrow.stream` and declares
`X-Loom-Authority: verified-derived`; corruption or journal verification failure
returns a refusal rather than a fallback JSON dataset.

Projection is schema-evolution aware without weakening current authority. The
known pre-Guardian runtime `2026.08.24.0` has no Guardian journal by design, so
its verified semantic history is projected with the explicit
`semantic-only-legacy` profile. That exception also requires a terminal
hash-verified journal whose `SESSION_STARTED` receipt predates the Guardian
release, no `guardian.tsv` in the generation, and an agreeing generation
snapshot when one exists. A descriptor-only runtime downgrade therefore
refuses instead of laundering modern history. HTTP receipts expose both
`X-Loom-Guardian-Sessions` and
`X-Loom-Legacy-Semantic-Only-Sessions`. A current or unknown runtime that omits
`guardian_journal_file`, or any descriptor that names a missing Guardian
journal, still refuses the complete projection. `/api/events` carries the same
per-session `journal_profile`, keeping JSON and Arrow consumers on one
authority boundary.

This split is intentional:

```text
control plane   = journals, leases, ACKs, recovery, authority
spectral plane  = Arrow IPC batches for scan, visualization, ML, and WebGPU
```

The Arrow plane is not a transaction log, recovery source, or authority store.
Its schema is an interoperability contract for disposable projections.

## Epistemic Machine v0

Loom can now persist a bounded epistemic worldline independently of any one
provider CLI or UI process. A worldline records observations with five separate
`Knowledge` axes, evidence-bearing claims, explicit falsifier challenges,
exclusive write capabilities, and counterfactual forks bound to a verified
parent head.

```sh
bin/loom world-create --state-dir STATE --world alpha \
  --agent codex --lane experiment
bin/loom knowledge-observe --state-dir STATE --world alpha \
  --knowledge k1 --value 42.0 --error 0.01 \
  --uncertainty interval-0.2 --confidence 0.91 --provenance SHA256
bin/loom epistemic-claim-open --state-dir STATE --world alpha \
  --claim c1 --knowledge k1 --evidence SHA256
bin/loom epistemic-claim-challenge --state-dir STATE --world alpha \
  --claim c1 --challenge x1 --falsifier SHA256
bin/loom epistemic-capability-acquire --state-dir STATE --world alpha \
  --capability cap1 --resource PATH --owner codex --generation generation-1
bin/loom world-fork --state-dir STATE --parent alpha --child beta \
  --agent grok --lane hostile-review --hypothesis 'the mechanism is false'
bin/loom world-verify --state-dir STATE --world alpha
bin/loom world-list --state-dir STATE
```

The OCaml reducer replays the SHA-256 journal, resolves object references,
enforces one live capability per exact resource across all local worldlines,
and binds every fork to an observed parent head. Before appending an event it
asks native Sounio frame `9008` to admit the nominal transition. A named
sabotage test removes only `knowledge_axes_are_bound` and proves that this rule,
rather than an incidental parser failure, is what rejects an incomplete
observation.

Epistemic worldline events also appear in `loom-spectral-events-v1` with
`journal=epistemic-worldline`. That Arrow stream remains a verified-derived
projection: journal corruption refuses export and cannot be used to launder a
damaged history into a visually plausible table. The v0 boundary is one local
filesystem authority. It does not claim distributed consensus, deterministic
LLM reruns, physical causality, or exactly-once external effects.

## Counterfactual Attention Compiler v0

The first attention compiler chooses one next experiment from a canonical TSV
candidate set and persists both the complete set and the decision in the
epistemic worldline. It keeps information gain, falsification power,
counterfactual divergence, risk, and cost separate instead of hiding them in a
weighted score.

```text
candidate_id target_world claim provider resource information falsification divergence cost risk evidence_sha256 falsifier_sha256
```

The actual file is tab-separated and begins with that exact header. It accepts
at most 64 unique candidates. The three explicit policies are
`information-first`, `falsification-first`, and `counterfactual-first`; lower
risk and then lower cost break axis ties, followed by stable candidate ID.
Candidates whose integer cost exceeds the budget are infeasible.

```sh
bin/loom attention-compile --state-dir STATE --world scheduler \
  --plan plan-1 --candidates candidates.tsv --budget 100 \
  --policy falsification-first --owner claude --generation generation-1
bin/loom attention-complete --state-dir STATE --world scheduler \
  --plan plan-1 --owner claude --generation generation-1 \
  --outcome SHA256
```

Native Sounio frame `9009` checks the selected candidate against every feasible
rival with `attention_selected_not_dominated`. The compile event atomically
reserves the selected exact resource; completion with the same owner and
generation releases it. This makes ordinary capabilities and attention plans
mutually exclusive across all local worldlines without a scheduling-receipt to
capability-acquire crash window. Replay recompiles the stored candidate set,
repeats all native pair checks, validates every target world and claim, and
refuses the entire Arrow projection on divergence or journal damage.

These axes are bounded author estimates and the policy order is governance, not
an objective law of scientific value. V0 proves deterministic local replay and
resource exclusion, not optimal research strategy, fairness, starvation
freedom, distributed scheduling, or model reliability.

## Pareto Portfolio Attention Compiler v0

The portfolio compiler chooses a compatible set of experiments rather than a
single winner. For 1 through 18 candidates it enumerates every nonempty subset,
rejects subsets that repeat an exact resource or exceed token, wall, GPU, or
quota budgets, retains the complete eight-axis Pareto frontier, and selects one
frontier portfolio under the same explicit policy family.

```text
candidate_id target_world claim provider resources information falsification divergence token_cost wall_cost gpu_cost quota_cost risk evidence_sha256 falsifier_sha256
```

The file is tab-separated. `resources` is a comma-separated, sorted, unique set
of exact identities. Token and wall costs are positive; GPU and quota costs may
be zero. Risk and all four costs are minimized while information,
falsification, and divergence are maximized. Sorted candidate IDs are the final
tie-break, never an implicit weighted score.

```sh
bin/loom attention-portfolio-compile --state-dir STATE --world scheduler \
  --portfolio wave-1 --candidates portfolio.tsv \
  --token-budget 8000 --wall-budget 600 --gpu-budget 2 --quota-budget 4 \
  --policy information-first --owner codex --generation generation-1
bin/loom attention-portfolio-complete --state-dir STATE --world scheduler \
  --portfolio wave-1 --owner codex --generation generation-1 \
  --outcome SHA256
```

One compile event atomically reserves the selected resource union; a refusal
reserves none of it. Completion releases the same union. The reducer recomputes
the feasible subsets, frontier, selected set, and domain-separated digests from
the canonical candidate set on every replay. It then asks native Sounio frame
`9010` to compare the selected aggregate with every frontier rival. The named
`portfolio_selected_not_dominated` sabotage control admits the exact dominated
frame that the production rule refuses, identifying that comparator as
load-bearing.

V0 materializes at most 256 skyline members and 1 MiB of canonical frontier.
It refuses a larger working skyline before journaling rather than silently
truncating or approximating it, so accepted decisions retain exact semantics.

The verified event and canonical frontier flow into the existing Arrow spectral
plane for UI, WebGPU, and analysis without making Arrow scheduling authority.
This is a bounded exact local compiler over declared integer estimates. It does
not establish that epistemic value is additive: correlations, redundancy, and
submodular effects between experiments are outside v0. It also does not
establish objective novelty, calibrated utility, distributed resource truth,
consensus, fairness, or globally optimal science.

## Robust Contingent Policy Compiler v0

The contingent compiler selects an action now and a different continuation for
each declared nominal outcome. It compiles a bounded acyclic action/outcome DAG
into an exact history-conditioned policy tree using backward induction. This is
the adaptive object that the open-loop portfolio compiler deliberately does not
claim to produce.

```text
state action_id target_world claim provider resource information falsification divergence token_cost wall_cost gpu_cost quota_cost risk evidence_sha256 falsifier_sha256
action_id variant_index variant_count outcome_id successor_state branch_evidence_sha256
```

Both files are tab-separated and begin with those exact headers. For every
action, outcome variants must be indexed exactly `0..variant_count-1`; gaps,
duplicates, and count drift refuse the entire graph. `successor_state=-` is a
terminal branch. All declared states must be reachable from the root and the
graph must be acyclic.

```sh
bin/loom contingent-policy-compile --state-dir STATE --world scheduler \
  --contingent-policy policy-1 --root-state start \
  --actions actions.tsv --outcomes outcomes.tsv \
  --token-budget 8000 --wall-budget 600 --gpu-budget 2 --quota-budget 4 \
  --order information-first --owner codex --generation generation-1
bin/loom contingent-policy-observe --state-dir STATE --world scheduler \
  --contingent-policy policy-1 --outcome observed-variant \
  --owner codex --generation generation-1 --outcome-digest SHA256
```

At each action, information, falsification, and divergence are the immediate
value plus the minimum continuation value across its mutually exclusive
branches. Risk and the four costs are the immediate burden plus the maximum
continuation burden. Local Pareto pruning preserves the complete achievable
non-dominated value frontier; the chosen retained witness uses the explicit
information, falsification, or counterfactual order. It may retain multiple
equal-value trees, but does not claim to enumerate every structurally distinct
tree that a parent min/max bottleneck maps to an already represented vector.
No probabilities, expectation, weighted score, submodularity, or approximation
theorem are implied.

Compile atomically reserves only the root action's exact resource. An observed
outcome atomically releases that resource and either reserves the exact next
action resource or completes the policy. A failed handoff appends nothing and
keeps the current resource live; future-branch resources are never reserved in
advance. Replay rebuilds the graph, frontier, selected tree, and observed route,
then rechecks native Sounio frame `9011`. Its named rules separately establish
non-domination, total nominal partitions, and exact branch routing.

This handoff is serialized by the state directory's exclusive `machine.lock`,
which remains held across replay, next-resource validation, verified append,
fsync, and post-append replay. Concurrent contenders therefore act as a
compare-and-swap on the derived live-resource set; the precheck is not trusted
outside that critical section.

V0 accepts at most 8 reachable states, 18 actions, and 3 outcomes per action. It
refuses after 65,536 constructed policies, after a working frontier reaches 257
members, or when canonical policy/frontier text exceeds 1 MiB. These are
fail-closed exact bounds, not silent truncation. Completeness is only over the
declared nominal outcome variants. Mapping a physical observation to one of
those variants, calibrating the declared estimates, causal validity,
distributed resource truth, fairness, and global scientific optimality remain
outside the compiler's evidence authority.

## Signed outcome evidence authority v0

A contingent policy can commit separate measurement and classification
authorities at compile time:

```sh
bin/loom contingent-policy-compile --state-dir STATE --world scheduler \
  --contingent-policy policy-1 --root-state root \
  --actions ACTIONS.tsv --outcomes OUTCOMES.tsv \
  --token-budget 8000 --wall-budget 600 --gpu-budget 2 --quota-budget 4 \
  --order information-first --owner codex --generation generation-1 \
  --measurement-principal instrument-a \
  --measurement-public-key measurement-public.pem \
  --classifier-principal classifier-a \
  --classifier-public-key classifier-public.pem \
  --classifier-spec-digest SHA256

bin/loom contingent-measurement-attest --state-dir STATE --world scheduler \
  --contingent-policy policy-1 --measurement measurement.bin \
  --measurement-principal instrument-a \
  --measurement-private-key measurement-private.pem \
  --measurement-nonce measurement-1 --receipt measurement.receipt

bin/loom contingent-classification-attest --state-dir STATE --world scheduler \
  --contingent-policy policy-1 --measurement-receipt measurement.receipt \
  --outcome observed-variant --classifier-principal classifier-a \
  --classifier-private-key classifier-private.pem \
  --receipt classification.receipt

bin/loom contingent-policy-observe-attested --state-dir STATE \
  --world scheduler --contingent-policy policy-1 \
  --measurement-receipt measurement.receipt \
  --classification-receipt classification.receipt \
  --owner codex --generation generation-1
```

The three committed principals, policy owner, measurer, and classifier, must be
pairwise distinct, and the two Ed25519 keys must differ. The measurement signs
the exact bytes plus the replay-derived policy, cursor, action, path,
generation, and current journal head. The classifier signs the complete
measurement receipt digest, the precommitted classifier specification, the
recomputed current nominal partition, and one outcome at that same machine
coordinate. Loom reloads and recomputes those bindings under `machine.lock`,
checks native frame `9012`, independently checks route frame `9011`, then stores
both receipts and the transition in one hash-chained event.

The exact journal head acts as a local compare-and-swap coordinate, so an
unrelated intervening append stales both receipts even if the policy cursor did
not move. The carried measurement nonce provides correlation and receipt-digest
uniqueness; it is not a global replay ledger. A strict signed policy cannot fall
back to the legacy opaque outcome-digest command, while legacy policies remain
supported in their separate mode.

This path proves authorization and state binding within one filesystem
authority. It does not prove physical measurement truth, classifier accuracy,
private-key custody, process isolation, or organizational independence. By
itself it also does not resist rollback of an entire internally consistent
state directory. The optional Witness Mesh below adds a bounded external
monotonic checkpoint; the signed outcome path alone retains no rollback claim.

## Witness Mesh v0

Witness Mesh moves the epistemic journal's monotonic high-water mark outside
the protected local state directory. Configure one Ed25519 anchor authority,
exactly three distinct Ed25519 witness members, and their current network
endpoints. All four keys must be distinct:

```text
anchor_public_key<TAB>/anchor/authority-public.pem
witness_id<TAB>public_key
w1<TAB>/authority-a/witness-public.pem
w2<TAB>/authority-b/witness-public.pem
w3<TAB>/authority-c/witness-public.pem
```

```text
witness_id<TAB>host<TAB>port
w1<TAB>witness-a.internal<TAB>9441
w2<TAB>witness-b.internal<TAB>9441
w3<TAB>witness-c.internal<TAB>9441
```

Each authority runs a separately stateful service:

```sh
bin/loom witness-serve --witness-state-dir AUTHORITY_STATE \
  --membership membership.tsv --witness w1 \
  --private-key witness-private.pem --bind 127.0.0.1 --port 9441
```

The client anchors and strictly verifies one epistemic world:

```sh
bin/loom witness-mesh-anchor --state-dir STATE --world scheduler \
  --membership membership.tsv --endpoints endpoints.tsv \
  --anchor-private-key /anchor/authority-private.pem
bin/loom witness-mesh-verify --state-dir STATE --world scheduler \
  --membership membership.tsv --endpoints endpoints.tsv
# Availability-oriented verification for non-equivocating witnesses:
bin/loom witness-mesh-verify --state-dir STATE --world scheduler \
  --membership membership.tsv --endpoints endpoints.tsv --policy crash-quorum
```

An anchor signs and sends each service the literal journal suffix from that
service's last signed `(event_count, head)` to the proposed checkpoint. The
service authenticates the anchor request, reparses every event, requires a
contiguous predecessor, persists its canonical Ed25519 receipt before replying,
and makes an exact retry of its latest request idempotent. Each witness-signed
receipt carries the independently verifiable anchor authorization. Native
Sounio frame `9013` admits only two matching shares from a fixed three-member
set. The local certificate chain binds each preceding certificate, but local
self-consistency is never sufficient authority.

Anchoring and `crash-quorum` verification require 2/3, so one unavailable
non-equivocating service does not stop that mode. The default
`byzantine-strict` verification requires all 3/3 current receipts. It therefore
fails closed when any witness is unavailable, but detects a stale/equivocating
answer from one dishonest witness because the other two current states are also
required. A restarted lagging witness replays its entire missed raw suffix from
its own retained head. If the client dies after external quorum persistence but
before the local certificate write, the next client completes or reuses the
signed next-sequence receipts and reconstructs the missing certificate without
rolling a witness backward.

The guarantee is deliberately policy-bounded. `crash-quorum` detects rollback
or a fork through the latest checkpoint only under non-equivocation/state
retention; it does not tolerate a Byzantine witness as the sole intersection of
two 2/3 quorums. `byzantine-strict` detects the concrete one-dishonest-witness
rollback attack while all three services are reachable, but it is still not a
consensus protocol and makes no progress claim for a silent or refusing
dishonest witness. An unanchored suffix is refused by either verifier but is not
rollback-protected. V0 is not Byzantine consensus, general f=1 Byzantine fault
tolerance, threshold signatures, TLS, confidentiality, trusted time, membership
rotation, hardware key custody, or organizational independence. A membership
change invalidates the existing chain and requires a new world/epoch with
isolated witness state. Frames are bounded to 4 MiB and raw segments to 1 MiB;
authenticated plaintext still exposes traffic and permits delay, loss, replay,
and denial of service, but an unauthenticated peer cannot advance witness state
without the anchor key. Witness signatures establish journal continuity and
keyholder authorship, not physical or scientific truth, and MUST NOT by
themselves authorize clinical, dosing, or classifier decisions. The exact
semantic and falsification boundary is in
`docs/internal/concepts/loom-witness-mesh.contract`.

## Witness Mesh v1

V1 makes the availability/safety trade explicit in the membership schema and
uses four fixed witnesses with a 3-of-4 quorum. The first line selects v1; a v0
membership file remains byte-for-byte compatible with the previous parser:

```text
schema<TAB>loom-witness-membership-v1
anchor_public_key<TAB>/anchor/authority-public.pem
witness_id<TAB>public_key
w1<TAB>/authority-a/witness-public.pem
w2<TAB>/authority-b/witness-public.pem
w3<TAB>/authority-c/witness-public.pem
w4<TAB>/authority-d/witness-public.pem
```

The endpoint file has the same header as v0 and one row for each of `w1` through
`w4`. The existing `witness-serve`, `witness-mesh-anchor`, and
`witness-mesh-verify` commands select v1 from the membership schema. Native
Sounio frame `9014` consumes three distinct matching linear shares. Its five
rules independently check member separation, canonical 3-of-4 quorum flags,
membership binding, checkpoint agreement, and strict monotonic advance.
Both v1 verification policies therefore require 3-of-4. `crash-quorum` retains
the non-equivocation/state-retention claim, while default `byzantine-strict`
uses the `f <= 1` honest-intersection argument below.

For any two quorums `Q1` and `Q2` of size three in a fixed four-member universe,
`|Q1 intersection Q2| >= 3 + 3 - 4 = 2`. If at most one member is dishonest,
that intersection contains at least one honest member. Distinct 3-subsets
intersect in exactly two members; identical quorums intersect in all three.
Consequently, under
fixed membership, retained honest state, verified signatures, and `f <= 1`, a
3-of-4 current-status verification cannot accept a rollback view whose only
bridge to the anchoring quorum is dishonest. Unlike v0 strict mode, v1 can both
anchor and strictly verify with one unavailable member when the other three
return valid matching receipts.

The executable attack control advances `w1,w2,w3`, leaves `w4` behind, makes
`w1` unavailable, rolls the dishonest `w2` back to a valid older signed state,
and keeps honest `w3` current. Verification refuses the rolled local view
because the required 3-of-4 response contains that current honest intersection.
The recovery control then catches `w2` and `w4` up from their own retained
predecessors and verifies with `w1` still unavailable.

This is a bounded honest-intersection checkpoint protocol, not a general BFT
consensus protocol. There is one configured anchor authority, no leader
election, view change, asynchronous liveness proof, dynamic reconfiguration, or
state transfer by assertion. Progress requires three protocol-valid matching
responses; network reachability alone is insufficient. A dishonest refusal or
invalid response plus another unavailable service halts it. V1 also does not
provide TLS, confidentiality, trusted time, threshold signatures, HSM custody,
organizational independence, protection after two dishonest/rolled-back
witnesses, or truth of the journal payload. Membership rotation starts a new
world or epoch. The same 4 MiB frame and 1 MiB raw-segment bounds apply.
A dishonest witness may also force a fail-closed denial of service with a stale,
future, malformed, or otherwise nonmatching signed status; no Byzantine
liveness claim is made.

## Proof-Carrying Witness Epoch Handoff v0

Epoch handoff rotates a fixed Witness Mesh v1 membership without mutating the
old configuration in place. Both old and new roots must independently verify a
current 3-of-4 checkpoint over the same world, event count, and journal head.
The memberships, root identities, and boundary certificate digests must differ,
and the epoch number must increase by exactly one.

```sh
bin/loom witness-epoch-handoff \
  --epoch-state-dir EPOCH_STATE --world scheduler \
  --from-epoch 1 --to-epoch 2 \
  --old-state-dir OLD_STATE --old-membership OLD_MEMBERSHIP \
  --old-endpoints OLD_ENDPOINTS \
  --new-state-dir NEW_STATE --new-membership NEW_MEMBERSHIP \
  --new-endpoints NEW_ENDPOINTS
bin/loom witness-epoch-verify \
  --epoch-state-dir EPOCH_STATE --world scheduler \
  --active-state-dir NEW_STATE --membership NEW_MEMBERSHIP \
  --endpoints NEW_ENDPOINTS
```

Native Sounio frame `9015` consumes a linear verified-old-quorum token and a
separate verified-new-quorum token. Both tokens carry their epoch number, and
admission requires an exact `verified=1`, `required=3`, `members=4` summary
under the explicit host-trust effect. The resulting private linear receipt
carries both membership/root/certificate
summaries through consumption. Its seven named rules bind both exact
`3/4` configurations, adjacent epochs, distinct memberships and roots, one
shared checkpoint, distinct boundary certificates, and a well-formed
predecessor digest. The OCaml verifier retains and replays both complete frame
`9014` certificate chains plus the frame `9015` handoff chain; embedded receipt
bytes never substitute for missing retained state.

Preparation durably writes one deterministic canonical handoff receipt before
activation. The active epoch is a separate fsync-and-rename pointer. A crash in
the first window reuses the prepared receipt without publishing an epoch; a
crash after pointer replacement returns the already active transition as an
idempotent retry. Existing conflicting prepared or active state refuses rather
than being overwritten.

This is a proof-carrying joint-authority transition, not dynamic Byzantine
consensus. It has no leader election, view change, asynchronous liveness,
automatic state transfer, threshold signature, or progress claim below three
valid old and three valid new statuses. Epoch one is an explicit bootstrap
boundary. This v0 supports `from_epoch` 1 through 64, giving at most 64
handoffs and active epochs through 65; a longer history requires a future
compaction or roll-forward authority. The old/new roots, memberships,
certificates, and status receipts must remain available.

Process-crash recovery is executable; adversarial filesystem freshness remains
a custody assumption. If the epoch-control directory can be rolled back with
all of its receipts, a self-consistent old active pointer can replay. Deploy it
under a monotonic authority outside the protected roots' rollback domain before
claiming rollback resistance. The bounded semantic and falsification contract
is `docs/internal/concepts/loom-witness-epoch-handoff.contract`.

## Witness Epoch Transparency v0

Epoch transparency moves the handoff history outside the epoch-control
directory's rollback domain. A signed append-only operator stores the canonical
handoff journal, while a fixed Witness Mesh v1 council independently anchors
the exact tree size and RFC6962-style Merkle root. The operator is storage, not
truth: verification replays the complete bounded journal, recomputes every leaf
and internal-node hash, checks the operator signature, and queries a fresh
3-of-4 witness quorum before accepting the active epoch.

```sh
bin/loom witness-epoch-log-serve \
  --log-state-dir LOG_STATE --operator log-operator \
  --operator-public-key operator-public.pem \
  --operator-private-key operator-private.pem \
  --publisher-public-key publisher-public.pem \
  --bind LOG_ADDRESS --log-port 9442

bin/loom witness-epoch-transparency-publish \
  --epoch-state-dir EPOCH_STATE --transparency-state-dir TRANSPARENCY_STATE \
  --world scheduler --log-host LOG_HOST --log-port 9442 \
  --operator log-operator --operator-public-key operator-public.pem \
  --publisher-public-key publisher-public.pem \
  --publisher-private-key publisher-private.pem \
  --transparency-membership transparency-membership.tsv \
  --transparency-endpoints transparency-endpoints.tsv \
  --transparency-anchor-private-key transparency-anchor-private.pem

bin/loom witness-epoch-transparency-verify \
  --epoch-state-dir EPOCH_STATE --transparency-state-dir TRANSPARENCY_STATE \
  --world scheduler --log-host LOG_HOST --log-port 9442 \
  --operator log-operator --operator-public-key operator-public.pem \
  --transparency-membership transparency-membership.tsv \
  --transparency-endpoints transparency-endpoints.tsv
```

Native Sounio frame `9016` consumes separate linear proofs for the verified
frame `9015` handoff, signed reachable log, current witness quorum, independent
operator principal, monotonic append, materialized-prefix inclusion, exact leaf
binding, and latest-epoch agreement. Its eight named rules are each covered by
a same-frame necessity control: the unmodified rule refuses the attack and
replacing only that rule with `true` admits the identical frame under the test
harness. The private
host receipt cannot be constructed by callers, and the quorum proof cannot be
reused.

Publishing requires the active handoff to be the newest leaf: for tree size
`n`, the appended transition must have index `n` and `to_epoch = n + 2` after
the explicit epoch-one bootstrap. A crash after operator append but before
witness anchoring is recovered by retrying the exact persisted leaf; a
conflicting retry refuses. The bounded v0 journal supports at most 64
transitions and deliberately uses a full materialized prefix instead of a
compact consistency proof.

Production verification refuses an operator that resolves to the local host.
The environment variable
`SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1` exists only for
deterministic local gates and labels its output
`custody=SIMULATED_NOT_CLAIMED`; it is not evidence of external custody.

The proven claim is narrow: a verifier connected to the signed operator and a
fresh 3-of-4 council refuses rollback below the latest quorum-witnessed epoch,
signed split views, forged inclusion, reordered or dropped handoffs,
host-principal collapse, stale signed tree heads, and an unreachable log. The
cross-node gate runs the client, node-local operator storage, and four witness
services on five distinct Slurm hosts, stages checksum-verified artifacts, and
accepts only `custody=EXTERNAL_HOST`; see
`scripts/ci/sounio_loom_witness_epoch_transparency_cross_node_selftest.sh`.

This is not a freeze detector, availability protocol, consensus service,
trusted-time system, compact transparency log, or backup of the log bytes. It
makes no safety claim with two dishonest witnesses, under a full partition, or
when the operator and at least two witnesses collude. The bounded contract is
`docs/internal/concepts/loom-witness-epoch-transparency.contract`.

When a session has exited, `snapshot` falls back to terminal offline replay. It
accepts that path only after both the semantic and Guardian journals reach their
terminal states, the Guardian cursor equals the durable output length, and every
output chunk still matches the SHA-256 digest recorded by the Guardian. The
returned range is assembled from those verified chunks. A same-length mutation
of `output.bin` is therefore refused by `guardian-output:digest-mismatch`, not by
an incidental cursor or file-length check. This is an integrity check, not a
monotonic-freshness proof against replaying an older, internally consistent
output and journal pair.

## Provider ABI v1

Loom owns the public control surface while provider CLIs remain isolated native
adapters. The first ABI exposes:

```sh
bin/loom provider-list --json
bin/loom provider-status --provider codex --json
bin/loom provider-plan --provider codex --session-id UUID --cwd DIR \
  --prompt-file PROMPT --isolate-context --json
bin/loom provider-start --provider codex --agent codex --lane work \
  --session-id UUID --cwd DIR --prompt-file PROMPT --isolate-context
bin/loom provider-open --provider codex --agent codex --lane persistent-work \
  --session-id UUID --cwd DIR --prompt-file PROMPT
bin/loom provider-open --provider kimi --agent kimi --lane persistent-work \
  --session-id UUID --cwd DIR --prompt-file PROMPT
bin/loom provider-auth-login --provider codex
```

`provider-plan` never emits the raw prompt. It publishes its byte length and
SHA-256 digest and replaces the argv value with a digest-bearing placeholder.
Provider credentials stay under the provider's native authority; Loom invokes
the native status or login operation and never reads or copies token files.
Dangerous auto-approval flags are absent unless `--unsafe-auto` is explicit.

`provider-start` executes the selected CLI without a shell through an internal
OCaml trampoline that closes stdin and removes inherited Codex, Claude, Kimi,
Cursor, Grok, tmux, and Sounio agentd/session identity variables while retaining
native provider credentials. `--isolate-context` maps provider-specific
reductions in memory, rules, and subagent context; it is deliberately not
described as a sandbox. The provider remains the Guardian-owned child, so kernel
recovery does not replace its process identity. Codex, Claude Code, Kimi Code,
Grok, and OpenCode are supported. Their stream, authentication, and
session-binding differences remain typed rather than being flattened into a
false common denominator. The complete contract and current boundaries are in
`tools/loom/PROVIDER_ABI_V1.md`.

`provider-open` is the persistent counterpart. Its provider stdin remains
attached to the Guardian-owned PTY and is reachable only through Loom's
exclusive input lease or authenticated wake transport. Codex takes its initial
prompt in the native TUI argv; Kimi starts with a prompt-free argv and receives
its bootstrap through the authenticated lease because its TUI exposes no
positional prompt contract. Persistent adapters for other providers and
context-isolation requests that the native TUIs cannot honor fail closed.
Killing and recovering the disposable Loom kernel preserves the Guardian,
provider process, instance identity, conversation, and durable output cursor.
A woken lane can answer with
`bin/sounio-coord reply --agent A --lane L --reply-to MESSAGE_ID --message TEXT`;
the command derives the original sender and thread instead of requiring the
provider to reconstruct routing metadata.

The kernel treats claim refresh, process presence, and the Loom delivery
endpoint as one convergent registration attempt. A failed operation exits the
refresh child nonzero; the parent retries after lane-jittered delays of 1, 2, 4,
8, 16, and at most 30 seconds. Only a complete endpoint registration restores
the five-minute steady-state refresh period. A transient lock refusal or an
endpoint-specific refusal therefore cannot silently leave a live lane
unreachable for the rest of its lease. The integration gate holds the
coordination lock for the first attempt, refuses the first endpoint registration
after presence succeeds, and requires the same kernel to converge without
terminal input.

Native lifecycle hooks are rooted in the active shared runtime, not in a
possibly old worktree. The hook command resolves the content-addressed runtime
behind `current` once and uses both its OCaml executable and its frozen Sounio
language-authority capsule. The capsule carries the exact manifest, source, and
entrypoint bytes with independent hashes in the runtime manifest. Worktrees
that predate the LOOM product can therefore join the same native control plane
without Python and without importing semantics from OCaml. Missing or modified
capsule bytes, a broken activation link, or a bundle switch during attestation
fails closed. Each hook receipt identifies whether authority came from the
worktree or the immutable runtime capsule.

Native hook adoption across a control checkout uses
`scripts/dev/install_sounio_loom_native_hooks.sh --target-root PATH --activate`.
It freezes the candidate bytes, holds the target Git index against branch
switches, keeps an out-of-worktree backup, swaps both provider configurations
atomically, and runs a production-mode policyless canary before returning. A
failed canary restores both files. The installer is operational machinery only;
the ALLOW decisions and semantic hashes still come from the packaged Sounio
authority.

## Durable Obligations

A directed coordination `request` is now projected into a durable Loom
obligation automatically. The message file digest becomes the immutable work
identity. New requests carry `obligation_schema=loom-durable-obligation-v1`, so
reconciliation ignores historical messages created before this contract. Set
`SOUNIO_COORD_DURABLE_OBLIGATIONS=0` only when deliberately exercising the
legacy message-only fallback. The shared installer also writes an immutable
`loom-obligation-activation.v1` watermark beside coordination state. Exactly
directed requests emitted by pre-upgrade clients after that epoch are reconciled
as obligations even when their stale launcher cannot write the schema field;
older historical requests remain outside the obligation system. A current client
using the fallback writes `obligation_opt_out=1`, making that exclusion explicit
and auditable instead of indistinguishable from a stale client. Inbox hooks and
the obligation supervisor run the bridge, so enforcement lives in shared
authority rather than in the worktree-local launcher. `obligation-consume`
binds acceptance to a verified process generation under its own expiring lease;
`obligation-claim` creates one renewable exclusive claim;
`obligation-interrupt` fences that claim; `obligation-recover` changes the
generation; and `obligation-complete` requires distinct, non-empty outcome and
evidence files. The OCaml runtime replays the complete hash chain under a file
lock, asks native Sounio frame `9007` to admit each transition, and fsyncs the
event before reporting success.

`obligation-supervisor-ensure` owns the tmux-free service lifecycle. It takes a
state-local bootstrap lock and uses `setsid` to launch the physically selected
immutable runtime bundle. The detached wrapper holds a separate nonblocking
lifetime leader lock; a raw second `obligation-supervise` therefore exits 73
before replay. Lifecycle discovery enumerates only same-UID, PID-1-parented
wrappers whose immutable script and coordination state match this repository.
It also validates the published OCaml child by PID, Linux process-start tick,
parent wrapper, and executable path. If `current` moves during an upgrade or
rollback, the next ensure retires every matching legacy wrapper and starts one
new generation. A singleton wrapper whose OCaml child is still replaying is
given a bounded warm-up period instead of being mistaken for a dead service.
Restart waits for descendant-held lifetime locks to be released.
`obligation-supervisor-stop` terminates the verified wrapper and waits until its
identity is no longer live. These checks prevent a corrupted state file from
redirecting lifecycle control at an unrelated reused PID. The detached service
explicitly closes the bootstrap-lock descriptor before launch; killing an
`ensure` caller mid-start therefore cannot leave a surviving daemon that holds
the bootstrap election. These commands are the inner control-plane API; a
Pod-external guardian such as Beagle should call `ensure` after a Pod restart
rather than manufacturing a tmux session. The production and sabotage receipts
are recorded in
`tools/loom/evidence/loom-autonomous-coordination-v1-20260827.txt`.
The detached wrapper also exports its exact coordination state root so an
independent primary checkout can rediscover the same leader without confusing
its own Git common directory for the supervisor's custody boundary.
When a live leader exists, runtime activation itself now performs this handoff:
the installer returns `ACTIVATED` only after the selected immutable bundle owns
a new verified generation. If that assumption fails, activation restores the
previous bundle and its leader before refusing.

The authoritative state lives under the shared coordination directory in
`loom-obligations/*/journal.tsv`. The TUI, GUI, JSON endpoint, and supervisor are
disposable projections of those journals. Killing all Loom processes therefore
does not close, acknowledge, or lose unfinished work. A new
`obligation-supervise` process reconstructs every unclosed object by replay.
`obligation-reconcile` repairs the bounded crash window between publishing a
request message and opening its obligation.

## Durable execution outcomes

Shared runtime `2026.08.27.39` adds frozen Sounio action `9022` and the
`loom-durable-execution-outcome-v1` capability. Consuming an in-memory execution
grant now opens a kernel-owned outcome obligation. The OCaml broker supervises
the measured leaf, records exit or signal, asks Sounio to admit the complete
receipt, and closes the obligation through an authenticated `EXEC_OUTCOME`
transition. A crash before that commit replays as explicit `INCOMPLETE`; it is
never inferred as success. Receipt, semantics, manifest, runtime, grant,
generation, command, environment, executable, toolchain, hardware, and both
pre-execution Sounio decisions are hash-bound. See
`tools/loom/EXECUTION_OUTCOME_V1.md` and
`tools/loom/EXECUTION_CUSTODY_V2.md` for the proof boundary. Arbitrary Bash/Exec
attachment remains disabled until general shell closure is classifiable.

The Pod-external lane guardian uses the separate `sounio-fleet` authority
boundary. `sounio-fleet watch --apply-recovery` is start-only: it may consume a
bounded recovery budget to restore an enabled missing generation, but it holds
all newly planned stop actions and all starts without a named budget. Manual
`reconcile --apply` remains the only path that may consume a linear stop
capability. This lets migrated lanes and still-running legacy lanes share one
catalog during rollout without giving an unattended recovery loop destructive
authority over the legacy processes.

Use the coordination entrypoint for live lanes so generation identity comes
from verified process presence:

```sh
bin/sounio-coord obligation-consume --agent codex --lane session-ID --message msg-ID
bin/sounio-coord obligation-claim --agent codex --lane session-ID --message msg-ID
bin/sounio-coord obligation-complete --agent codex --lane session-ID \
  --message msg-ID --claim claim-ID --outcome PATH --evidence PATH
```

This is a single-filesystem crash-consistency and generation-fencing claim under
bounded forward clock skew relative to lease TTLs. It is not exactly-once
execution, replicated storage, partition tolerance, or distributed consensus.
See `docs/internal/concepts/loom-obligation.contract` for the exact semantic
boundary and preregistered sabotage control.

`fleet-enroll` stores desired lane intent under the repository's persistent Git
common directory. Catalog v2 makes `custody=agentd|loom` part of desired state.
The default remains `agentd` for compatibility. A `loom` slot records a stable
agent, session UUID, native provider kind, credential home, shared coordination
authority, and SHA-256-bound bootstrap prompt. The raw prompt is copied into
private catalog storage rather than embedded in the descriptor.

Verified persistent catalog kinds are currently `codex` and `kimi`. The Kimi
bootstrap is delivered only after the native TUI is under Loom custody, and the
catalog sabotage gate proves that changing the stored kind to an unverified
provider such as Cursor is refused by the persistent-adapter allowlist.
Because Kimi binds sessions through its native store, concurrent cataloged Kimi
slots must use distinct HOME directories; same-HOME enrollment fails before a
second prompt or descriptor is published.

`fleet-reconcile` is a no-mutation plan by default. It observes both the legacy
fleet adapter and Loom before taking action. A slot whose non-selected authority
is active is refused with `fleet-authority-conflict`; it is never "repaired" by
starting a second CLI. An absent Loom slot is opened with `provider-open`, while
a surviving Guardian with a dead kernel is recovered without replacing the
provider. Repeated application is idempotent, and `fleet-disable` prevents an
intentional stop from being relaunched.

The coordination authority defaults to the `sounio-coord-state` directory of
the worktree from which enrollment runs and can be pinned with `--coord-dir`.
It is injected into every open or recovered kernel as `SOUNIO_COORD_DIR`. This
keeps a provider working in another Git repository on the Sounio fleet bus
instead of silently registering its endpoint in that repository's private Git
state.

An existing active Loom lane can enter the catalog only with `--adopt-active`.
Enrollment verifies agent, lane, worktree, session UUID, and provider command
before atomically publishing desired state. This adopts already-existing Loom
custody; it does not seize or convert a live agentd/tmux PTY. Stop the old
authority first when migrating a legacy lane. The complete state machine and
sabotage boundaries are in `tools/loom/FLEET_CATALOG_V2.md`.

`scripts/dev/install_sounio_loom_kubernetes_hook.sh` installs the one-shot
reconciler in the workspace StatefulSet. It refuses any update strategy other
than `OnDelete`, runs reconciliation as the lane-owning user instead of the
container's root user, and verifies that applying the template did not replace
the current Pod. Use `--dry-run` to inspect the strategic merge patch.

`beagle-serve` is a loopback compatibility backend for the existing Beagle
Workspace Agent. It implements `beagle-pty-supervisor-v1` without changing the
external `beagle-terminal-v1` client protocol. Beagle retains agent routing,
sessions, blocks, redaction, and memory policy; Loom owns PTY custody, process
identity, input authority, replay, and recovery. The bridge exposes verified
local journal heads, recovery counts, and a hash-chained generation-lineage
receipt as additive response fields. If kernel, Guardian, and harness all die,
Loom truthfully creates a new physical generation linked to the verified
predecessor instead of presenting a replacement process as the old one. See
`docs/internal/concepts/loom-multiplexer.contract` for the authority matrix and
canary rules.

Set `SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1`,
`SOUNIO_LOOM_SIGNING_KEY`, and `SOUNIO_LOOM_VERIFY_KEY` to activate the
fail-closed Ed25519 receipt protocol. The private key signs a canonical payload
containing the native Sounio verdict, adapter digest, continuity facts, and
predecessor receipt token. `verify-continuity-receipt` needs only the public key
and adapter: it verifies the signature, canonical encoding, adapter identity,
and a replay of the Sounio policy. A signed successor refuses an unsigned
predecessor, a different-generation receipt signed by the same key, or an
incomplete keypair. With signing unset, Loom retains receipt v1 compatibility;
that mode carries no authenticity claim.

Set `SOUNIO_LOOM_REQUIRE_INDEPENDENT_OBSERVER=1` and
`SOUNIO_LOOM_OBSERVER_VERIFY_KEY` to require a second principal before a lost or
exited generation is replaced. The observer attestation binds the exact signed
predecessor receipt, fact digest, signer identity, and adapter digest. Before
`start_command`, the OCaml preflight verifies both signatures and asks the
native Sounio adapter to produce a private disjoint-principal proof. Principal
identity is the SHA-256 digest of the canonical SubjectPublicKeyInfo DER, not the
PEM file bytes or role label. Reusing the signer key as the observer key is
therefore refused before a successor generation exists even when the same key
has a different PEM serialization.

Set `SOUNIO_LOOM_REQUIRE_INDEPENDENT_MEASUREMENT=1` to strengthen that preflight
from receipt precommitment to source-separated measurement. The observer signs
a fact vector it derives from the retained generation descriptor plus the raw
semantic and Guardian journals: generation ID, generation fingerprint, semantic
journal head, and Guardian journal head. The OCaml host verifies the decision
receipt and measurement attestation separately, recomputes the measured vector
from those raw artifacts, and sends both vectors to Sounio without comparing
them. Only the private Sounio `VerifiedMeasurementAgreement` constructor may
join them, and only when all four tokens agree. Any disagreement refuses before
`start_command`; no successor receipt is created. This mode implies signed
receipts and independent-observer mode and requires the same signer and observer
key configuration.

Set `SOUNIO_LOOM_REQUIRE_OBSERVATION_AUTHORITY=1` to add a third, write-time
authority. This mode implies signed receipts, independent measurement, and
journal authority. Configure `SOUNIO_LOOM_JOURNAL_AUTHORITY_SOCKET`,
`SOUNIO_LOOM_JOURNAL_AUTHORITY_VERIFY_KEY`, and
`SOUNIO_LOOM_JOURNAL_AUTHORITY_EPOCH`; run `journal-authority-serve` with the
matching private key in a separately supervised process. Every semantic and
Guardian event receives an epoch-scoped Ed25519 signature before append. The
authority persists one monotonic `(sequence, head)` state per journal context,
revalidates that state's own signature on read, and fsyncs the containing
directory after each atomic replacement, so retries are idempotent but rewrites
and forks are refused. The authority state directory and private key must be in
the authority supervisor's custody; giving the workload deletion access to both
journal state and its restart boundary is outside this local protocol. Set
`SOUNIO_LOOM_JOURNAL_AUTHORITY_REVOKED_EPOCHS` to a comma-separated denylist
when verifying retired or compromised epochs.

Set `SOUNIO_LOOM_JOURNAL_AUTHORITY_QUORUM=2` to replace that legacy single
authority with a fixed two-of-three certificate. Configure each member with
`SOUNIO_LOOM_JOURNAL_AUTHORITY_{1,2,3}_SOCKET` and
`SOUNIO_LOOM_JOURNAL_AUTHORITY_{1,2,3}_VERIFY_KEY`. The three canonical SPKI-DER
principal identities must be pairwise distinct. Every sixteen-field journal
record retains all three identities and either an Ed25519 signature or `-` for
each member; append and replay require at least two valid signatures over the
same context, epoch, sequence, previous head, and event hash. One unavailable
daemon therefore does not stop progress. An authority that misses an event
cannot rejoin automatically because its monotonic state is then behind; audited
state transfer and member reconfiguration are separate future protocols.

Receipt v3 signs the raw generation, fingerprint, semantic checkpoint, and
Guardian checkpoint together with their domain-separated SHA-256 digests. The
observer verifies that both signed journal heads occur in the fully verified
append-only streams, then measures the same checkpoint while binding current
journal and descriptor digests. Native Sounio receives each SHA-256 value as
eight canonical 32-bit limbs and admits only exact equality plus pairwise
disjoint signer, observer, and journal principals. Older receipt v2,
measurement v1, and pre-spawn `9003`/`9004` modes remain available when the new
mode is not requested; they make no observation-authority claim.

Quorum measurement uses attestation v3 and pre-spawn frame `9006`. The observer
binds the configured principal set, required quorum, and minimum valid signature
count observed across both retained journals. The checkpoint also includes the
SHA-256 digest of each literal journal. Because every 16-field journal record
contains the ordered principal identities and its signature-or-absence slots,
those journal digests commit every per-event quorum certificate and signer
subset, not merely the aggregate minimum. Native Sounio constructs private
`VerifiedJournalAuthorityQuorum` and
`VerifiedObservationAuthorityQuorumAdmission` terms only when the minimum is at
least two and signer, observer, and all three journal principals are pairwise
disjoint. The single-authority `9005` proof remains a distinct legacy type.

## Three-Agent Recovery Canary

`scripts/ci/sounio_loom_three_agent_recovery_canary.sh` runs Codex, Grok, and
MiniMax through their real CLIs under three independent Loom Guardians. It kills
all three disposable kernels, requires each lane to become `recoverable` with no
stale Guardian bridge, then starts new kernels and checks that every Guardian
PID, CLI PID, and Loom instance ID remains unchanged. Each agent must create a
physical receipt through its Bash tool and its unique token must be present in
durable replay; a model merely claiming success cannot satisfy the gate.

The retained 2026-08-25 run passed with three replaced kernel PIDs, three stable
Guardians, three stable CLI processes, three tool receipts, and three replay
tokens. The measured sequential recovery interval was 549 ms. The preregistered
inputs, status transitions, raw snapshots, outcome, and checksums are retained
under `tools/loom/evidence/three-agent-recovery-20260825/`.

## Evidence Boundary

Loom-1.6 tolerates observer, interactive-client, GUI, and kernel loss on one Unix
host. The three-agent canary exercises concurrent real CLI processes on that
same-host boundary; it does not establish Guardian, host, storage, provider-auth,
or network-partition recovery. `recover` reconciles bytes fsynced by the guardian
while no kernel existed and semantically revokes input leases whose sockets died
with the old kernel.
It cannot re-adopt the same PTY after Guardian or host loss. It can detect that
loss and reconcile a Beagle pane into a new generation whose append-only
lineage receipt binds both verified predecessor journal heads. The Kubernetes
startup hook activates that one-shot reconciliation after Pod loss. The local
coordination gate and a source-pinned Kubernetes canary now exercise four
complete generations: an unacknowledged wake reaches generations one through
three, same-generation retries deduplicate, and an ACK recorded after the
independent third-generation depth control suppresses generation four. This is
at-least-once wake delivery, not exactly-once execution of the work named by the
message.
Existing fleet generations therefore retain the Python `agentd` launch adapter
during migration.
The Beagle bridge passed its source gate, an isolated second-process canary,
and a source-derived four-Pod canary with a dedicated retained PVC. The native
Sounio adapter keeps initial generation, clean respawn, and Pod resurrection
promotion states distinct. Signed promotion additionally requires a private
`VerifiedSignedPodResurrection` proof type; independent admission additionally
requires nominally distinct decision and observation proof terms plus a private
`VerifiedDisjointPrincipals` value. Role collapse is rejected with `E009`, and
external disjointness construction is rejected with `E176`. The Ed25519 gate
additionally refuses missing keys,
payload and signature mutation, the wrong public key, and a validly signed
receipt spliced from another generation. These controls establish bounded
receipt integrity and keyholder authorship under the mounted key and adapter,
not semantic truth against colluding authorities, hardware attestation, or
protection against compromise of both keys. The
`sounio_loom_correct_signature_wrong_facts_probe.sh` now precommits an
independently signed digest before a legitimate signer rewrites and re-signs the
facts; the mismatch is refused before successor creation. This proves bounded
post-observation tamper detection, not that the observer measured the semantic
facts independently or would reject an initially false receipt.
Independent-measurement mode goes further: after a legitimate signer rewrites
and re-signs the semantic-head decision fact, the observer derives the original
head from the retained raw journal and signs that measurement. Normal Sounio
refuses the disagreement before spawn; replacing only
`measurement_tokens_agree` with unconditional `true` admits the unchanged
witness. Separate decision and measurement types reject role collapse with
`E009`, and external agreement construction rejects with `E176`.
This establishes source-separated measurement of four retained artifacts and
identifies the Sounio equality rule as load-bearing. Observation-authority mode
closes two bounded weaknesses in that result: fact agreement uses lossless
SHA-256 values rather than only compact aliases, and journal history requires a
third principal at write time. In the retained causal controls, a rehashed
semantic journal plus newly signed decision and observer artifacts is refused
before spawn because its per-event authority signatures are stale; replacing
only `journal_authority_signature_is_valid` admits the same witness. A separate
alias witness is refused by full-digest equality; replacing only
`full_digest_vectors_agree` admits it. The controls establish that both rules
are load-bearing. They do not establish organizational, process, hardware, or
network independence, compromise resistance of the journal key, Byzantine
consensus, or trusted key and monotonic-state custody. Compact 60-bit principal-token collisions
remain fail-closed false-refusal risks at approximately 2^-60; fact equality no
longer relies on those compact tokens. The real-Pod
witness relocated
compute from `t560-proxmox` to `r740-proxmox` over one retained Ceph RBD RWOP
PVC. It is not state replication, simultaneous multi-node execution, or a
partition/consensus witness. Deployed Cockpit, canonical-memory, Warp, Madaros
parity, signer custody/rotation, and exactly-once external effects remain
separate gates; see the 2026-08-24 receipts under `tools/loom/evidence/`.

Journal-quorum mode narrows one part of that custody boundary. A certificate
cannot be produced or replay-verified by only one configured journal key, and
the retained gate continues after one daemon is killed. The unchanged
single-share frame is refused by native Sounio; replacing only
`journal_quorum_is_satisfied` admits it. Reusing one key in two member slots is
refused before certificate interpretation, external quorum construction refuses
with `E176`, and substituting a legacy `VerifiedJournalAuthority` where a quorum
is required refuses with `E009`. This is a cryptographic two-of-three
authorization claim, not threshold-signature cryptography, Byzantine consensus,
semantic review by the signers, or proof that the three keys are held by
different organizations, machines, users, or hardware devices. A workload can
still ask two structurally honest daemons to sign a new but semantically false
event; quorum protects the chain from one-key custody failure, not event truth.
The retained gate independently recomputes the authority checkpoint from the
literal certificate-bearing journals and descriptor, so deleting or changing a
signer slot changes the committed history before native admission.
See `docs/internal/concepts/loom-multiplexer.contract` for the full semantic and
falsification contract.
