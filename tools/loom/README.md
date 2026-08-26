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
bin/loom provider-auth-login --provider codex
```

`provider-plan` never emits the raw prompt. It publishes its byte length and
SHA-256 digest and replaces the argv value with a digest-bearing placeholder.
Provider credentials stay under the provider's native authority; Loom invokes
the native status or login operation and never reads or copies token files.
Dangerous auto-approval flags are absent unless `--unsafe-auto` is explicit.

`provider-start` executes the selected CLI without a shell through an internal
OCaml trampoline that closes stdin and removes inherited Codex, Claude, and tmux
harness identity variables. `--isolate-context` maps provider-specific
reductions in memory, rules, and subagent context; it is deliberately not
described as a sandbox. The provider remains the Guardian-owned child, so kernel
recovery does not replace its process identity. Codex, Claude Code, Grok, and
OpenCode are supported. Their stream, authentication, and session-binding
differences remain typed rather than being flattened into a false common
denominator. The complete contract and current boundaries are in
`tools/loom/PROVIDER_ABI_V1.md`.

`provider-open` is the persistent counterpart. Its provider stdin remains
attached to the Guardian-owned PTY and is reachable only through Loom's
exclusive input lease or authenticated wake transport. The initial vertical is
Codex-only and fails closed for persistent resume or context-isolation requests
that the native TUI cannot honor. Killing and recovering the disposable Loom
kernel preserves the Guardian, Codex process, instance identity, conversation,
and durable output cursor. A woken lane can answer with
`bin/sounio-coord reply --agent A --lane L --reply-to MESSAGE_ID --message TEXT`;
the command derives the original sender and thread instead of requiring the
provider to reconstruct routing metadata.

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
state-local bootstrap lock, validates both PID and Linux process-start tick,
refuses duplicate starts, and uses `setsid` to launch the physically selected
immutable runtime bundle. If `current` moves during an upgrade or rollback,
the next ensure replaces the old-bundle supervisor with a new generation.
`obligation-supervisor-stop` terminates the verified OCaml process and waits
until its identity is no longer live. Before sending any signal, both commands
also require the executable to belong to an installed immutable Loom bundle (or
the expected local build), so a corrupted state file cannot redirect lifecycle
control at an unrelated reused PID. The detached service explicitly closes the
bootstrap-lock descriptor before `exec`; killing an `ensure` caller mid-start
therefore cannot leave a surviving daemon that holds the election lock. These
commands are the inner control-plane API; a Pod-external guardian such as Beagle
should call `ensure` after a Pod restart rather than manufacturing a tmux session.

The authoritative state lives under the shared coordination directory in
`loom-obligations/*/journal.tsv`. The TUI, GUI, JSON endpoint, and supervisor are
disposable projections of those journals. Killing all Loom processes therefore
does not close, acknowledge, or lose unfinished work. A new
`obligation-supervise` process reconstructs every unclosed object by replay.
`obligation-reconcile` repairs the bounded crash window between publishing a
request message and opening its obligation.

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
