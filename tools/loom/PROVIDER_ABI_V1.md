# Loom Provider ABI v1

Schema: `loom-provider-abi-v1`

## Purpose

The Loom CLI is the stable user and automation interface. Codex CLI, Claude
Code, Kimi Code CLI, Grok CLI, and OpenCode are provider adapters. Each adapter
retains its own release cadence, credential store, session representation,
event stream, and permission model.

The ABI normalizes capabilities and custody. It does not claim that provider
semantics are identical.

## Commands

### `provider-list`

Reports the built-in provider catalog, executable availability, native version,
normalized authentication state, stream type, session binding, and capability
set. `--json` emits one `loom-provider-abi-v1` document.

### `provider-status`

Probes one provider. Authentication states are intentionally non-Boolean:

- `authenticated`: the native CLI positively reports an authenticated identity.
- `unauthenticated`: the native CLI positively reports no authenticated identity.
- `delegated`: authentication belongs to a nested multiprovider store.
- `unknown`: the native CLI has no suitable offline probe or returned an
  inconclusive result.
- `unavailable`: the executable is absent.

Authentication status is not a billing, model-entitlement, or network-health
claim.

### `provider-plan`

Constructs a provider-native argv without executing it. The default
`--lifecycle turn` is headless. `--lifecycle persistent` plans a long-lived
interactive provider whose stdin authority is held by a Loom input lease. The JSON plan
contains:

- provider and ABI identity;
- native executable path;
- stream and session-binding contracts;
- Loom and provider session identities;
- working directory and selected model;
- prompt length and SHA-256 digest;
- full argv SHA-256 digest;
- redacted argv;
- explicit unsafe-auto state;
- explicit context-isolation state.
- lifecycle and stdin-authority contracts.

The raw prompt is never rendered. Prompt files must be regular files, are capped
at 1 MiB, and cannot contain NUL.

### `provider-start`

Builds the same plan and starts the provider as a Guardian-owned Loom child. An
internal OCaml trampoline closes stdin, removes inherited Codex and tmux harness
identity variables, and calls `execve` directly; no shell parses the provider
argv. Provider credential variables remain under the native CLI's authority.
The existing Loom journal, PTY, replay, crash, recover, and snapshot contracts
then apply unchanged.

### `provider-open`

Starts a persistent interactive provider under the same Guardian custody.
Unlike `provider-start`, the internal OCaml trampoline keeps the PTY connected
and gives input authority only to Loom's exclusive interactive lease and
authenticated wake path. Kernel recovery preserves the Guardian PID, provider
PID, instance identity, output cursor, and provider conversation.

Persistent mode supports Codex and Kimi. Codex receives its initial prompt in
the native TUI argv. Kimi's TUI has no positional bootstrap-prompt contract, so
Loom starts the native process first and sends the initial prompt through the
same authenticated input lease used for later wakes. The plan and receipt expose
this distinction as `prompt_transport=argv|loom-wake`; Kimi's bootstrap prompt
does not appear in the process argv. `loom-wake` is executable-only by
construction: any future adapter that adds even one provider argument under
that transport is refused before the process starts. The bootstrap message ID
is derived from provider kind, Loom session UUID, and prompt digest, preventing
same-prompt sessions from aliasing one durable wake identity.

Persistent Claude, Grok, OpenCode, and Cursor adapters are refused until each
has a tested native input and session-resume contract. `provider-open` also
refuses `--isolate-context`: neither supported TUI currently exposes isolation
equivalent to its headless contract.

### `provider-auth-login`

Prints a delegation receipt and replaces itself with the provider's native login
command. Loom does not ingest the credential, authentication callback, or token
material.

## Provider Matrix

| Provider | Event stream | New-session binding | Auth probe |
| --- | --- | --- | --- |
| Codex | JSONL | stream-observed | `codex login status` |
| Claude Code | stream-json | caller UUID | `claude auth status --json` |
| Kimi Code | stream-json | native store | unknown; no offline status contract |
| Grok | streaming-json | caller UUID | unknown; no offline status contract |
| OpenCode | JSON events | stream-observed | delegated multiprovider store |

`caller` means Loom supplies the UUID when the provider supports it.
`stream-observed` means the provider assigns its session identity and emits it in
the event stream. `native store` means the provider assigns and persists the
identity, but its current headless stream does not expose that identity to Loom.

## Context Isolation

`--isolate-context` maps a bounded reduction in inherited provider context:

| Provider | Context-isolation mapping |
| --- | --- |
| Codex | `--ephemeral --ignore-rules` |
| Claude Code | `--safe-mode` |
| Kimi Code | unavailable; request fails closed |
| Grok | `--no-memory --no-subagents --disable-web-search --max-turns 2` |
| OpenCode | `--pure` |

The internal trampoline always removes Codex session IDs, `CLAUDECODE` and
related Claude harness IDs, and tmux identity variables so a nested provider
cannot mistake the parent harness for its own session. `--isolate-context` is
not a network, filesystem, process, or provider-account sandbox. These mappings
apply to `lifecycle=turn`; persistent mode fails closed when isolation is
requested.

## Permission Boundary

Provider plans are non-escalating by default. `--unsafe-auto` maps explicitly to
the provider's native dangerous or auto-approval flag:

| Provider | Explicit unsafe mapping |
| --- | --- |
| Codex | `--dangerously-bypass-approvals-and-sandbox` |
| Claude Code | `--dangerously-skip-permissions` |
| Kimi Code | `--auto` |
| Grok | `--always-approve` |
| OpenCode | `--auto` |

The plan and start receipt expose the unsafe state and argv digest. Loom does
not reinterpret one provider's permission policy as equivalent to another's.

## Executable Boundary

Provider binaries are resolved from `PATH` without a shell. A deployment may
pin an absolute executable with `SOUNIO_LOOM_PROVIDER_<PROVIDER>`. Relative,
missing, non-regular, or non-executable overrides are refused.

## Evidence

`scripts/ci/sounio_loom_provider_abi_selftest.sh` uses five deterministic fake
CLIs to test catalog normalization, native credential authority, nonzero auth
status parsing, prompt redaction, unsafe opt-in, UUID enforcement, override
validation, context-isolation mappings, inherited-harness removal, native login
delegation, headless stdin closure, persistent input leasing, kernel replacement
with stable Guardian/provider identities, Kimi bootstrap through authenticated
`loom-wake`, provider/session-bound bootstrap identity, and verified terminal
replay. Harness removal covers Codex, Claude, Kimi, Cursor, Grok, tmux, and
inherited Sounio agentd/session identities without removing provider credential
variables.

On 2026-08-25, the retained three-agent canary launched real Codex, Grok, and
MiniMax-via-OpenCode processes exclusively through `bin/loom provider-start`.
All three created physical tool receipts while their disposable Loom kernels
were destroyed and recovered. All Guardian, CLI, and instance identities were
preserved; all three tokens were recovered from verified replay. The sequential
kernel-recovery interval was 548 ms. Claude Code was installed but its native
status reported unauthenticated, so no Claude model request was made. Evidence
and hashes are in `tools/loom/evidence/three-agent-recovery-20260825/`.

## Current Boundary

Provider ABI v1 does not yet:

- project stream-observed or native-store provider session IDs into a durable
  provider catalog;
- normalize provider event payloads into a shared typed event algebra;
- expose persistent interactive adapters beyond Codex and Kimi;
- resume a native-store Kimi session after its provider process dies;
- resume a persistent provider after Guardian or host loss;
- broker or replicate credentials;
- prove provider readiness by making a paid model request during status;
- admit provider launch capabilities through a native Sounio policy proof.

These are subsequent protocol layers, not implied properties of v1.
