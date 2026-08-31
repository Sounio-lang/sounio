# LOOM Native Hook Cutover

Concept-ID: `SOUNIO-LOOM-NATIVE-HOOK-CUTOVER`

## Authority

Sounio action 9045 is the semantic authority for one provider hook event.
Sounio action 9046 is the semantic authority for draining old hook generations
and promoting a bridge-free fleet. OCaml normalizes provider payloads, invokes
the frozen Sounio executables, and realizes admitted decisions. Provider CLIs,
configuration files, shell launchers, LLM output, Python, Rust, receipt
consumers, and the UI cannot manufacture an ALLOW or CUTOVER_READY result.

## Contract

The cutover admits a hook event only when all of the following are affirmative:

- action 9044 is frozen and the exact action-9045 source, freeze, OCaml runtime,
  and provider configuration hashes are bound;
- the provider and raw dialect are one exact pair: Codex/snake case,
  Claude/snake case, Cursor/Cursor camel case, or Grok/Grok camel case;
- the raw event, payload, session, working directory, agent, and lane are bound
  before normalization;
- a pre-tool event binds the tool name and complete tool input before execution;
- the provider configuration directly executes the native OCaml runtime and
  contains no Python, Rust, or disposable-language oracle bridge;
- missing policy, malformed input, timeout, hash drift, or runtime failure
  refuses before the provider operation;
- an append-only decision receipt records both ALLOW and DENY outcomes; and
- CLAIM_READY requires atomic configuration promotion, rollback evidence, and
  live canaries from Codex, Claude, Cursor, and Grok.

The fleet cutover reaches `CUTOVER_READY` only when all of the following are
affirmative in the same freshness-bounded inventory epoch:

- the legacy and candidate runtime hashes plus the drain and final provider
  configuration hashes are bound;
- every live process is bound to its exact process generation and loaded hook
  capability;
- the inventory is fresh and complete, every member is classified, and the
  classified count equals the total count;
- the native count equals the non-zero total count, while legacy, unknown, and
  unresponsive counts are all zero;
- four-provider canaries and the paired atomic rollback have passed; and
- the action-9046 decision receipt is complete and no prohibited oracle ran.

This is an affirmative absence proof. Merely failing to observe a Python hook,
or deleting its file while a process still has the command cached, is not
evidence that the legacy generation is absent.

## Provider Dialects

The raw provider dialect remains part of the authority input even after event
normalization. Two providers that normalize to `PreToolUse` are not thereby
the same principal. A dialect mismatch is a semantic refusal, not a parser
fallback.

## Semantic Boundary

Actions 9045 and 9046 do not redefine action 9044 material-change authority,
provider authentication, model behavior, or the meaning of a provider result.
Action 9045 owns the transition from a provider-specific hook envelope to a
frozen Sounio admission decision. Action 9046 owns only the transition from a
mixed fleet generation to a bridge-free current generation and the evidence
required to promote it.

An OCaml runtime may operate the cutover because it is a material executor of
the frozen Sounio decision. It is not semantic authority. The target remains a
Sounio-compiled ingress when the compiler can host the complete runtime safely.

## Forbidden Claims

- Installing a native command string is not proof that the Python bridge is
  absent from the shipped runtime.
- A clean filesystem scan is not proof that no live process retains a cached
  legacy hook command.
- An action-9045 hook admission is not an action-9046 fleet cutover decision.
- A synthetic hook fixture is not a live provider canary.
- A wake receipt is not an acknowledgement or an execution result.
- A provider CLI exit code cannot override a Sounio DENY.
- An LLM review cannot promote a hook or receipt to semantic authority.
