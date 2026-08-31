# LOOM Native Hook Cutover

Concept-ID: `SOUNIO-LOOM-NATIVE-HOOK-CUTOVER`

## Authority

Sounio action 9045 is the semantic authority for provider hook ingress.
OCaml normalizes provider payloads, invokes the frozen Sounio executable, and
realizes an admitted decision. Provider CLIs, configuration files, shell
launchers, LLM output, Python, Rust, and receipt consumers cannot manufacture
an ALLOW result.

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

## Provider Dialects

The raw provider dialect remains part of the authority input even after event
normalization. Two providers that normalize to `PreToolUse` are not thereby
the same principal. A dialect mismatch is a semantic refusal, not a parser
fallback.

## Semantic Boundary

Action 9045 does not redefine action 9044 material-change authority, provider
authentication, model behavior, or the meaning of a provider result. It owns
only the transition from a provider-specific hook envelope to a frozen Sounio
admission decision and the evidence required to promote that transition.

An OCaml runtime may operate the cutover because it is a material executor of
the frozen Sounio decision. It is not semantic authority. The target remains a
Sounio-compiled ingress when the compiler can host the complete runtime safely.

## Forbidden Claims

- Installing a native command string is not proof that the Python bridge is
  absent from the shipped runtime.
- A synthetic hook fixture is not a live provider canary.
- A wake receipt is not an acknowledgement or an execution result.
- A provider CLI exit code cannot override a Sounio DENY.
- An LLM review cannot promote a hook or receipt to semantic authority.
