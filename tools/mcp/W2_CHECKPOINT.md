# CC-2 W2 Checkpoint

Date: 2026-05-17
Intended branch: `feature/mcp-server`
Observed checkout during top-tier audit: `feature/mcp-server`
Worktree: `/workspace/sounio`

## Artefact Inventory

Regenerate with:

```bash
find tools/mcp .cursor/rules/sounio.mdc agent_logs/CC2_convergence.md tests/fixtures/broken_dose.sio tests/fixtures/broken tests/fixtures/mcp_init.json examples/hello.sio \
  -type f \
  -not -path '*/__pycache__/*' \
  -not -path '*/.pytest_cache/*' \
  -not -path '*/.ruff_cache/*' \
  -not -name 'ARTIFACT_MANIFEST.sha256' \
  -print | sort | xargs sha256sum
```

Key artefacts:

The command above is the canonical inventory command for this checkpoint. The
latest captured inventory is `tools/mcp/ARTIFACT_MANIFEST.sha256`; re-run the
command after the branch is corrected and immediately before commit so the hashes
describe the exact submitted artefacts.

Twenty benchmark fixtures live under `tests/fixtures/broken/broken_01.sio` through
`tests/fixtures/broken/broken_20.sio`; their hashes are covered by the command
above.

## Adversarial Self-Critique

See [ADVERSARIAL_SELF_CRITIQUE.md](ADVERSARIAL_SELF_CRITIQUE.md).

## Regression Check

```bash
bash scripts/run_sio_test_suite.sh
```

Observed:

- Pass: 929
- Fail: 3
- Skip: 40
- Total: 972

Clean baseline comparison:

```bash
git worktree add --detach /tmp/sounio-cc2-baseline feature/mcp-server
bash scripts/run_sio_test_suite.sh
```

Observed on clean baseline at `feature/mcp-server`:

- Pass: 929
- Fail: 3
- Skip: 40
- Total: 972

The matching failures are:

- `refinement_f64_return_violation.sio` expected compile failure but passed.
- `test_pd_gum_voi.sio` timed out after 30 seconds in the full parallel suite.
- `test_pinn_training_d6.sio` timed out after 30 seconds in the full parallel
  suite.

`test_steady_state.sio` timed out in one patched full-suite run but passed in
the clean baseline, passed in the patched tree when filtered, and disappeared on
the patched full-suite rerun. It is classified as parallel timeout noise, not an
MCP regression.

## Quality Gates

| Gate | Status | Evidence |
|---|---|---|
| Q1 | PASS | `ruff check tools/mcp`; `mypy --strict tools/mcp/sounio_mcp` |
| Q2 | PASS | `agent_logs/CC2_convergence.md`, 4 implementation/critique/revision cycles |
| Q3 | PASS | `tools/mcp/ADVERSARIAL_SELF_CRITIQUE.md` |
| Q4 | PASS | Full harness rerun matched clean `feature/mcp-server` baseline: 929 pass / 3 fail / 40 skip |
| Q5 | PASS | New `.sio` fixtures use small Sounio-native programs; `examples/hello.sio` runs |
| Q6 | PASS | Tool/resource docstrings plus `tools/mcp/README.md`, structured content, and usage examples |
| Q7 | PASS | Loop benchmark pins `deterministic-fixture-agent-v0.1` and seed `1729` |
| Q8 | PASS | New docs use EN-UK style where applicable |
| Q9 | PASS | `tools/mcp/AI_DISCLOSURE.md` |
| Q10 | N/A | No numerical scientific claims beyond loop convergence counts |

## Off-Path Flags

- The live `modelcontextprotocol/servers` repo no longer accepts community
  server README PRs. Registry handoff is now `MCP_REGISTRY_PR_DRAFT.md` plus
  `registry_server.draft.json` for `modelcontextprotocol/registry` publication
  after PyPI packaging and operator authentication.
- The FastMCP SDK decorators are used for the server surface, but stdio runtime
  dispatch is local JSON-RPC because the SDK stdio reader hung on piped stdin in
  this headless environment.
- `bin/souc-linux-x86_64` was not modified. The observed SHA-256 in this checkout
  is `8be13f02ac21967ceda23d0df5a4e26f9f00d00568fb830b31f45687ded57cc7`,
  which differs from the SHA stated in the sprint brief; this branch leaves the
  binary untouched.
