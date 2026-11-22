# AGENTS.md

<!-- context7 -->
Use the `ctx7` CLI to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service -- even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use even when you think you know the answer -- your training data may not reflect recent changes. Prefer this over web search for library docs.

Do not use for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

## Steps

1. Resolve library: `npx ctx7@latest library <name> "<user's question>"`
2. Pick the best match (ID format: `/org/project`) by: exact name match, description relevance, code snippet count, source reputation (High/Medium preferred), and benchmark score (higher is better). If results don't look right, try alternate names or queries (e.g., "next.js" not "nextjs", or rephrase the question)
3. Fetch docs: `npx ctx7@latest docs <libraryId> "<user's question>"`
4. Answer using the fetched documentation

You MUST call `library` first to get a valid ID unless the user provides one directly in `/org/project` format. Use the user's full question as the query -- specific and detailed queries return better results than vague single words. Do not run more than 3 commands per question. Do not include sensitive information (API keys, passwords, credentials) in queries.

For version-specific docs, use `/org/project/version` from the `library` output (e.g., `/vercel/next.js/v14.3.0`).

If a command fails with a quota error, inform the user and suggest `npx ctx7@latest login` or setting `CONTEXT7_API_KEY` env var for higher limits. Do not silently fall back to training data.
<!-- context7 -->

## Purpose

Guidance for AI coding agents (Codex and others) that write Sounio code or work in this repository. The language rules, build commands, tooling and known limitations are in [`CLAUDE.md`](CLAUDE.md); this file does not repeat them. Project intent: [`FOUNDER_INTENT.md`](FOUNDER_INTENT.md).

Repository facts and executable scripts override stale documentation when they disagree. When in doubt, prefer:

1. actual repo files and executable scripts
2. committed docs
3. assumptions

---

## Defects: fix, don't file

A defect you find while working is fixed in the same change, with a test that fails before the fix and passes after it. Open an issue only when one of these holds, and say which:

1. the fix needs a maintainer decision (language semantics, public API, or a choice between legitimate alternatives);
2. the fix clearly exceeds the scope of the task;
3. you cannot write to the repository that holds the defect.

The issue must localise the cause (file and function) and include the discriminating test: a minimal program with expected and actual output. Symptom-only issues, and issues for defects you could have fixed, are not filed. Search the open issues first and extend an existing one rather than opening a duplicate. Same rule as `CLAUDE.md` §6.

---

## Writing Sounio

Sounio is not Rust. Read `CLAUDE.md` §7 before writing any `.sio` file, and [`docs/guide/LLM_PROGRAMMING_GUIDE.md`](docs/guide/LLM_PROGRAMMING_GUIDE.md) for the full reference. After every edit, run `./bin/souc check <file>`: compilation is the test of existence, and `souc run` on a library file is a category error.

---

## Compiler resolution

Do not hardcode ad hoc compiler routing if a repo resolver already exists.

### Canonical resolution path
Use:
- `scripts/lib/resolve_souc.sh` for the public compiler entrypoint (`bin/souc`).
- `scripts/lib/resolve_madaros.sh` for the Stage1 modular compiler (`bin/madaros`).

as the canonical compiler-resolution logic unless the task explicitly targets another resolver for cleanup or compatibility reasons.

`bin/souc` is currently a compatibility wrapper whose default engine is Madaros.
The legacy lean_single engine remains the bootstrap seed and can be forced with
`SOUNIO_SOUC_ENGINE=lean_single`. For the engine status and the stale-binary caveat,
read `docs/MADAROS_STATUS.md`; stale raw `artifacts/self-hosted/madaros`
binaries are not evidence against current `origin/main`.

---

## Tooling

The repository ships a Sounio LSP ([`tools/lsp/README.md`](tools/lsp/README.md)) and an MCP server that exposes `check`, `compile`, `run`, `test`, stdlib docs and compiler-error resources ([`tools/mcp/README.md`](tools/mcp/README.md)). Use `sounio_check` as the first step of the repair loop for `.sio` edits. Details in `CLAUDE.md` §5.

---

## Known limitations

See `CLAUDE.md` §13 and [`docs/compiler/KNOWN_LIMITATIONS.md`](docs/compiler/KNOWN_LIMITATIONS.md).
