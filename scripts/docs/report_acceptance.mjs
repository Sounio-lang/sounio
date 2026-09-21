// On-demand corpus-wide acceptance numbers (total governed topics, per-authority
// and per-owner breakdowns, evidence-bearing topics, validation-surface union).
//
// These numbers are NOT committed to docs/governance/DOCS_ACCEPTANCE_REPORT.md
// and NOT gated by scripts/docs/check_docs_registry.mjs -- they are a pure
// function of every governed doc present in the tree at scan time, so a
// snapshot committed by one PR goes stale the instant any *other* PR adds or
// removes a governed doc, even though neither PR touched the other one's
// files. See formatAcceptanceReportStub in governance_registry.mjs for the
// full account of why this moved out of the committed, gated surface.
//
// Usage: node scripts/docs/report_acceptance.mjs
import path from 'node:path';
import { buildGovernedTopicRegistry, formatAcceptanceReport } from './governance_registry.mjs';

const rootDir = path.resolve(process.cwd());
const registry = await buildGovernedTopicRegistry(rootDir);
process.stdout.write(`${formatAcceptanceReport(registry).trimEnd()}\n`);
