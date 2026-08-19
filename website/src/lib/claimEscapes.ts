/**
 * Counted escapes. Each id is a numeral that is not a MeasurementClaim.
 * The check prints this table on every run. Growing the table is a
 * review signal — do not hide a new well by adding a silent skip.
 */
import { publicContract } from '../data/artifactStatus';
import { claimEscape } from './measurementClaim';
import { getProofData } from './proofData';

function formatLines(n: number | null): string {
  if (n == null) return '—';
  if (n >= 1_000_000) return `${Math.round(n / 100_000) / 10}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return String(n);
}

export function fullSuitePublished(): boolean {
  return publicContract.metrics.fullTestSuite != null;
}

export function readmeBadgeDiffers(): boolean {
  const { readmeBadge, checkedArtifact } = publicContract.versions;
  return Boolean(readmeBadge && readmeBadge !== checkedArtifact);
}

export function escapeStdlibInventoryFiles(): string {
  const n = publicContract.metrics.stdlibInventoryFiles;
  return claimEscape({
    id: 'stdlib-inventory-files',
    class: 'inventory',
    reason: 'File count from the May reliability sync. Not a gate outcome.',
    text: n == null ? '—' : String(n),
  });
}

export function escapeScienceLanes(): string {
  return claimEscape({
    id: 'science-lanes',
    class: 'inventory',
    reason: 'Lane count from the science pipeline JSON. Inventory, not a remasure.',
    text: String(publicContract.metrics.scienceLanes),
  });
}

export function escapeSelfHostedLines(): string {
  return claimEscape({
    id: 'self-hosted-source-lines',
    class: 'inventory',
    reason: 'Compiler LOC from the May sync. Not a gate outcome or remasure.',
    text: formatLines(publicContract.metrics.selfHostedSourceLines),
  });
}

export function escapeSelfHostedFiles(): string {
  const n = publicContract.metrics.selfHostedSourceFiles;
  return claimEscape({
    id: 'self-hosted-source-files',
    class: 'inventory',
    reason: 'Self-hosted file count from the May sync. Inventory, not a green.',
    text: n == null ? 'inventory pending' : String(n),
  });
}

export function escapeSelfHostedLineNote(): string {
  const n = publicContract.metrics.selfHostedSourceLines;
  return claimEscape({
    id: 'self-hosted-source-line-note',
    class: 'inventory',
    reason: 'Companion line-count note for the self-hosted file inventory.',
    text: n == null ? 'line count unavailable' : `${n.toLocaleString('en-US')} lines`,
  });
}

export function escapeFullSuite(unpublished = '—'): string {
  const full = publicContract.metrics.fullTestSuite;
  return claimEscape({
    id: 'readme-full-suite',
    class: 'inventory',
    reason: 'README snapshot field. Null on current main — not a remasure and not a gate green.',
    text: full ? `${full.pass}/${full.total}` : unpublished,
  });
}

export function escapeFullSuiteSentence(): string {
  const full = publicContract.metrics.fullTestSuite;
  return claimEscape({
    id: 'readme-full-suite-sentence',
    class: 'inventory',
    reason: 'README snapshot sentence. Empty while the field is unpublished.',
    text: full ? ` The README full-suite snapshot is ${full.pass}/${full.total} tests pass.` : '',
  });
}

export function escapeFullSuiteInline(): string {
  const full = publicContract.metrics.fullTestSuite;
  return claimEscape({
    id: 'readme-full-suite-inline',
    class: 'inventory',
    reason: 'README snapshot inline on /language. Empty while unpublished.',
    text: full ? ` README full-suite snapshot: ${full.pass}/${full.total}.` : '',
  });
}

export function escapeCheckedArtifact(): string {
  return claimEscape({
    id: 'checked-artifact-version',
    class: 'version',
    reason: 'Compiler launcher version string from the public contract. Identity, not a green-count.',
    text: publicContract.versions.checkedArtifact,
  });
}

export function escapeReadmeBadge(): string {
  return claimEscape({
    id: 'readme-badge-version',
    class: 'version',
    reason: 'README badge string. Identity during release reconciliation, not a green-count.',
    text: publicContract.versions.readmeBadge ?? '',
  });
}

export function escapeReleaseVersion(): string {
  return claimEscape({
    id: 'release-provenance-version',
    class: 'version',
    reason: 'Release provenance version from proofData. Identity, not a gate numeral.',
    text: `v${getProofData().release.version}`,
  });
}

export function escapeActiveEntrypoints(): string {
  return claimEscape({
    id: 'stdlib-active-entrypoints',
    class: 'inventory',
    reason: 'Module entrypoint inventory from proofData. Not a reliability gate numeral.',
    text: String(getProofData().stdlib.activeModuleEntrypoints),
  });
}

export function escapeAttestedTargets(): string {
  return claimEscape({
    id: 'gpu-attested-targets',
    class: 'inventory',
    reason: 'Attested binary target count from proofData. Inventory next to the GPU refusal, not the 13-count.',
    text: String(getProofData().gpu.attestedTargetCount),
  });
}

export function escapeScienceStatus(): string {
  return claimEscape({
    id: 'science-runtime-status',
    class: 'identity',
    reason: 'Science pipeline status word from proofData. A label, not a green-count.',
    text: getProofData().stdlib.scienceStatus,
  });
}

export function escapeVancomycinEpsilon(): string {
  return claimEscape({
    id: 'ashp-vancomycin-epsilon',
    class: 'literature',
    reason: 'ASHP 2020 §8.3 threshold cited as literature. Not a remasured gate numeral.',
    text: 'ε ≥ 0.82',
  });
}

export function escapePbpkDemoCount(): string {
  return claimEscape({
    id: 'pbpk-demo-count',
    class: 'inventory',
    reason: 'Count of interactive dissertation demos on the site. Not a gate outcome.',
    text: '2',
  });
}

export function escapeOctonion168(): string {
  return claimEscape({
    id: 'octonion-168-theorem',
    class: 'literature',
    reason: 'Paper identity (the 168 theorem). Not a site measurement or gate outcome.',
    text: '168',
  });
}
