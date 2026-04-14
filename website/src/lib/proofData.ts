import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

type AnyRecord = Record<string, unknown>;

interface ArtifactSource {
  label: string;
  path: string;
  status: string;
  generatedAtUtc: string | null;
}

export interface ProofData {
  generatedAtUtc: string | null;
  release: {
    version: string;
    status: string;
    generatedAtUtc: string | null;
    assetUrl: string | null;
    sha256: string | null;
    signatureVerified: boolean;
    statusLabel: string;
  };
  gpu: {
    status: string;
    generatedAtUtc: string | null;
    sourceVersion: string;
    publicPassCount: number;
    publicCheckCount: number;
    privateSurfaceCount: number;
    attestedTargetCount: number;
    nativeLanePassCount: number;
    nativeLaneCount: number;
  };
  stdlib: {
    status: string;
    generatedAtUtc: string | null;
    passCount: number;
    totalCount: number;
    skipCount: number;
    activeModuleEntrypoints: number;
    scienceStatus: string;
    scienceRuntimeRegressionStatus: string;
    hyperPassCount: number;
    hyperTotalCount: number;
  };
  boundaries: string[];
  sources: ArtifactSource[];
}

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(moduleDir, '../../..');

let cached: ProofData | null = null;

function readJson<T>(relativePath: string, fallback: T): T {
  const absolutePath = path.join(repoRoot, relativePath);
  if (!existsSync(absolutePath)) return fallback;

  try {
    return JSON.parse(readFileSync(absolutePath, 'utf8')) as T;
  } catch {
    return fallback;
  }
}

function asRecord(value: unknown): AnyRecord {
  return value && typeof value === 'object' ? (value as AnyRecord) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asNullableString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function latestIso(values: Array<string | null>): string | null {
  const timestamps = values
    .filter((value): value is string => Boolean(value))
    .map((value) => Date.parse(value))
    .filter((value) => Number.isFinite(value));

  if (timestamps.length === 0) return null;
  return new Date(Math.max(...timestamps)).toISOString();
}

function compactBoundary(reason: string): string {
  return reason
    .replace(/^BLOCKED:\s*/i, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function getProofData(): ProofData {
  if (cached) return cached;

  const releaseArtifact = asRecord(readJson<unknown>('artifacts/omega/souc_release_provenance.v1.json', {}));
  const gpuArtifact = asRecord(readJson<unknown>('artifacts/omega/gpu_public_contract.v1.json', {}));
  const stdlibArtifact = asRecord(readJson<unknown>('artifacts/stdlib/stdlib_reliability_status.v1.json', {}));

  const releaseGeneratedAt = asNullableString(releaseArtifact.generated_at_utc);
  const gpuGeneratedAt = asNullableString(gpuArtifact.generated_at_utc);
  const stdlibGeneratedAt = asNullableString(stdlibArtifact.generated_at_utc);

  const releaseSha256 = asRecord(releaseArtifact.sha256);
  const releaseSignature = asRecord(releaseArtifact.signature);

  const gpuChecks = asArray(gpuArtifact.checks).map(asRecord);
  const gpuSupportEvidence = asRecord(gpuArtifact.support_evidence);
  const gpuAttestedTargets = asArray(gpuSupportEvidence.attested_targets);
  const gpuNativeLanes = asArray(gpuSupportEvidence.native_lanes).map(asRecord);
  const stdlibTotals = asRecord(stdlibArtifact.totals);
  const stdlibInventory = asRecord(stdlibArtifact.inventory);
  const stdlibSciencePipeline = asRecord(stdlibArtifact.science_pipeline);
  const stdlibHyperPipeline = asRecord(stdlibArtifact.hyper_pipeline);
  const stdlibHyperTotals = asRecord(stdlibHyperPipeline.totals);

  const publicChecks = gpuChecks.filter((check) => {
    const status = asString(check.status);
    return status === 'pass' || status === 'fail';
  });
  const publicPassCount = publicChecks.filter((check) => asString(check.status) === 'pass').length;
  const privateSurfaceCount = gpuChecks.filter((check) => asString(check.status) === 'not_public').length;
  const nativeLanePassCount = gpuNativeLanes.filter((lane) => asString(lane.status) === 'pass').length;

  const stdlibIgnored = asArray(stdlibArtifact.ignored).map(asRecord);
  const boundaryNotes = [
    ...gpuChecks
      .filter((check) => asString(check.status) === 'not_public')
      .map((check) => compactBoundary(asString(check.note))),
    ...stdlibIgnored.slice(0, 2).map((entry) => compactBoundary(asString(entry.reason))),
  ].filter(Boolean);

  cached = {
    generatedAtUtc: latestIso([releaseGeneratedAt, gpuGeneratedAt, stdlibGeneratedAt]),
    release: {
      version: asString(releaseArtifact.version, 'unpublished'),
      status: asString(releaseArtifact.status, 'unknown'),
      generatedAtUtc: releaseGeneratedAt || null,
      assetUrl: asNullableString(releaseArtifact.asset_url),
      sha256: asNullableString(releaseSha256.actual),
      signatureVerified: asBoolean(releaseSignature.verified),
      statusLabel: asBoolean(releaseSignature.verified)
        ? 'Signature verified'
        : 'Unsigned or unverifiable in current workspace',
    },
    gpu: {
      status: asString(gpuArtifact.status_summary, 'unknown'),
      generatedAtUtc: gpuGeneratedAt || null,
      sourceVersion: asString(asRecord(gpuArtifact.profile).souc_version, 'unknown'),
      publicPassCount,
      publicCheckCount: publicChecks.length,
      privateSurfaceCount,
      attestedTargetCount: gpuAttestedTargets.length,
      nativeLanePassCount,
      nativeLaneCount: gpuNativeLanes.length,
    },
    stdlib: {
      status: asString(stdlibArtifact.status_summary, 'unknown'),
      generatedAtUtc: stdlibGeneratedAt || null,
      passCount: asNumber(stdlibTotals.pass),
      totalCount: asNumber(stdlibTotals.total),
      skipCount: asNumber(stdlibTotals.skip),
      activeModuleEntrypoints: asNumber(stdlibInventory.active_module_entrypoints),
      scienceStatus: asString(stdlibSciencePipeline.status, 'unknown'),
      scienceRuntimeRegressionStatus: asString(stdlibSciencePipeline.runtime_regression_summary_status, 'unknown'),
      hyperPassCount: asNumber(stdlibHyperTotals.pass),
      hyperTotalCount: asNumber(stdlibHyperTotals.total),
    },
    boundaries: boundaryNotes,
    sources: [
      {
        label: 'Release provenance',
        path: 'artifacts/omega/souc_release_provenance.v1.json',
        status: asString(releaseArtifact.status, 'unknown'),
        generatedAtUtc: releaseGeneratedAt || null,
      },
      {
        label: 'GPU public contract',
        path: 'artifacts/omega/gpu_public_contract.v1.json',
        status: asString(gpuArtifact.status_summary, 'unknown'),
        generatedAtUtc: gpuGeneratedAt || null,
      },
      {
        label: 'Stdlib reliability status',
        path: 'artifacts/stdlib/stdlib_reliability_status.v1.json',
        status: asString(stdlibArtifact.status_summary, 'unknown'),
        generatedAtUtc: stdlibGeneratedAt || null,
      },
    ],
  };

  return cached;
}
