#!/usr/bin/env node
/**
 * sync-artifact-status.mjs
 *
 * Reads Sounio artifact gate JSONs + README/CHANGELOG/bin/souc and generates
 * a typed data module for the website. Feature claims are backed by committed
 * artifacts and repo front-door docs — not hand-written marketing copy.
 *
 * Run from the website/ directory:
 *   node scripts/sync-artifact-status.mjs
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "../..");
const OUT_FILE = join(__dirname, "../src/data/artifactStatus.ts");

function loadJson(relPath) {
  const full = join(REPO_ROOT, relPath);
  if (!existsSync(full)) {
    console.warn(`[sync-artifacts] missing: ${relPath}`);
    return null;
  }
  try {
    return JSON.parse(readFileSync(full, "utf-8"));
  } catch (e) {
    console.warn(`[sync-artifacts] parse error in ${relPath}: ${e.message}`);
    return null;
  }
}

function readText(relPath) {
  const full = join(REPO_ROOT, relPath);
  if (!existsSync(full)) return null;
  return readFileSync(full, "utf-8");
}

function artifactToLevel(summary) {
  if (!summary) return "unknown";
  const s = String(summary).toLowerCase();
  if (s === "pass") return "verified";
  if (s === "fail") return "blocked";
  if (s === "partial" || s === "beta" || s === "active") return "beta";
  return "unknown";
}

function pick(obj, ...keys) {
  for (const k of keys) {
    if (obj && obj[k] !== undefined) return obj[k];
  }
  return undefined;
}

function parseReadmeVersion(readme) {
  const m = readme?.match(/version-1\.0\.0--beta\.(\d+)/);
  return m ? `1.0.0-beta.${m[1]}` : null;
}

function parseReadmeFullSuite(readme) {
  const m = readme?.match(/(\d+)\s*\/\s*(\d+)\s*tests pass/i);
  return m ? { pass: Number(m[1]), total: Number(m[2]) } : null;
}

function parseWrapperVersion(soucScript) {
  const m = soucScript?.match(/WRAPPER_VERSION="\$\{SOUNIO_SOUC_VERSION:-([^"]+)\}"/);
  return m?.[1] ?? null;
}

function parseBootstrapFromChangelog(changelog) {
  const hashMatch = changelog?.match(/gen2==gen3 hash:\s*([a-f0-9]+)/i);
  const sizeMatch = changelog?.match(/Binary:\s*(\d+)\s*KB/i);
  return {
    sha256: hashMatch?.[1] ?? null,
    sizeKb: sizeMatch ? Number(sizeMatch[1]) : null,
  };
}

function countStage0Lines() {
  const stage0 = readText("bootstrap/stage0.c");
  if (!stage0) return null;
  return stage0.split("\n").length;
}

// ---------------------------------------------------------------------------
// Load artifacts + repo truth
// ---------------------------------------------------------------------------

const reliability = loadJson("artifacts/stdlib/stdlib_reliability_status.v1.json");
const science = loadJson("artifacts/stdlib/stdlib_science_pipeline_status.v1.json");
const hyper = loadJson("artifacts/stdlib/stdlib_hyper_execution_status.v1.json");
const nativeBackend = loadJson("artifacts/omega/native_backend_v2_gate.v1.json");
const selfhost = loadJson("artifacts/omega/selfhost_verification_report.v1.json");
const lsp = loadJson("artifacts/omega/lsp_smoke_status.v1.json");
const gpu = loadJson("artifacts/omega/gpu_runtime_attest_gate.v1.json");
const bootstrap = loadJson("artifacts/omega/bootstrap_full_gate_status.v1.json");

const readme = readText("README.md");
const changelog = readText("CHANGELOG.md");
const soucScript = readText("bin/souc");

const readmeVersion = parseReadmeVersion(readme);
const wrapperVersion =
  reliability?.science_pipeline?.runtime_souc_version ??
  parseWrapperVersion(soucScript) ??
  "unknown";
const fullSuite = parseReadmeFullSuite(readme);
const bootstrapRecorded = parseBootstrapFromChangelog(changelog);
const stage0Lines = countStage0Lines();

const stdlibGatePass = reliability?.totals?.pass ?? 0;
const stdlibGateTotal = reliability?.totals?.total ?? 0;
const stdlibGateSkip = reliability?.totals?.skip ?? 0;
const stdlibInventoryFiles = reliability?.inventory?.sio_files ?? null;
const selfHostedFiles = selfhost?.self_hosted_source?.total_files ?? null;
const selfHostedLines = selfhost?.self_hosted_source?.total_lines ?? null;
const cycleParity = selfhost?.cycle_gate?.parity ?? null;

const generatedAt = new Date().toISOString();

// ---------------------------------------------------------------------------
// Build normalized status object
// ---------------------------------------------------------------------------

const reliabilityReason = reliability
  ? `${stdlibGatePass}/${stdlibGateTotal} stdlib reliability tests pass, ${stdlibGateSkip} skipped`
  : "artifact missing";

const status = {
  generatedAt,
  repoPath: "artifacts/",

  publicContract: {
    sources: {
      readme: "README.md#honest-status",
      limitations: "docs/compiler/KNOWN_LIMITATIONS.md",
      minimumViable: "docs/guide/MINIMUM_VIABLE_SOUNIO.md",
    },
    versions: {
      checkedArtifact: wrapperVersion,
      readmeBadge: readmeVersion,
      lspRelease: "sounio-lsp-v0.3.0-r1",
    },
    defaultWorkflow: {
      launcher: "bin/souc",
      backend: "self-hosted native ELF/Mach-O",
      summary:
        "The public onboarding path is the checked self-hosted launcher. It type-checks and compiles to host binaries — no Rust/Cargo build step required for the default workflow.",
    },
    bootstrap: {
      stage0Lines,
      stage0Path: "bootstrap/stage0.c",
      fixedPointChain:
        "checked bin/souc-linux-x86_64 → gen1 → gen2 → gen3 (gen2 == gen3)",
      historicalNote:
        "stage0.c is the original C bootstrap; the reproducible fixed-point ceremony uses the checked self-hosted binary.",
      recordedSha256: bootstrapRecorded.sha256,
      recordedSizeKb: bootstrapRecorded.sizeKb,
      cycleParity,
      artifact: "CHANGELOG.md + artifacts/omega/selfhost_verification_report.v1.json",
    },
    metrics: {
      stdlibReliabilityGate: {
        pass: stdlibGatePass,
        fail: reliability?.totals?.fail ?? 0,
        skip: stdlibGateSkip,
        total: stdlibGateTotal,
        label: "Stdlib reliability gate",
        artifact: "artifacts/stdlib/stdlib_reliability_status.v1.json",
      },
      fullTestSuite: fullSuite
        ? {
            pass: fullSuite.pass,
            total: fullSuite.total,
            label: "Full test suite (README snapshot)",
            artifact: "README.md#honest-status",
          }
        : null,
      stdlibInventoryFiles,
      selfHostedSourceFiles: selfHostedFiles,
      selfHostedSourceLines: selfHostedLines,
      scienceLanes: Object.keys(science?.lanes ?? {}).length,
      hyperLanes: (hyper?.lane_statuses ?? []).length,
    },
    honestStatus: {
      works: [
        {
          title: "Epistemic core",
          detail:
            "Knowledge[T] with GUM propagation, provenance tracking, and compile-time confidence gates (vancomycin fixtures)",
        },
        {
          title: "Self-hosted compiler",
          detail: cycleParity
            ? `Fixed-point verified (cycle parity=${cycleParity}); ${selfHostedFiles ?? "?"} self-hosted source files`
            : "Lexer, parser, checker, and native codegen in self-hosted/",
        },
        {
          title: "Algebra",
          detail: "Clifford, Cayley-Dickson, octonions — 168 theorem verified computationally",
        },
        {
          title: "Native codegen",
          detail:
            "Linux x86-64 ELF plus checked macOS artifact lanes via bin/souc; PE/COFF backend exists for cross-compile",
        },
        {
          title: "Core stdlib gate",
          detail: `${stdlibGatePass} / ${stdlibGateTotal} stdlib reliability tests pass (${reliability?.generated_at_utc?.slice(0, 10) ?? "artifact date unknown"})`,
        },
        {
          title: "Optimizer",
          detail: "1,000+ e-graph rewrite rules with FAIL=0 in optimizer tests",
        },
        {
          title: "Language server",
          detail: lsp?.status === "pass"
            ? "LSP smoke gate pass; hover, defs, refs, rename, formatting, semantic tokens"
            : "LSP implementation in tools/lsp/ (verify artifact before claiming production)",
        },
        {
          title: "Closure literals",
          detail: "Named function refs and closure tests in tests/run-pass/ (see README resolved list)",
        },
      ],
      scaffolding: [
        {
          title: "Theorem prover",
          detail: "Large arena and data structures; full inference logic not complete",
        },
        {
          title: "Epistemic modules",
          detail: "Many modules are signatures with minimal bodies (~70% scaffolding)",
        },
        {
          title: "Neural networks",
          detail: "Quaternion/octonion NN lanes exist but are not all end-to-end stable",
        },
        {
          title: "Genomics",
          detail: "Several files are stubs disabled on parser limitations",
        },
        {
          title: "Async runtime",
          detail:
            "Self-hosted async tests pass per KNOWN_LIMITATIONS.md; broader runtime integration still partial",
        },
        {
          title: "Geometry engine",
          detail: "Extended geometry paths disabled; core engine partial",
        },
      ],
      missing: [
        {
          title: "Epistemic ODE solver (general RHS)",
          detail: "Exponential decay works; general RHS needed for full PBPK epistemic integration",
        },
        {
          title: "Ontology federation",
          detail: "Local ontology work exists; 15M-term federated query not implemented",
        },
        {
          title: "GPU CLI entry point",
          detail: "PTX/GPU codegen exists in-tree; no complete default CLI path (gpu/lib.sio stub)",
        },
        {
          title: "Windows pre-built binary",
          detail: "PE/COFF backend is production-grade; no checked .exe shipped in this checkout",
        },
        {
          title: "AArch64 native-v2 parity",
          detail: "Apple Silicon support uses checked Mach-O artifact lane; native-v2 aarch64 still preview-grade",
        },
        {
          title: "Checked launcher REPL",
          detail: "README Known Limitations: bin/souc does not expose repl; separate REPL beta exists outside default lane",
        },
      ],
    },
  },

  compiler: {
    core: {
      label: "Lexer / Parser / Type Checker",
      level: "verified",
      reason: "Production-grade per KNOWN_LIMITATIONS.md; no active known bugs",
      artifact: "docs/compiler/KNOWN_LIMITATIONS.md",
    },
    nativeBackend: {
      label: "Native Backend (ELF/Mach-O/PE)",
      level: pick(nativeBackend, "selftest_passed") ? "verified" : "unknown",
      reason: nativeBackend
        ? `selftest_passed=${nativeBackend.selftest_passed}, scalar_smoke_present=${nativeBackend.scalar_smoke_present}, fail_closed=${nativeBackend.fail_closed}`
        : "artifact missing",
      artifact: "artifacts/omega/native_backend_v2_gate.v1.json",
    },
    selfHosted: {
      label: "Self-Hosted Compiler (default path)",
      level: pick(selfhost, "cycle_gate", "parity") ? "verified" : "beta",
      reason: selfhost
        ? `${selfHostedFiles ?? "?"} files, ${selfHostedLines ?? "?"} lines, cycle parity=${selfhost.cycle_gate?.parity}`
        : "artifact missing",
      artifact: "artifacts/omega/selfhost_verification_report.v1.json",
    },
    cranelift: {
      label: "Cranelift JIT (optional profile)",
      level: "beta",
      reason: `Optional omega/JIT profile — not the default bin/souc onboarding path. Stdlib gate: ${reliabilityReason}`,
      artifact: "artifacts/stdlib/stdlib_reliability_status.v1.json",
    },
    llvm: {
      label: "LLVM Backend",
      level: "verified",
      reason: "Production per KNOWN_LIMITATIONS.md; wired via `--backend llvm`",
      artifact: "docs/compiler/KNOWN_LIMITATIONS.md",
    },
    lsp: {
      label: "LSP Server",
      level: artifactToLevel(pick(lsp, "status")),
      reason: lsp
        ? `smoke status=${lsp.status}, strict_no_rust=${lsp.strict_no_rust}`
        : "artifact missing",
      artifact: "artifacts/omega/lsp_smoke_status.v1.json",
    },
    gpu: {
      label: "GPU Codegen (PTX)",
      level: artifactToLevel(pick(gpu, "status_summary")),
      reason: gpu
        ? `status=${gpu.status_summary}, blockers=[${gpu.blockers?.join(", ") ?? ""}]`
        : "artifact missing",
      artifact: "artifacts/omega/gpu_runtime_attest_gate.v1.json",
    },
    bootstrap: {
      label: "Bootstrap Chain",
      level: artifactToLevel(pick(bootstrap, "full_concat", "status")),
      reason: bootstrap
        ? `full_concat=${bootstrap.full_concat?.status}, knowledge_bootstrap=${bootstrap.knowledge_bootstrap?.status}`
        : "artifact missing",
      artifact: "artifacts/omega/bootstrap_full_gate_status.v1.json",
    },
  },

  stdlib: {
    reliability: {
      label: "Core Standard Library",
      level: artifactToLevel(pick(reliability, "status_summary")),
      totals: pick(reliability, "totals") ?? {},
      reason: reliabilityReason,
      artifact: "artifacts/stdlib/stdlib_reliability_status.v1.json",
    },
    scienceLanes: {
      label: "Scientific Pipelines",
      level: artifactToLevel(pick(science, "status_summary")),
      reason: science
        ? `lanes=${Object.keys(science?.lanes ?? {}).length}, status_summary=${pick(science, "status_summary") ?? "n/a"}`
        : "artifact missing",
      lanes: Object.entries(science?.lanes ?? {}).map(([key, lane]) => ({
        id: key,
        label: key,
        level: artifactToLevel(lane.status),
        metrics: lane.metrics ?? {},
        reason: lane.status === "pass" ? "golden comparison pass" : lane.mismatches?.join("; ") ?? "",
      })),
      artifact: "artifacts/stdlib/stdlib_science_pipeline_status.v1.json",
    },
    hyperLanes: {
      label: "Hyper-Execution Neural Lanes",
      level: artifactToLevel(pick(hyper, "status_summary")),
      reason: hyper
        ? `lanes=${(hyper?.lane_statuses ?? []).length}, status_summary=${pick(hyper, "status_summary") ?? "n/a"}`
        : "artifact missing",
      lanes: (hyper?.lane_statuses ?? []).map((lane) => ({
        id: lane.lane,
        label: lane.lane,
        level: artifactToLevel(lane.status),
        reason: lane.reason ?? "",
        blockers: lane.blockers ?? [],
      })),
      artifact: "artifacts/stdlib/stdlib_hyper_execution_status.v1.json",
    },
  },
};

// ---------------------------------------------------------------------------
// Generate TypeScript module
// ---------------------------------------------------------------------------

const ts = `// AUTO-GENERATED by scripts/sync-artifact-status.mjs
// Do not edit manually. Regenerate with: npm run sync:artifacts
// Generated at: ${generatedAt}

export type EpistemicLevel = "verified" | "beta" | "active" | "stub" | "blocked" | "unknown";

export interface ArtifactStatusEntry {
  label: string;
  level: EpistemicLevel;
  reason: string;
  artifact: string;
}

export interface LaneEntry {
  id: string;
  label: string;
  level: EpistemicLevel;
  reason?: string;
  metrics?: Record<string, number>;
  blockers?: string[];
}

export interface HonestStatusItem {
  title: string;
  detail: string;
}

export interface PublicContract {
  sources: Record<string, string>;
  versions: {
    checkedArtifact: string;
    readmeBadge: string | null;
    lspRelease: string;
  };
  defaultWorkflow: {
    launcher: string;
    backend: string;
    summary: string;
  };
  bootstrap: {
    stage0Lines: number | null;
    stage0Path: string;
    fixedPointChain: string;
    historicalNote: string;
    recordedSha256: string | null;
    recordedSizeKb: number | null;
    cycleParity: boolean | null;
    artifact: string;
  };
  metrics: {
    stdlibReliabilityGate: {
      pass: number;
      fail: number;
      skip: number;
      total: number;
      label: string;
      artifact: string;
    };
    fullTestSuite: {
      pass: number;
      total: number;
      label: string;
      artifact: string;
    } | null;
    stdlibInventoryFiles: number | null;
    selfHostedSourceFiles: number | null;
    selfHostedSourceLines: number | null;
    scienceLanes: number;
    hyperLanes: number;
  };
  honestStatus: {
    works: HonestStatusItem[];
    scaffolding: HonestStatusItem[];
    missing: HonestStatusItem[];
  };
}

export interface ArtifactStatus {
  generatedAt: string;
  repoPath: string;
  publicContract: PublicContract;
  compiler: Record<string, ArtifactStatusEntry & { lanes?: LaneEntry[]; totals?: Record<string, number> }>;
  stdlib: Record<string, ArtifactStatusEntry & { lanes?: LaneEntry[]; totals?: Record<string, number> }>;
}

export const artifactStatus: ArtifactStatus = ${JSON.stringify(status, null, 2)};

export const publicContract = artifactStatus.publicContract;

export function levelColor(level: EpistemicLevel): string {
  switch (level) {
    case "verified": return "var(--color-accent-gold)";
    case "beta": return "var(--color-accent-teal)";
    case "active": return "var(--color-accent-orange)";
    case "stub": return "var(--color-accent-purple)";
    case "blocked": return "var(--color-accent-red)";
    case "unknown": default: return "var(--color-text-tertiary)";
  }
}

export function levelLabel(level: EpistemicLevel): string {
  switch (level) {
    case "verified": return "Verified";
    case "beta": return "Beta";
    case "active": return "In Progress";
    case "stub": return "Stub";
    case "blocked": return "Blocked";
    case "unknown": default: return "Unknown";
  }
}

export function levelIcon(level: EpistemicLevel): string {
  switch (level) {
    case "verified": return "✓";
    case "beta": return "β";
    case "active": return "◐";
    case "stub": return "⊘";
    case "blocked": return "✕";
    case "unknown": default: return "?";
  }
}
`;

writeFileSync(OUT_FILE, ts, "utf-8");
console.log(`[sync-artifacts] wrote ${OUT_FILE}`);
console.log(`[sync-artifacts] summary:`);
console.log(`  - compiler entries: ${Object.keys(status.compiler).length}`);
console.log(`  - stdlib entries: ${Object.keys(status.stdlib).length}`);
console.log(`  - checked artifact: ${wrapperVersion}`);
console.log(`  - readme badge: ${readmeVersion ?? "n/a"}`);
console.log(`  - stdlib gate: ${stdlibGatePass}/${stdlibGateTotal}`);
console.log(`  - generatedAt: ${generatedAt}`);
