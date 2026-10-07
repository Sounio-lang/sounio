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
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from "node:fs";
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

// Language release: the README version badge (shields.io escapes "-" as "--").
function parseReadmeVersion(readme) {
  const m = readme?.match(/badge\/version-((?:[^-]|--)+)-/);
  return m ? m[1].replace(/--/g, "-") : null;
}

// Compiler build: the string `bin/souc --version` prints, read from the source
// that produces it (self-hosted/compiler/main.sio), e.g. "Madaros v0.80.0".
function parseCompilerBuild(mainSio) {
  const m = mainSio?.match(/println\("(Madaros v[0-9.]+) -- the Sounio self-hosted compiler"\)/);
  return m?.[1] ?? null;
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
  return stage0.split("\n").length - (stage0.endsWith("\n") ? 1 : 0); // matches `wc -l`
}

// ---------------------------------------------------------------------------
// Load artifacts + repo truth
// ---------------------------------------------------------------------------

const reliability = loadJson("artifacts/stdlib/stdlib_reliability_status.v1.json");
// The published stdlib pass figure comes from the end-to-end result written by
// `bash scripts/stdlib/run_stdlib_e2e.sh`. The reliability-status artifact
// (2026-05-12) predates module-privacy enforcement and is only used for the
// file inventory below, never for a pass count.
const STDLIB_E2E_ARTIFACT = "artifacts/stdlib/stdlib_e2e_result.v1.json";
const stdlibE2e = loadJson(STDLIB_E2E_ARTIFACT);
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
// Do not fall back to the launcher string recorded inside old stdlib artifacts
// (it carried the retired 1.0.0-beta scheme); the compiler build is what
// `./bin/souc --version` prints.
const wrapperVersion =
  parseCompilerBuild(readText("self-hosted/compiler/main.sio")) ??
  parseWrapperVersion(soucScript) ??
  "unknown";
const fullSuite = parseReadmeFullSuite(readme);
const bootstrapRecorded = parseBootstrapFromChangelog(changelog);
const stage0Lines = countStage0Lines();

const stdlibGatePass = stdlibE2e?.totals?.pass ?? 0;
const stdlibGateFail = stdlibE2e?.totals?.fail ?? 0;
const stdlibGateTotal = stdlibE2e?.totals?.total ?? 0;
const stdlibGateSkip = stdlibE2e?.totals?.skip ?? 0;
const stdlibE2eDate = stdlibE2e?.generated_at_utc?.slice(0, 10) ?? "undated";
const stdlibE2eCommand = stdlibE2e?.command ?? "bash scripts/stdlib/run_stdlib_e2e.sh";
const stdlibInventoryFiles = reliability?.inventory?.sio_files ?? null;
// Live inventory of self-hosted/*.sio at sync time (the omega report is from 2026-02-28).
function countSio(dir) {
  let files = 0, lines = 0;
  if (!existsSync(dir)) return null;
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      const sub = countSio(full);
      if (sub) { files += sub.files; lines += sub.lines; }
    } else if (name.endsWith(".sio")) {
      files += 1;
      const text = readFileSync(full, "utf8");
      lines += text.split("\n").length - 1; // newline count, as `wc -l`
    }
  }
  return { files, lines };
}
const selfHostedInventory = countSio(join(REPO_ROOT, "self-hosted"));
const selfHostedFiles = selfHostedInventory?.files ?? selfhost?.self_hosted_source?.total_files ?? null;
const selfHostedLines = selfHostedInventory?.lines ?? selfhost?.self_hosted_source?.total_lines ?? null;
const cycleParity = selfhost?.cycle_gate?.parity ?? null;

const generatedAt = new Date().toISOString();

// ---------------------------------------------------------------------------
// Build normalized status object
// ---------------------------------------------------------------------------

// Stated exactly as measured: pass, fail and skip out of the total, with the
// measurement date and the command that produced it.
const reliabilityReason = stdlibE2e
  ? `Stdlib end-to-end: ${stdlibGatePass} of ${stdlibGateTotal} test programs pass (${stdlibGateFail} fail, ${stdlibGateSkip} skipped), measured ${stdlibE2eDate} with \`${stdlibE2eCommand}\`; result in \`${STDLIB_E2E_ARTIFACT}\``
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
      backend: "self-hosted native x86-64 ELF (Linux only)",
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
        fail: stdlibGateFail,
        skip: stdlibGateSkip,
        total: stdlibGateTotal,
        label: "Stdlib end-to-end",
        artifact: STDLIB_E2E_ARTIFACT,
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
            "Knowledge[T] with GUM propagation and compile-time confidence bounds (vancomycin ε ≥ 0.82 refused on Madaros); first-order variance does not yet cross user calls (KL-11)",
        },
        {
          title: "Self-hosted compiler",
          detail:
            "lean_single seed reaches a CI-checked fixed point (gen2 = gen3); Madaros, the default engine, does not reach its own yet",
        },
        {
          title: "Algebra",
          detail: "Clifford, Cayley-Dickson, octonions — 168 theorem verified computationally",
        },
        {
          title: "Native codegen",
          detail:
            "Linux x86-64 static ELF only (TOUR.md section 6); no macOS, Windows or AArch64 target ships",
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
          title: "Optimizer",
          detail: "e-graph rewriter in self-hosted/ir/egraph.sio is unit-tested but not wired into the default pipeline",
        },
        {
          // 196 of 533 passing is not "works": listed with the partial lanes.
          title: "Stdlib end-to-end tests",
          detail: reliabilityReason,
        },
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
          title: "General GPU backend",
          detail: "`souc build --backend gpu` emits PTX for a skeleton of named kernel patterns (empty bodies), not a general GPU backend (TOUR.md section 6)",
        },
        {
          title: "Windows, macOS and AArch64",
          detail: "Not targeted: the compiler emits Linux x86-64 static ELF only",
        },
        {
          title: "WASM backend",
          detail: "Blocked draft (#2237); not shipped",
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
      label: "Native Backend (x86-64 ELF)",
      level: pick(nativeBackend, "selftest_passed") ? "verified" : "unknown",
      reason: nativeBackend
        ? `selftest_passed=${nativeBackend.selftest_passed}, scalar_smoke_present=${nativeBackend.scalar_smoke_present}, fail_closed=${nativeBackend.fail_closed}`
        : "artifact missing",
      artifact: "artifacts/omega/native_backend_v2_gate.v1.json",
    },
    selfHosted: {
      label: "Self-Hosted Compiler",
      // The omega report (2026-02-28) compares a bytecode cache, not ELFs; the live
      // fixed point is the lean_single `make build` chain gated in CI.
      level: "verified",
      reason:
        "lean_single seed: gen2 = gen3 byte-identical (make build; CI step 'Canonical lean_single fixed point'). Madaros (default) has no fixed point yet (scripts/ci/madaros_fixed_point_gate.sh).",
      artifact: "Makefile · .github/workflows/ci.yml",
    },
    cranelift: {
      label: "Cranelift JIT (retired)",
      level: "stub",
      reason: "No longer shipped since 2.1.0 (CHANGELOG.md). Madaros native x86-64 is the only shipped engine.",
      artifact: "CHANGELOG.md",
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
      level: "unknown",
      totals: stdlibE2e
        ? { pass: stdlibGatePass, fail: stdlibGateFail, skip: stdlibGateSkip, total: stdlibGateTotal }
        : {},
      reason: reliabilityReason,
      artifact: STDLIB_E2E_ARTIFACT,
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
console.log(`  - stdlib e2e: ${stdlibGatePass}/${stdlibGateTotal} pass (${stdlibGateFail} fail, ${stdlibGateSkip} skip), ${stdlibE2eDate}`);
console.log(`  - generatedAt: ${generatedAt}`);
