#!/usr/bin/env node
// ADR009: structural orchestration only. Scientific extraction/arithmetic is C++23.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const REPO = join(dirname(fileURLToPath(import.meta.url)), '../..');
const REF = process.env.CORPUS_REF ?? 'HEAD';
const SRC = 'examples/real_world/01_dose_uncertainty.sio';
const OUT = join(REPO, 'artifacts/site/dose_scene.v1.json');
const git = (...a) => execFileSync('git', ['-C', REPO, ...a], { encoding: 'utf-8', maxBuffer: 1 << 30 });
const temporary = mkdtempSync(join(tmpdir(), 'sounio-dose-oracle-'));
try {
  const source = join(temporary, 'source.sio'), binary = join(temporary, 'dose-scene');
  writeFileSync(source, git('show', `${REF}:${SRC}`));
  execFileSync(process.env.CXX ?? 'c++', ['-std=c++23', '-O2', '-Wall', '-Wextra', '-Werror', join(REPO, 'site-next/science/dose_scene.cpp'), '-o', binary], { stdio: 'inherit' });
  const result = JSON.parse(execFileSync(binary, [source], { encoding: 'utf-8' }));
  const artifact = {
    schema: 'sounio.site.dose_scene.v2', generated_at: new Date().toISOString(),
    reason: 'Source-linked pedagogical dose scene. C++23 delta-method arithmetic; no clinical validation implied.',
    ref: REF, source: SRC, commit: git('rev-parse', REF).trim(),
    source_blob: git('rev-parse', `${REF}:${SRC}`).trim(),
    oracle_blob: git('hash-object', join(REPO, 'site-next/science/dose_scene.cpp')).trim(),
    oracle: result.oracle,
    metrics: { ...result.metrics, default_reading: result.default_reading, readings: result.readings },
  };
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify(artifact, null, 2) + '\n');
  console.log(`written ${OUT} by ${artifact.oracle}`);
} finally { rmSync(temporary, { recursive: true, force: true }); }
