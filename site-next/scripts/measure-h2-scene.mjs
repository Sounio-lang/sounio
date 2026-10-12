#!/usr/bin/env node
// ADR009: structural orchestration only. Scientific extraction/arithmetic is C++23.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const REPO = join(dirname(fileURLToPath(import.meta.url)), '../..');
const REF = process.env.CORPUS_REF ?? 'HEAD';
const SRC = 'benchmarks/chemistry/RESULTS.md';
const OUT = join(REPO, 'artifacts/site/h2_scene.v1.json');
const git = (...a) => execFileSync('git', ['-C', REPO, ...a], { encoding: 'utf-8', maxBuffer: 1 << 30 });
const temporary = mkdtempSync(join(tmpdir(), 'sounio-h2-oracle-'));
try {
  const source = join(temporary, 'RESULTS.md'), binary = join(temporary, 'h2-scene');
  writeFileSync(source, git('show', `${REF}:${SRC}`));
  execFileSync(process.env.CXX ?? 'c++', ['-std=c++23', '-O2', '-Wall', '-Wextra', '-Werror', join(REPO, 'site-next/science/h2_scene.cpp'), '-o', binary], { stdio: 'inherit' });
  const result = JSON.parse(execFileSync(binary, [source], { encoding: 'utf-8' }));
  const artifact = {
    schema: 'sounio.site.h2_scene.v2', generated_at: new Date().toISOString(),
    reason: 'Recorded C++23/Sounio tables normalized by C++23; historical measurements, not fresh chemistry execution.',
    ref: REF, source: SRC, commit: git('rev-parse', REF).trim(),
    source_blob: git('rev-parse', `${REF}:${SRC}`).trim(),
    oracle_blob: git('hash-object', join(REPO, 'site-next/science/h2_scene.cpp')).trim(),
    oracle: result.oracle,
    metrics: result.metrics,
  };
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify(artifact, null, 2) + '\n');
  console.log(`written ${OUT} by ${artifact.oracle}`);
} finally { rmSync(temporary, { recursive: true, force: true }); }
