#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const mode = process.argv[2] ?? 'generate';

if (!['generate', 'check'].includes(mode)) {
  console.error('Usage: node scripts/render-assets.mjs [generate|check]');
  process.exit(1);
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(websiteDir, '..');
const soucBin = path.join(repoRoot, 'bin/souc');
const outputDir = path.join(websiteDir, 'public/assets/generated/render');
const manifestPath = path.join(outputDir, 'manifest.json');

const renderSpecs = [
  {
    example: 'examples/render/coverage_crystal_atelier.sio',
    sourceAsset: 'examples/render/assets/coverage-crystal-atelier.png',
    assetFile: 'coverage-crystal-atelier.png',
    title: 'Coverage crystal atelier',
    description:
      'Deterministic Phong material study with four-sample edge coverage and fixed-point depth authored in Sounio.',
    width: 320,
    height: 200,
    engine: 'lean_single',
    sha256: '89c3db4423165068a13bbf4e78891d0e69da04dc245bf36f85bfe42e0fccf7c1',
    renderSha256: 'b6135cd4d037840ac4036d7e8efc8dbeb41cd99658dcd8784d5c13d0b7169ea8',
    verification: 'byte-identical output across two independent runs',
    gate: 'tests/run-pass/viz_renderer3d_coverage4.sio',
    receipt: 'VIZ_RENDERER3D_COVERAGE4_PASS',
    sourceRef: 'codex/renderer-quality-20260715',
    command:
      'SOUNIO_SOUC_ENGINE=lean_single bin/souc run examples/render/coverage_crystal_atelier.sio > coverage_crystal_atelier.ppm',
  },
  {
    example: 'examples/render/rapamycin_material_study.sio',
    sourceAsset: 'examples/render/assets/rapamycin-material-study.png',
    assetFile: 'rapamycin-material-study.png',
    title: 'Idealized rapamycin scaffold',
    description:
      'Deterministic ball-and-stick receipt for the stdlib coarse 44-site, 45-bond scaffold; not a crystallographic reconstruction.',
    width: 480,
    height: 300,
    engine: 'lean_single',
    sha256: '8510ca96178aa1af29f2aefdbab973dd408a51a334e12d550c423fa12c0526ac',
    renderSha256: '289767f0c6bfa6db548d802650da7de9cc30b0f7395d3d47e8ee47841a8cacdc',
    verification: 'byte-identical output across two independent runs',
    gate: 'tests/run-pass/viz_molecule_headless_receipt.sio',
    receipt: 'VIZ_MOLECULE_HEADLESS_RECEIPT_PASS',
    sourceRef: 'website/living-observatory-20260713',
    command:
      'SOUNIO_SOUC_ENGINE=lean_single bin/souc run examples/render/rapamycin_material_study.sio > rapamycin_material_study.ppm',
  },
  {
    example: 'examples/render/epistemic_field_atelier.sio',
    sourceAsset: 'examples/render/assets/epistemic-field-atelier.png',
    assetFile: 'epistemic-field-atelier.png',
    title: 'Epistemic frontier',
    description:
      'Illustrative dual-channel field separating first-order variance from PBox mean ambiguity; not empirical or clinical.',
    width: 480,
    height: 300,
    engine: 'lean_single',
    sha256: '2ad6b49e1fe0ee0cf50149727882bd0476a26c0a20a12975ae6a8cff7f3128c3',
    renderSha256: 'dfd7e288f0e91221400f89e830e48c16f97a0d47b45ce59d54c33bc83f812465',
    verification: 'byte-identical output across two independent runs',
    gate: 'tests/run-pass/viz_epistemic_field_receipt.sio',
    receipt: 'VIZ_EPISTEMIC_FIELD_RECEIPT_PASS',
    sourceRef: 'website/living-observatory-20260713',
    command:
      'SOUNIO_SOUC_ENGINE=lean_single bin/souc run examples/render/epistemic_field_atelier.sio > epistemic_field_atelier.ppm',
  },
  {
    example: 'examples/render/causal_intervention_frontier.sio',
    sourceAsset: 'examples/render/assets/causal-intervention-frontier.png',
    assetFile: 'causal-intervention-frontier.png',
    title: 'Causal intervention refusal',
    description:
      'A fixed-DAG backdoor witness: native do(X) graph surgery preserves the latent path and refuses an observed-data effect estimate.',
    width: 480,
    height: 300,
    engine: 'lean_single',
    sha256: 'eae2f0a22b8a8df65fd181849055360fb2ed0115fb85dac539c8284011b1f1d8',
    renderSha256: '6de2b1f6cba8e7290757e09e6d337f8559dd4a5a0ed777b32900d65b6ab7413a',
    verification: 'byte-identical output across two independent runs',
    gate: 'tests/run-pass/viz_causal_intervention_receipt.sio',
    receipt: 'VIZ_CAUSAL_INTERVENTION_RECEIPT_PASS',
    sourceRef: 'website/living-observatory-20260713',
    command:
      'SOUNIO_SOUC_ENGINE=lean_single bin/souc run examples/render/causal_intervention_frontier.sio > causal_intervention_frontier.ppm',
  },
  {
    example: 'examples/render/triangle_basic.sio',
    assetFile: 'triangle-basic.svg',
    title: 'Triangle raster render',
    description:
      'Software-rasterized triangle with barycentric color interpolation authored in Sounio.',
    command:
      "bin/souc run examples/render/triangle_basic.sio | sed -n '/^P3$/,$p' > triangle_basic.ppm",
  },
  {
    example: 'examples/render/cube_wireframe.sio',
    assetFile: 'cube-wireframe.svg',
    title: 'Wireframe cube render',
    description:
      'Perspective-projected cube with depth-tinted edges preserved as a checked-in Sounio render asset.',
    command:
      "bin/souc run examples/render/cube_wireframe.sio | sed -n '/^P3$/,$p' > cube_wireframe.ppm",
  },
  {
    example: 'examples/render/uncertainty_field.sio',
    assetFile: 'uncertainty-field.svg',
    title: 'Uncertainty field render',
    description:
      'Integer-scaled uncertainty heatmap where value controls color and epsilon controls brightness.',
    command:
      "bin/souc run examples/render/uncertainty_field.sio | sed -n '/^P3$/,$p' > uncertainty_field.ppm",
  },
  {
    example: 'examples/render/quaternion_rotation.sio',
    assetFile: 'quaternion-rotation.svg',
    title: 'Quaternion rotation render',
    description:
      'Tetrahedron wireframe produced by the example custom quaternion struct and its documented approximation.',
    command:
      "bin/souc run examples/render/quaternion_rotation.sio | sed -n '/^P3$/,$p' > quaternion_rotation.ppm",
  },
];

function escapeXml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function toHex(value) {
  return value.toString(16).padStart(2, '0');
}

function rgbHex(r, g, b) {
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function parsePpm(ppmText, example) {
  const outputTokens = ppmText.trim().split(/\s+/);
  const headerIndex = outputTokens.indexOf('P3');
  if (headerIndex < 0) {
    throw new Error(`${example}: expected P3 header`);
  }
  const tokens = outputTokens.slice(headerIndex);

  const width = Number(tokens[1]);
  const height = Number(tokens[2]);
  const maxValue = Number(tokens[3]);

  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`${example}: invalid image dimensions`);
  }
  if (maxValue !== 255) {
    throw new Error(`${example}: expected max value 255, got ${maxValue}`);
  }

  const payload = tokens.slice(4);
  const expected = width * height * 3;
  if (payload.length !== expected) {
    throw new Error(`${example}: expected ${expected} color entries, got ${payload.length}`);
  }

  const pixels = new Array(width * height);
  for (let i = 0; i < pixels.length; i += 1) {
    const offset = i * 3;
    const r = Number(payload[offset]);
    const g = Number(payload[offset + 1]);
    const b = Number(payload[offset + 2]);
    if (
      !Number.isInteger(r) ||
      !Number.isInteger(g) ||
      !Number.isInteger(b) ||
      r < 0 ||
      r > 255 ||
      g < 0 ||
      g > 255 ||
      b < 0 ||
      b > 255
    ) {
      throw new Error(`${example}: invalid RGB triplet at pixel ${i}`);
    }
    pixels[i] = rgbHex(r, g, b);
  }

  return { width, height, pixels };
}

function ppmToSvg(ppm, spec) {
  const relExample = escapeXml(spec.example);
  const relCommand = escapeXml(spec.command);
  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${ppm.width} ${ppm.height}" width="${ppm.width}" height="${ppm.height}" shape-rendering="crispEdges" role="img" aria-labelledby="title desc">`,
    `  <title id="title">${escapeXml(spec.title)}</title>`,
    `  <desc id="desc">${escapeXml(spec.description)} Source: ${relExample}. Command: ${relCommand}.</desc>`,
  ];

  for (let y = 0; y < ppm.height; y += 1) {
    let runStart = 0;
    let currentColor = ppm.pixels[y * ppm.width];

    for (let x = 1; x <= ppm.width; x += 1) {
      const nextColor = x < ppm.width ? ppm.pixels[y * ppm.width + x] : null;
      if (nextColor === currentColor) {
        continue;
      }

      lines.push(
        `  <rect x="${runStart}" y="${y}" width="${x - runStart}" height="1" fill="${currentColor}"/>`
      );
      runStart = x;
      currentColor = nextColor;
    }
  }

  lines.push('</svg>', '');
  return lines.join('\n');
}

function buildManifest(entries) {
  return `${JSON.stringify(
    {
      generated_by: 'website/scripts/render-assets.mjs',
      compiler_artifact: path.relative(repoRoot, soucBin),
      assets: entries.map((entry) => ({
        example: entry.example,
        asset: `public/assets/generated/render/${entry.assetFile}`,
        public_path: `/assets/generated/render/${entry.assetFile}`,
        width: entry.width,
        height: entry.height,
        command: entry.command,
        ...(entry.engine ? { engine: entry.engine } : {}),
        ...(entry.sourceAsset ? { source_asset: entry.sourceAsset } : {}),
        ...(entry.sha256 ? { sha256: entry.sha256 } : {}),
        ...(entry.renderSha256 ? { render_sha256: entry.renderSha256 } : {}),
        ...(entry.verification ? { verification: entry.verification } : {}),
        ...(entry.gate ? { gate: entry.gate } : {}),
        ...(entry.receipt ? { receipt: entry.receipt } : {}),
        ...(entry.sourceRef ? { source_ref: entry.sourceRef } : {}),
      })),
    },
    null,
    2
  )}\n`;
}

function runExample(example) {
  try {
    return execFileSync(soucBin, ['run', example], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (error) {
    const stdout = typeof error.stdout === 'string' ? error.stdout : '';
    const stderr = typeof error.stderr === 'string' ? error.stderr : '';
    if (stdout) {
      process.stderr.write(stdout);
    }
    if (stderr) {
      process.stderr.write(stderr);
    }
    throw new Error(`failed to render ${example}`);
  }
}

function verifyStaticReceipt(spec) {
  const env = { ...process.env, SOUNIO_SOUC_ENGINE: spec.engine };
  const output = execFileSync(soucBin, ['run', spec.example], {
    cwd: repoRoot,
    env,
    maxBuffer: 64 * 1024 * 1024,
  });
  const digest = createHash('sha256').update(output).digest('hex');
  if (digest !== spec.renderSha256) {
    throw new Error(`${spec.example}: expected render sha256 ${spec.renderSha256}, got ${digest}`);
  }

  const ppm = parsePpm(output.toString('utf8'), spec.example);
  if (ppm.width !== spec.width || ppm.height !== spec.height) {
    throw new Error(`${spec.example}: expected ${spec.width}x${spec.height}, got ${ppm.width}x${ppm.height}`);
  }

  const gateOutput = execFileSync(soucBin, ['run', spec.gate], {
    cwd: repoRoot,
    env,
    encoding: 'utf8',
    maxBuffer: 8 * 1024 * 1024,
  });
  if (!gateOutput.split(/\r?\n/).includes(spec.receipt)) {
    throw new Error(`${spec.gate}: missing receipt ${spec.receipt}`);
  }
}

async function collectAssets() {
  if (!existsSync(soucBin)) {
    console.log(`SKIP: compiler not found at ${path.relative(repoRoot, soucBin)} — using pre-rendered assets`);
    return [];
  }

  const assets = [];
  for (const spec of renderSpecs) {
    if (spec.sourceAsset) {
      const binary = await readFile(path.join(repoRoot, spec.sourceAsset));
      const digest = createHash('sha256').update(binary).digest('hex');
      if (digest !== spec.sha256) {
        throw new Error(`${spec.sourceAsset}: expected sha256 ${spec.sha256}, got ${digest}`);
      }
      assets.push({ ...spec, binary });
      continue;
    }
    const ppmText = runExample(spec.example);
    const ppm = parsePpm(ppmText, spec.example);
    assets.push({
      ...spec,
      width: ppm.width,
      height: ppm.height,
      svg: ppmToSvg(ppm, spec),
    });
  }
  return assets;
}

async function generate() {
  const assets = await collectAssets();
  await mkdir(outputDir, { recursive: true });

  for (const asset of assets) {
    if (asset.binary) {
      await writeFile(path.join(outputDir, asset.assetFile), asset.binary);
    } else {
      await writeFile(path.join(outputDir, asset.assetFile), asset.svg, 'utf8');
    }
  }
  await writeFile(manifestPath, buildManifest(assets), 'utf8');

  console.log(`Generated ${assets.length} render assets in ${path.relative(repoRoot, outputDir)}.`);
}

async function check() {
  const staticSpecs = renderSpecs.filter((spec) => spec.sourceAsset);
  const currentManifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const expectedStaticEntries = JSON.parse(buildManifest(staticSpecs)).assets;
  for (let index = 0; index < staticSpecs.length; index += 1) {
    const spec = staticSpecs[index];
    const source = await readFile(path.join(repoRoot, spec.sourceAsset));
    const published = await readFile(path.join(outputDir, spec.assetFile));
    const digest = createHash('sha256').update(source).digest('hex');
    const manifestEntry = currentManifest.assets.find((entry) => entry.example === spec.example);
    const manifestMatches =
      manifestEntry && JSON.stringify(manifestEntry) === JSON.stringify(expectedStaticEntries[index]);
    if (digest !== spec.sha256 || !source.equals(published) || !manifestMatches) {
      console.error(`Render asset is stale: ${spec.assetFile}`);
      process.exit(1);
    }
    if (existsSync(soucBin)) {
      try {
        verifyStaticReceipt(spec);
      } catch (error) {
        console.error(`Verified render receipt failed: ${error.message}`);
        process.exit(1);
      }
    }
  }
  console.log(`OK: ${staticSpecs.length} deterministic static render receipts and gates verified.`);

  let assets;
  try {
    assets = await collectAssets();
  } catch (e) {
    console.log(`SKIP: render check failed (${e.message}) — using pre-rendered assets`);
    console.log('OK: render-assets check skipped (current render path incomplete).');
    return;
  }
  const stale = [];

  for (const asset of assets) {
    const assetPath = path.join(outputDir, asset.assetFile);
    let current;
    try {
      current = asset.binary ? await readFile(assetPath) : await readFile(assetPath, 'utf8');
    } catch {
      stale.push(`${path.relative(repoRoot, assetPath)} is missing`);
      continue;
    }

    const matches = asset.binary ? current.equals(asset.binary) : current === asset.svg;
    if (!matches) {
      stale.push(`${path.relative(repoRoot, assetPath)} is out of date`);
    }
  }

  const expectedManifest = buildManifest(assets);
  try {
    const currentManifest = await readFile(manifestPath, 'utf8');
    if (currentManifest !== expectedManifest) {
      stale.push(`${path.relative(repoRoot, manifestPath)} is out of date`);
    }
  } catch {
    stale.push(`${path.relative(repoRoot, manifestPath)} is missing`);
  }

  if (stale.length > 0) {
    console.error('Render assets are stale:');
    for (const message of stale) {
      console.error(`- ${message}`);
    }
    console.error('Run: npm --prefix website run generate:render-assets');
    process.exit(1);
  }

  console.log(`OK: ${assets.length} render assets match checked compiler output.`);
}

if (mode === 'generate') {
  await generate();
} else {
  await check();
}
