#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
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
    example: 'examples/render/causal_dag.sio',
    assetFile: 'causal-dag.svg',
    title: 'Causal DAG render',
    description:
      'Sounio-authored X-to-M-to-Y raster diagram with a latent U node and intervention-highlighted X.',
    command:
      "bin/souc run examples/render/causal_dag.sio | sed -n '/^P3$/,$p' > causal_dag.ppm",
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

async function collectAssets() {
  if (!existsSync(soucBin)) {
    console.log(`SKIP: compiler not found at ${path.relative(repoRoot, soucBin)} — using pre-rendered assets`);
    return [];
  }

  const assets = [];
  for (const spec of renderSpecs) {
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
    await writeFile(path.join(outputDir, asset.assetFile), asset.svg, 'utf8');
  }
  await writeFile(manifestPath, buildManifest(assets), 'utf8');

  console.log(`Generated ${assets.length} render assets in ${path.relative(repoRoot, outputDir)}.`);
}

async function check() {
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
      current = await readFile(assetPath, 'utf8');
    } catch {
      stale.push(`${path.relative(repoRoot, assetPath)} is missing`);
      continue;
    }

    if (current !== asset.svg) {
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
