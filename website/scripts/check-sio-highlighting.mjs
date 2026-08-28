import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();

function toPosix(p) {
  return p.split(path.sep).join('/');
}

function stripExtAndIndex(relPath) {
  let slug = toPosix(relPath).replace(/\.mdx$/i, '');
  if (slug.endsWith('/index')) slug = slug.slice(0, -'/index'.length);
  return slug;
}

async function walkMdx(dir) {
  const out = [];
  const stack = [dir];

  while (stack.length > 0) {
    const current = stack.pop();
    const entries = await readdir(current, { withFileTypes: true });

    for (const entry of entries) {
      const abs = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(abs);
      } else if (entry.isFile() && entry.name.endsWith('.mdx')) {
        out.push(abs);
      }
    }
  }

  return out;
}

function countFencesByLang(content, lang) {
  const re = /^```([^\s`]*)?/gim;
  let count = 0;
  let match;
  while ((match = re.exec(content)) !== null) {
    const fenceLang = (match[1] || 'plaintext').toLowerCase();
    if (fenceLang === lang) count += 1;
  }
  return count;
}

function countRenderedByLang(html, lang) {
  const re = /data-language="([^"]+)"/g;
  let count = 0;
  let match;
  while ((match = re.exec(html)) !== null) {
    if (match[1].toLowerCase() === lang) count += 1;
  }
  return count;
}

async function validateSourceSet({ sourceDir, mapToDist }) {
  const errors = [];
  let checked = 0;

  const sourceAbs = path.join(root, sourceDir);
  const files = await walkMdx(sourceAbs);

  for (const abs of files) {
    const rel = path.relative(sourceAbs, abs);
    const source = await readFile(abs, 'utf8');
    const sourceSioCount = countFencesByLang(source, 'sio');
    if (sourceSioCount === 0) continue;

    const sourceRustCount = countFencesByLang(source, 'rust');
    const distRel = mapToDist(rel);
    const distAbs = path.join(root, distRel);

    let html;
    try {
      html = await readFile(distAbs, 'utf8');
    } catch {
      errors.push(`Missing rendered file for sio fence source: ${toPosix(path.relative(root, abs))} -> ${toPosix(distRel)}`);
      continue;
    }

    const renderedRustCount = countRenderedByLang(html, 'rust');
    const renderedSioCount = countRenderedByLang(html, 'sio');

    // sio fences must render under their own language id (real grammar,
    // website/src/shiki/sounio.tmLanguage.json) — never remapped to rust.
    // Remapping to rust was the defect this check used to require: it
    // tokenized every &!, ++, var, and with-clause using Rust's keywords,
    // on a language whose own CLAUDE.md lists those as compile errors.
    if (renderedSioCount < sourceSioCount) {
      errors.push(
        `Missing real sio-highlighted blocks: ${toPosix(path.relative(root, abs))} -> ${toPosix(distRel)} (expected >= ${sourceSioCount} data-language="sio" blocks, got ${renderedSioCount})`
      );
    }

    // Native rust fences (unrelated to any sio content) still owe their own
    // rendered blocks — sio fences must not silently satisfy this count.
    if (renderedRustCount < sourceRustCount) {
      errors.push(
        `Insufficient rust-highlighted blocks for native rust fences: ${toPosix(path.relative(root, abs))} (expected >= ${sourceRustCount}, got ${renderedRustCount})`
      );
    }

    checked += 1;
  }

  return { checked, errors };
}

async function run() {
  const docsResult = await validateSourceSet({
    sourceDir: 'src/content/docs',
    mapToDist: (rel) => {
      const relPosix = toPosix(rel);
      const [locale, ...rest] = relPosix.split('/');
      const slug = stripExtAndIndex(rest.join('/'));
      const base = locale === 'en' ? 'dist/learn' : `dist/${locale}/learn`;
      return slug.length > 0 ? `${base}/${slug}/index.html` : `${base}/index.html`;
    },
  });

  const blogResult = await validateSourceSet({
    sourceDir: 'src/content/blog',
    mapToDist: (rel) => {
      const relPosix = toPosix(rel);
      const parts = relPosix.split('/');
      const locales = ['el', 'es', 'ja', 'pt', 'zh', 'zh-hk'];
      if (parts.length > 1 && locales.includes(parts[0])) {
        const locale = parts[0];
        const slug = stripExtAndIndex(parts.slice(1).join('/'));
        return slug.length > 0 ? `dist/${locale}/insights/${slug}/index.html` : `dist/${locale}/insights/index.html`;
      }
      const slug = stripExtAndIndex(relPosix);
      return slug.length > 0 ? `dist/insights/${slug}/index.html` : 'dist/insights/index.html';
    },
  });

  const showcaseResult = await validateSourceSet({
    sourceDir: 'src/content/showcases',
    mapToDist: (rel) => {
      const relPosix = toPosix(rel);
      const parts = relPosix.split('/');
      const locales = ['el', 'es', 'ja', 'pt', 'zh', 'zh-hk'];
      if (parts.length > 1 && locales.includes(parts[0])) {
        // Locale-prefixed showcase: e.g. pt/causal.mdx -> dist/pt/science/causal/index.html
        const locale = parts[0];
        const slug = stripExtAndIndex(parts.slice(1).join('/'));
        return slug.length > 0 ? `dist/${locale}/science/${slug}/index.html` : `dist/${locale}/science/index.html`;
      }
      const slug = stripExtAndIndex(relPosix);
      return slug.length > 0 ? `dist/science/${slug}/index.html` : 'dist/science/index.html';
    },
  });

  const errors = [...docsResult.errors, ...blogResult.errors, ...showcaseResult.errors];
  const checked = docsResult.checked + blogResult.checked + showcaseResult.checked;

  if (errors.length > 0) {
    console.error('Sio highlighting validation failed:');
    for (const err of errors) console.error(`- ${err}`);
    process.exit(1);
  }

  console.log(`OK: sio highlighting validated on ${checked} rendered pages.`);
}

await run();
