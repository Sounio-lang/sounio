/**
 * Static performance audit — analyzes built HTML for Lighthouse-relevant
 * patterns without requiring a browser.
 *
 * Checks:
 *  1. Render-blocking resources (CSS/JS in <head> without async/defer)
 *  2. Image optimization (missing width/height, missing lazy loading)
 *  3. Font preloading strategy
 *  4. Meta viewport
 *  5. Document size distribution
 *  6. Hydration strategy (client:load vs client:idle vs client:visible)
 *  7. Critical resource count
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const distDir = path.join(root, 'dist');

async function* walk(dir) {
  const entries = await readdir(dir);
  for (const entry of entries) {
    const abs = path.join(dir, entry);
    const info = await stat(abs);
    if (info.isDirectory()) yield* walk(abs);
    else yield abs;
  }
}

function countPattern(html, re) {
  let count = 0;
  re.lastIndex = 0;
  while (re.exec(html) !== null) count++;
  return count;
}

async function run() {
  const results = {
    totalPages: 0,
    totalSizeKB: 0,
    largestPage: { file: '', sizeKB: 0 },
    pagesOver200KB: [],
    missingViewport: [],
    missingLangAttr: [],
    renderBlockingScripts: [],
    imagesWithoutDimensions: [],
    imagesWithoutLazy: [],
    totalImages: 0,
    lazyImages: 0,
    fontsPreloaded: 0,
    fontsNotPreloaded: 0,
    clientLoadComponents: 0,
    clientIdleComponents: 0,
    clientVisibleComponents: 0,
    clientMediaComponents: 0,
    totalJSFiles: 0,
    totalCSSFiles: 0,
    totalJSSizeKB: 0,
    totalCSSSizeKB: 0,
  };

  // Scan HTML pages
  for await (const abs of walk(distDir)) {
    if (abs.endsWith('.html')) {
      const rel = path.relative(distDir, abs);
      const html = await readFile(abs, 'utf8');

      // Skip redirect stubs (meta http-equiv="refresh")
      if (/meta\s+http-equiv=["']refresh["']/i.test(html)) continue;

      const sizeKB = Math.round(Buffer.byteLength(html) / 1024 * 10) / 10;

      results.totalPages++;
      results.totalSizeKB += sizeKB;

      if (sizeKB > results.largestPage.sizeKB) {
        results.largestPage = { file: rel, sizeKB };
      }
      if (sizeKB > 200) {
        results.pagesOver200KB.push({ file: rel, sizeKB });
      }

      // Viewport
      if (!/<meta\s[^>]*name=["']viewport["']/i.test(html)) {
        results.missingViewport.push(rel);
      }

      // Lang attribute
      if (!/<html\s[^>]*lang=/i.test(html)) {
        results.missingLangAttr.push(rel);
      }

      // Render-blocking scripts in <head> (not async, not defer, not type=module)
      const headMatch = html.match(/<head[\s>]([\s\S]*?)<\/head>/i);
      if (headMatch) {
        const head = headMatch[1];
        const scriptRe = /<script\b([^>]*)>/gi;
        let sm;
        while ((sm = scriptRe.exec(head)) !== null) {
          const attrs = sm[1];
          if (!/\basync\b/i.test(attrs) && !/\bdefer\b/i.test(attrs) &&
              !/\btype\s*=\s*["']module["']/i.test(attrs) &&
              !/\bis:inline\b/i.test(attrs) &&
              !/\btype\s*=\s*["']application\/ld\+json["']/i.test(attrs)) {
            // Check if it has a src (external) — inline scripts are less of an issue
            if (/\bsrc\s*=/i.test(attrs)) {
              results.renderBlockingScripts.push(rel);
              break;
            }
          }
        }
      }

      // Images
      const imgRe = /<img\b([^>]*)>/gi;
      let im;
      while ((im = imgRe.exec(html)) !== null) {
        const attrs = im[1];
        results.totalImages++;
        if (!/\bwidth\s*=/i.test(attrs) || !/\bheight\s*=/i.test(attrs)) {
          // Only flag if not in a <picture> with explicit dimensions
          if (!/<picture[\s\S]{0,500}$/.test(html.slice(0, im.index))) {
            results.imagesWithoutDimensions.push({ file: rel, snippet: im[0].slice(0, 80) });
          }
        }
        if (/\bloading\s*=\s*["']lazy["']/i.test(attrs)) {
          results.lazyImages++;
        } else {
          // Only warn for below-fold images (not hero images)
          if (im.index > 2000) {
            results.imagesWithoutLazy.push({ file: rel, snippet: im[0].slice(0, 80) });
          }
        }
      }
    }

    // Aggregate JS/CSS sizes
    if (abs.endsWith('.js') || abs.endsWith('.mjs')) {
      results.totalJSFiles++;
      const info = await stat(abs);
      results.totalJSSizeKB += Math.round(info.size / 1024 * 10) / 10;
    }
    if (abs.endsWith('.css')) {
      results.totalCSSFiles++;
      const info = await stat(abs);
      results.totalCSSSizeKB += Math.round(info.size / 1024 * 10) / 10;
    }
  }

  // Scan Astro source for hydration directives
  const srcDir = path.join(root, 'src');
  for await (const abs of walk(srcDir)) {
    if (!abs.endsWith('.astro') && !abs.endsWith('.mdx')) continue;
    const src = await readFile(abs, 'utf8');
    results.clientLoadComponents += countPattern(src, /client:load/g);
    results.clientIdleComponents += countPattern(src, /client:idle/g);
    results.clientVisibleComponents += countPattern(src, /client:visible/g);
    results.clientMediaComponents += countPattern(src, /client:media/g);
  }

  // Check font preloading
  const indexHtml = path.join(distDir, 'index.html');
  try {
    const html = await readFile(indexHtml, 'utf8');
    results.fontsPreloaded = countPattern(html, /rel=["']preload["'][^>]*as=["']font["']/gi);
    results.fontsNotPreloaded = countPattern(html, /\bfont-face\b/gi);
  } catch {}

  // Report
  console.log('=== Sounio Website Performance Audit ===\n');

  console.log(`Pages: ${results.totalPages}`);
  console.log(`Total HTML size: ${Math.round(results.totalSizeKB)} KB (avg ${Math.round(results.totalSizeKB / results.totalPages)} KB/page)`);
  console.log(`Largest page: ${results.largestPage.file} (${results.largestPage.sizeKB} KB)`);
  console.log(`JS bundle: ${results.totalJSFiles} files, ${Math.round(results.totalJSSizeKB)} KB total`);
  console.log(`CSS bundle: ${results.totalCSSFiles} files, ${Math.round(results.totalCSSSizeKB)} KB total`);

  console.log(`\nHydration directives:`);
  console.log(`  client:load    ${results.clientLoadComponents} (eager — blocks interaction)`);
  console.log(`  client:idle    ${results.clientIdleComponents} (deferred — good)`);
  console.log(`  client:visible ${results.clientVisibleComponents} (lazy — best for below-fold)`);
  console.log(`  client:media   ${results.clientMediaComponents} (conditional)`);

  console.log(`\nImages: ${results.totalImages} total, ${results.lazyImages} lazy-loaded`);

  let warnings = 0;

  if (results.pagesOver200KB.length > 0) {
    console.log(`\n⚠ ${results.pagesOver200KB.length} page(s) over 200 KB:`);
    for (const p of results.pagesOver200KB.slice(0, 10)) {
      console.log(`  ${p.file}: ${p.sizeKB} KB`);
    }
    warnings += results.pagesOver200KB.length;
  }

  if (results.missingViewport.length > 0) {
    console.log(`\n⚠ ${results.missingViewport.length} page(s) missing viewport meta`);
    warnings += results.missingViewport.length;
  }

  if (results.missingLangAttr.length > 0) {
    console.log(`\n⚠ ${results.missingLangAttr.length} page(s) missing lang attribute`);
    warnings += results.missingLangAttr.length;
  }

  if (results.renderBlockingScripts.length > 0) {
    console.log(`\n⚠ ${results.renderBlockingScripts.length} page(s) with render-blocking external scripts in <head>`);
    for (const f of results.renderBlockingScripts.slice(0, 5)) {
      console.log(`  ${f}`);
    }
    warnings += results.renderBlockingScripts.length;
  }

  if (results.imagesWithoutDimensions.length > 0) {
    console.log(`\n⚠ ${results.imagesWithoutDimensions.length} image(s) missing width/height (causes CLS):`);
    for (const i of results.imagesWithoutDimensions.slice(0, 10)) {
      console.log(`  ${i.file}: ${i.snippet}...`);
    }
    warnings += results.imagesWithoutDimensions.length;
  }

  if (results.imagesWithoutLazy.length > 0) {
    console.log(`\n⚠ ${results.imagesWithoutLazy.length} below-fold image(s) without loading="lazy":`);
    for (const i of results.imagesWithoutLazy.slice(0, 10)) {
      console.log(`  ${i.file}: ${i.snippet}...`);
    }
    warnings += results.imagesWithoutLazy.length;
  }

  if (warnings === 0) {
    console.log('\nNo performance warnings found.');
  } else {
    console.log(`\n${warnings} total warning(s).`);
  }
}

await run();
