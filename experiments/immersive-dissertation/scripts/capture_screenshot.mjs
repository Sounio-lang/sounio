#!/usr/bin/env node

const DEFAULT_URL = "http://127.0.0.1:4173/";
const FALLBACK_PLAYWRIGHT = "/tmp/sounio-pw/node_modules/playwright/index.js";

async function loadPlaywright() {
  try {
    const module = await import("playwright");
    return module.default || module;
  } catch {
    const module = await import(process.env.PLAYWRIGHT_MODULE || FALLBACK_PLAYWRIGHT);
    return module.default || module;
  }
}

function parseArgs(argv) {
  const args = { url: DEFAULT_URL, output: "/tmp/sounio-immersive-recovery.png", browserName: "firefox" };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--output") args.output = argv[++i] || args.output;
    else if (arg === "--browser") args.browserName = argv[++i] || args.browserName;
    else if (!arg.startsWith("--")) args.url = arg;
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  const playwright = await loadPlaywright();
  const browserType = playwright[args.browserName];
  if (!browserType) throw new Error(`unsupported browser ${args.browserName}`);
  const browser = await browserType.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto(args.url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => document.querySelector("#model-kernel")?.textContent?.includes("GPU kernel"), null, { timeout: 30000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: args.output, fullPage: false });
  const badge = await page.locator("#runtime-badge").textContent();
  const quality = await page.locator("#quality-label").textContent();
  const canvasCount = await page.locator("canvas").count();
  await browser.close();
  console.log("IMMERSIVE_SCREENSHOT_CAPTURE_PASS");
  console.log(JSON.stringify({ url: args.url, output: args.output, browser: args.browserName, badge, quality, canvasCount }, null, 2));
}

main().catch((error) => {
  console.error("IMMERSIVE_SCREENSHOT_CAPTURE_FAIL");
  console.error(error.message);
  process.exit(1);
});
