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

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function main() {
  const url = process.argv[2] || DEFAULT_URL;
  const { firefox } = await loadPlaywright();
  const browser = await firefox.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => {
    const badge = document.querySelector("#runtime-badge")?.textContent || "";
    const kernel = document.querySelector("#model-kernel")?.textContent || "";
    return badge.includes("renderer:") && kernel.includes("GPU kernel") && kernel.includes("CYP2D6");
  }, null, { timeout: 30000 });
  await page.waitForTimeout(1200);
  const badge = await page.locator("#runtime-badge").textContent();
  assert(badge?.includes("WebGPU"), "runtime badge did not report WebGPU capability");
  assert(badge?.includes("WebGL"), "runtime badge did not report WebGL capability");
  const kernel = await page.locator("#model-kernel").textContent();
  assert(kernel?.includes("2DGX ready"), "model kernel did not expose 2DGX readiness");
  assert(kernel?.includes("WGSL 64 threads"), "model kernel did not expose WGSL workgroup size");
  assert(kernel?.includes("model curve only; observed C(t) blocked"), "clinical firewall missing from kernel");
  const firewall = await page.locator("#clinical-firewall").textContent();
  assert(firewall?.includes("no raw observed C(t) in browser"), "browser clinical firewall missing");
  const gpuKernel = await page.locator("#gpu-kernel-label").textContent();
  assert(gpuKernel?.includes("WGSL ready"), "footer GPU kernel label missing");
  const canvasCount = await page.locator("canvas").count();
  assert(canvasCount >= 2, "expected scene canvas and curve canvas");
  assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
  assert(consoleErrors.length === 0, `console errors: ${consoleErrors.join(" | ")}`);
  await browser.close();
  console.log("IMMERSIVE_BROWSER_INTERACTION_PASS");
  console.log(JSON.stringify({ url, badge, canvasCount }, null, 2));
}

main().catch((error) => {
  console.error("IMMERSIVE_BROWSER_INTERACTION_FAIL");
  console.error(error.message);
  process.exit(1);
});
