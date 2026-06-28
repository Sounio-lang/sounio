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
  const args = { url: DEFAULT_URL, browserName: "chromium", output: "", requireWebgpu: false };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--browser") args.browserName = argv[++i] || args.browserName;
    else if (arg === "--output") args.output = argv[++i] || "";
    else if (arg === "--require-webgpu") args.requireWebgpu = true;
    else if (!arg.startsWith("--")) args.url = arg;
  }
  return args;
}

async function writeReport(path, report) {
  if (!path) return;
  const fs = await import("node:fs/promises");
  await fs.writeFile(path, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

async function main() {
  const args = parseArgs(process.argv);
  const playwright = await loadPlaywright();
  const browserType = playwright[args.browserName];
  if (!browserType) throw new Error(`unsupported browser ${args.browserName}`);
  const launchOptions = { headless: true };
  if (args.browserName === "chromium") {
    launchOptions.args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,UseSkiaRenderer", "--ignore-gpu-blocklist"];
  }

  let browser;
  try {
    browser = await browserType.launch(launchOptions);
  } catch (error) {
    const report = {
      schema: "sounio.immersive_dissertation.webgpu_runtime_report.v1",
      url: args.url,
      browser: args.browserName,
      status: "browser_launch_failed",
      launchError: error.message,
    };
    await writeReport(args.output, report);
    console.error("WEBGPU_RUNTIME_BROWSER_LAUNCH_FAIL");
    console.error(JSON.stringify(report, null, 2));
    process.exit(1);
  }

  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(args.url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => document.querySelector("#runtime-badge")?.textContent?.includes("renderer:"), null, { timeout: 30000 });

  const report = await page.evaluate(async () => {
    const base = {
      schema: "sounio.immersive_dissertation.webgpu_runtime_report.v1",
      status: "not_available",
      probe: {
        navigatorGpu: Boolean(navigator.gpu),
        adapterAvailable: false,
        deviceAvailable: false,
        deviceLostHandlerRegistered: false,
      },
      runtimeBadge: document.querySelector("#runtime-badge")?.textContent || "",
      canvasCount: document.querySelectorAll("canvas").length,
      claimBoundary: "WebGPU may be promoted only after adapter, device, and PBPK kernel runtime checks pass.",
    };
    if (!navigator.gpu) return base;
    const adapter = await navigator.gpu.requestAdapter();
    base.probe.adapterAvailable = Boolean(adapter);
    if (!adapter) return base;
    const device = await adapter.requestDevice();
    base.probe.deviceAvailable = Boolean(device);
    if (!device) return base;
    base.probe.deviceLostHandlerRegistered = true;
    device.lost.then(() => {});
    device.destroy();
    base.status = "pass";
    return base;
  });

  report.url = args.url;
  report.browser = args.browserName;
  report.errors = errors;
  if (errors.length && report.status === "pass") report.status = "validation_failed";
  await writeReport(args.output, report);
  await browser.close();

  const ready = report.probe?.navigatorGpu && report.probe?.adapterAvailable && report.probe?.deviceAvailable;
  if (!ready) {
    console.log("WEBGPU_RUNTIME_NOT_AVAILABLE");
    console.log(JSON.stringify(report, null, 2));
    process.exit(args.requireWebgpu ? 1 : 0);
  }
  if (errors.length || report.status !== "pass") {
    console.error("WEBGPU_RUNTIME_FAIL");
    console.error(JSON.stringify(report, null, 2));
    process.exit(1);
  }
  console.log("WEBGPU_RUNTIME_PASS");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error("WEBGPU_RUNTIME_FAIL");
  console.error(error.message);
  process.exit(1);
});
