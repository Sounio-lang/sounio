#!/usr/bin/env node

const DEFAULT_URL = "http://127.0.0.1:4173/";
const FALLBACK_PLAYWRIGHT = "/tmp/sounio-pw/node_modules/playwright/index.js";

async function loadPlaywright() {
  try {
    const module = await import("playwright");
    return module.default || module;
  } catch (firstError) {
    const fallback = process.env.PLAYWRIGHT_MODULE || FALLBACK_PLAYWRIGHT;
    const module = await import(fallback);
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

function validateKernelReport(report) {
  const errors = [];
  if (!String(report.claimBoundary || "").includes("not observed C(t) calibration")) {
    errors.push("missing not observed C(t) calibration boundary");
  }
  if (report.status !== "pass") return errors;
  for (const key of ["navigatorGpu", "adapterAvailable", "deviceAvailable"]) {
    if (!report.probe?.[key]) errors.push(`${key} false`);
  }
  if (report.kernel?.workgroupSize !== 64) errors.push("unexpected workgroup size");
  const outputs = report.kernel?.outputs || [];
  if (outputs.length < 4) errors.push("missing kernel outputs");
  for (const [index, row] of outputs.entries()) {
    for (const key of ["release_fraction", "parent_ng_ml", "odv_ng_ml", "odv_parent_ratio"]) {
      const value = row[key];
      if (!Number.isFinite(value)) errors.push(`row ${index} ${key} is not finite`);
      if (value < 0) errors.push(`row ${index} ${key} is negative`);
    }
    if (row.release_fraction > 1.0001) errors.push(`row ${index} release_fraction > 1`);
  }
  const early = outputs.find((row) => row.time_h === 1) || {};
  const parentPeak = outputs.find((row) => row.time_h === 5.5) || {};
  const odvLater = outputs.find((row) => row.time_h === 9) || {};
  if ((parentPeak.parent_ng_ml || 0) <= (early.parent_ng_ml || 0)) {
    errors.push("parent modified-release peak did not exceed early value");
  }
  if ((odvLater.odv_ng_ml || 0) <= (early.odv_ng_ml || 0)) {
    errors.push("ODV formation did not increase after early value");
  }
  return errors;
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
      schema: "sounio.immersive_dissertation.webgpu_pbpk_kernel_runtime_report.v1",
      url: args.url,
      browser: args.browserName,
      status: "browser_launch_failed",
      launchError: error.message,
    };
    await writeReport(args.output, report);
    console.error("WEBGPU_PBPK_KERNEL_RUNTIME_BROWSER_LAUNCH_FAIL");
    console.error(JSON.stringify(report, null, 2));
    process.exit(1);
  }

  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const errors = [];
  const warnings = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
    if (message.type() === "warning") warnings.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(args.url, { waitUntil: "domcontentloaded", timeout: 30000 });

  const report = await page.evaluate(async () => {
    const contract = await fetch("./data/webgpu-pbpk-kernel-contract.json").then((response) => response.json());
    const shader = await fetch("./shaders/pbpk_release_compute.wgsl").then((response) => response.text());
    const base = {
      schema: "sounio.immersive_dissertation.webgpu_pbpk_kernel_runtime_report.v1",
      status: "not_available",
      contractStatus: contract.status,
      claimBoundary: contract.claim_boundary,
      shaderPath: contract.shader.path,
      probe: {
        navigatorGpu: Boolean(navigator.gpu),
        adapterAvailable: false,
        deviceAvailable: false,
        deviceLostHandlerRegistered: false,
        adapterInfo: null,
        deviceLimits: null,
      },
      kernel: { workgroupSize: contract.shader.workgroup_size, inputCount: 0, outputs: [] },
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
    base.probe.deviceLimits = {
      maxBufferSize: device.limits.maxBufferSize,
      maxBindGroups: device.limits.maxBindGroups,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    };

    const shaderModule = device.createShaderModule({ code: shader });
    if (typeof shaderModule.getCompilationInfo === "function") {
      const info = await shaderModule.getCompilationInfo();
      base.kernel.compilationMessages = info.messages.map((message) => ({
        type: message.type,
        message: message.message,
        lineNum: message.lineNum,
        linePos: message.linePos,
      }));
      if (base.kernel.compilationMessages.some((message) => message.type === "error")) {
        base.status = "shader_compilation_failed";
        device.destroy();
        return base;
      }
    }

    const inputs = [
      [1.0, 6.0, 0.45, 0.12, 0.06, 1.0, 150.0, 260.0],
      [5.5, 6.0, 0.45, 0.12, 0.06, 1.0, 150.0, 260.0],
      [9.0, 6.0, 0.45, 0.12, 0.06, 1.0, 150.0, 260.0],
      [18.0, 6.0, 0.45, 0.12, 0.06, 1.0, 150.0, 260.0],
    ];
    const flat = new Float32Array(inputs.flat());
    const inputBuffer = device.createBuffer({ size: flat.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outputBuffer = device.createBuffer({
      size: inputs.length * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      size: inputs.length * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(inputBuffer, 0, flat);
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: contract.shader.entry_point },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(inputs.length / contract.shader.workgroup_size));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, inputs.length * 4 * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    base.kernel.inputCount = inputs.length;
    base.kernel.outputs = inputs.map((input, index) => ({
      time_h: input[0],
      release_fraction: raw[index * 4],
      parent_ng_ml: raw[index * 4 + 1],
      odv_ng_ml: raw[index * 4 + 2],
      odv_parent_ratio: raw[index * 4 + 3],
    }));
    inputBuffer.destroy();
    outputBuffer.destroy();
    readbackBuffer.destroy();
    device.destroy();
    base.status = "pass";
    return base;
  });

  report.url = args.url;
  report.browser = args.browserName;
  report.warnings = warnings;
  report.errors = errors;
  report.validationErrors = validateKernelReport(report);
  if (report.validationErrors.length && report.status === "pass") report.status = "validation_failed";
  await writeReport(args.output, report);
  await browser.close();

  const ready = report.probe?.navigatorGpu && report.probe?.adapterAvailable && report.probe?.deviceAvailable;
  if (!ready) {
    console.log("WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE");
    console.log(JSON.stringify(report, null, 2));
    process.exit(args.requireWebgpu ? 1 : 0);
  }
  if (errors.length || report.validationErrors.length || report.status !== "pass") {
    console.error("WEBGPU_PBPK_KERNEL_RUNTIME_FAIL");
    console.error(JSON.stringify(report, null, 2));
    process.exit(1);
  }
  console.log("WEBGPU_PBPK_KERNEL_RUNTIME_PASS");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error("WEBGPU_PBPK_KERNEL_RUNTIME_FAIL");
  console.error(error.message);
  process.exit(1);
});
