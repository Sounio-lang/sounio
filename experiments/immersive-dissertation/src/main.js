const canvas = document.getElementById("scene-canvas");
const ctx = canvas.getContext("2d");
const curve = document.getElementById("curve-canvas");
const curveCtx = curve.getContext("2d");
const runtimeBadge = document.getElementById("runtime-badge");
const timeLabel = document.getElementById("time-label");
const phaseLabel = document.getElementById("phase-label");
const releaseLabel = document.getElementById("release-label");
const gpuKernelLabel = document.getElementById("gpu-kernel-label");
const modelKernel = document.getElementById("model-kernel");

const kernelContractUrl = new URL("../data/webgpu-pbpk-kernel-contract.json", import.meta.url);
const witnessDrug = "venlafaxine XR + O-desmethylvenlafaxine";

let contract;
let start = performance.now();
let rendererKind = "Canvas depth fallback";
let webgpuAvailable = false;
let webglAvailable = false;

init().catch((error) => {
  runtimeBadge.textContent = `renderer: unavailable (${error.message})`;
});

async function init() {
  contract = await fetch(kernelContractUrl).then((response) => response.json());
  webgpuAvailable = "gpu" in navigator;
  webglAvailable = canCreateWebgl();
  rendererKind = webgpuAvailable ? "Canvas proof with WebGPU-capable browser" : "Canvas depth fallback";
  runtimeBadge.textContent = runtimeContractLabel();
  gpuKernelLabel.textContent = webgpuAvailable ? "WGSL ready; hard proof required" : "WGSL ready; fallback proof only";
  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);
}

function canCreateWebgl() {
  const probe = document.createElement("canvas");
  try {
    return Boolean(probe.getContext("webgl2") || probe.getContext("webgl"));
  } catch {
    return false;
  }
}

function resize() {
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * scale));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function frame(now) {
  const elapsed = (now - start) / 1000;
  const t = (elapsed * 4.2) % 24;
  const state = modelAt(t);
  drawScene(t, state);
  drawCurve(t);
  updateKernel(t, state);
  timeLabel.textContent = `t = ${t.toFixed(1)} h`;
  phaseLabel.textContent = phaseAt(t);
  releaseLabel.textContent = state.release > 0.55 ? "diffusion tail" : "XR gel layer";
  requestAnimationFrame(frame);
}

function modelAt(t) {
  const release = Math.exp(-Math.pow((t - 5.5) / 6, 2));
  const parent = 150 * release * Math.exp(-0.12 * Math.max(0, t - 5.5));
  const odv = 260 * release * (1 - Math.exp(-0.18 * t)) * Math.exp(-0.06 * Math.max(0, t - 9));
  return {
    release,
    parent,
    odv,
    ratio: odv / Math.max(parent, 0.001),
  };
}

function phaseAt(t) {
  if (t < 2) return "hydration";
  if (t < 6) return "gel layer";
  if (t < 12) return "diffusion";
  return "ODV tail";
}

function drawScene(t, state) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createRadialGradient(w * 0.52, h * 0.42, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.72);
  bg.addColorStop(0, "rgba(45, 74, 84, 0.52)");
  bg.addColorStop(0.42, "rgba(22, 25, 34, 0.92)");
  bg.addColorStop(1, "rgba(6, 12, 15, 1)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const volumetric = ctx.createLinearGradient(0, h * 0.08, w, h * 0.82);
  volumetric.addColorStop(0, "rgba(136, 214, 194, 0.16)");
  volumetric.addColorStop(0.48, "rgba(255, 236, 205, 0.10)");
  volumetric.addColorStop(1, "rgba(234, 181, 111, 0.12)");
  ctx.fillStyle = volumetric;
  ctx.fillRect(0, 0, w, h);

  const focalGlow = ctx.createRadialGradient(w * 0.5, h * 0.52, 40, w * 0.5, h * 0.52, Math.max(w, h) * 0.42);
  focalGlow.addColorStop(0, "rgba(240, 248, 249, 0.18)");
  focalGlow.addColorStop(0.45, "rgba(136, 214, 194, 0.09)");
  focalGlow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = focalGlow;
  ctx.fillRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.52;
  const scale = Math.min(w, h) / 720;
  drawBody(cx, cy, scale, t, state);
  drawCapsule(cx - 210 * scale, cy + 132 * scale, scale, state);
  drawMolecule(cx + 205 * scale, cy - 20 * scale, scale, t, state);
  drawParticles(cx, cy, scale, t, state);
}

function drawBody(cx, cy, scale, t, state) {
  ctx.save();
  ctx.strokeStyle = "rgba(190, 219, 226, 0.18)";
  ctx.lineWidth = 2 * scale;
  ctx.beginPath();
  ctx.ellipse(cx, cy, 92 * scale, 230 * scale, 0, 0, Math.PI * 2);
  ctx.stroke();

  const organs = [
    ["blood", 0, -18, 52, "#9dd9de"],
    ["liver", -62, 56, 38, "#9ecb8d"],
    ["gut", 70, 92, 35, "#d7ae72"],
    ["brain", 0, -178, 36, "#b7a5e8"],
  ];
  for (const [, x, y, r, color] of organs) {
    const pulse = 1 + 0.045 * Math.sin(t * 0.8 + x);
    const g = ctx.createRadialGradient(cx + x * scale, cy + y * scale, 4, cx + x * scale, cy + y * scale, r * scale);
    g.addColorStop(0, "rgba(255,255,255,0.62)");
    g.addColorStop(0.32, `${color}cc`);
    g.addColorStop(1, "rgba(8,14,18,0.42)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(cx + x * scale, cy + y * scale, r * scale * pulse, r * 0.78 * scale, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = "rgba(123, 195, 170, 0.42)";
  ctx.lineWidth = 3 * scale;
  ctx.beginPath();
  ctx.moveTo(cx - 160 * scale, cy + 132 * scale);
  ctx.bezierCurveTo(cx - 84 * scale, cy + 80 * scale, cx + 6 * scale, cy + 80 * scale, cx + 62 * scale, cy + 92 * scale);
  ctx.bezierCurveTo(cx + 130 * scale, cy + 104 * scale, cx + 150 * scale, cy - 28 * scale, cx + 196 * scale, cy - 20 * scale);
  ctx.stroke();
  ctx.restore();
}

function drawCapsule(x, y, scale, state) {
  ctx.save();
  const gelRadius = (56 + state.release * 54) * scale;
  const gel = ctx.createRadialGradient(x, y, 4 * scale, x, y, gelRadius);
  gel.addColorStop(0, "rgba(232, 181, 111, 0.66)");
  gel.addColorStop(0.45, "rgba(255, 236, 205, 0.18)");
  gel.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = gel;
  ctx.beginPath();
  ctx.arc(x, y, gelRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(255,255,255,0.74)";
  ctx.lineWidth = 2 * scale;
  ctx.strokeRect(x - 62 * scale, y - 20 * scale, 124 * scale, 40 * scale);
  for (let i = 0; i < 9; i += 1) {
    ctx.fillStyle = i % 2 === 0 ? "#eab56f" : "#88d6c2";
    ctx.beginPath();
    ctx.arc(x - 42 * scale + i * 10 * scale, y + Math.sin(i) * 7 * scale, (4 + state.release * 2) * scale, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawMolecule(cx, cy, scale, t, state) {
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.strokeStyle = "rgba(220, 232, 238, 0.66)";
  ctx.lineWidth = 2 * scale;
  const atoms = [
    [0, 0, "#eab56f"],
    [28, -18, "#eab56f"],
    [58, 2, "#eab56f"],
    [86, -16, "#88d6c2"],
    [118, 4, "#88d6c2"],
  ];
  ctx.beginPath();
  atoms.forEach(([x, y], index) => {
    const px = cx + x * scale;
    const py = cy + (y + Math.sin(t * 0.7 + index) * 5) * scale;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  atoms.forEach(([x, y, color], index) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx + x * scale, cy + (y + Math.sin(t * 0.7 + index) * 5) * scale, 9 * scale, 0, Math.PI * 2);
    ctx.fill();
  });
  const glow = ctx.createRadialGradient(cx + 102 * scale, cy, 4 * scale, cx + 102 * scale, cy, (42 + state.ratio * 2) * scale);
  glow.addColorStop(0, "rgba(136,214,194,0.46)");
  glow.addColorStop(1, "rgba(136,214,194,0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx + 102 * scale, cy, (42 + Math.min(state.ratio, 2) * 10) * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawParticles(cx, cy, scale, t, state) {
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  for (let i = 0; i < 90; i += 1) {
    const f = i / 90;
    const x = cx - 180 * scale + f * 360 * scale + Math.sin(t * 0.3 + i) * 8 * scale;
    const y = cy + 118 * scale - state.release * 78 * scale + Math.cos(t * 0.5 + i) * 24 * scale;
    ctx.fillStyle = i % 3 === 0 ? "rgba(139,215,203,0.42)" : "rgba(232,181,111,0.28)";
    ctx.beginPath();
    ctx.arc(x, y, (1.8 + state.release * 1.8) * scale, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawCurve(tNow) {
  const w = curve.width;
  const h = curve.height;
  curveCtx.clearRect(0, 0, w, h);
  curveCtx.fillStyle = "rgba(5, 9, 13, 0.86)";
  curveCtx.fillRect(0, 0, w, h);
  curveCtx.strokeStyle = "rgba(255,255,255,0.14)";
  curveCtx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const y = 20 + i * 34;
    curveCtx.beginPath();
    curveCtx.moveTo(18, y);
    curveCtx.lineTo(w - 18, y);
    curveCtx.stroke();
  }
  drawOneCurve("parent", "#eab56f", (t) => modelAt(t).parent / 150);
  drawOneCurve("ODV", "#88d6c2", (t) => modelAt(t).odv / 260);
  const x = 18 + (tNow / 24) * (w - 42);
  curveCtx.strokeStyle = "rgba(255,255,255,0.8)";
  curveCtx.beginPath();
  curveCtx.moveTo(x, 16);
  curveCtx.lineTo(x, h - 18);
  curveCtx.stroke();
}

function drawOneCurve(label, color, fn) {
  const w = curve.width;
  const h = curve.height;
  curveCtx.strokeStyle = color;
  curveCtx.lineWidth = 2.4;
  curveCtx.beginPath();
  for (let i = 0; i < 96; i += 1) {
    const t = (i / 95) * 24;
    const x = 18 + (i / 95) * (w - 42);
    const y = h - 22 - Math.min(1, fn(t)) * (h - 44);
    if (i === 0) curveCtx.moveTo(x, y);
    else curveCtx.lineTo(x, y);
  }
  curveCtx.stroke();
  curveCtx.fillStyle = color;
  curveCtx.fillText(label, label === "parent" ? 26 : 92, 22);
}

function updateKernel(t, state) {
  const cells = [
    ["witness", witnessDrug, "input"],
    ["input", `ka_xr release ${Math.round(state.release * 100)}%`, "input"],
    ["CYP2D6", "NM conversion 1.00x", "conversion"],
    ["exposure", `parent ${state.parent.toFixed(1)} / ODV ${state.odv.toFixed(1)} ng/mL`, "exposure"],
    ["ODV/parent", state.ratio.toFixed(2), "conversion"],
    ["GPU kernel", webgpuKernelKernelValue(), "gpu"],
    ["firewall", "model curve only; observed C(t) blocked", "firewall"],
  ];
  modelKernel.innerHTML = cells.map(([label, value, stateName]) => `
    <div class="kernel-cell" data-state="${stateName}">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
}

function webgpuKernelKernelValue() {
  const size = contract?.shader?.workgroup_size || 64;
  return webgpuAvailable ? `2DGX ready; hard proof required; WGSL ${size} threads` : `2DGX ready; not promoted; WGSL ${size} threads`;
}

function runtimeContractLabel() {
  const webgpu = webgpuAvailable ? "WebGPU yes" : "WebGPU no";
  const webgl = webglAvailable ? "WebGL yes" : "WebGL no";
  return `renderer: ${rendererKind} | ${webgpu} | ${webgl}`;
}
