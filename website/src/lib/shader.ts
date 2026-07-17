/**
 * shader.ts — minimal WebGL2 fragment-shader canvas runtime.
 *
 * Used by the landing hero (P2). Falls back gracefully:
 *  - no WebGL2 / context lost  -> `onFallback` (caller shows static brand art)
 *  - prefers-reduced-motion    -> renders one static frame, no loop
 *  - off-screen                -> loop paused via IntersectionObserver
 */

export interface ShaderUniforms {
  [name: string]: number | [number, number] | [number, number, number] | [number, number, number, number];
}

export interface ShaderCanvasOptions {
  frag: string;
  uniforms?: ShaderUniforms;
  /** devicePixelRatio cap for perf (default 1.5) */
  maxDpr?: number;
  onFallback?: () => void;
}

const VERT = `#version 300 es
layout(location=0) in vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }
`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error('[shader] compile error:', gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

export function createShaderCanvas(canvas: HTMLCanvasElement, opts: ShaderCanvasOptions): () => void {
  const fail = () => {
    canvas.dataset.shaderFailed = 'true';
    opts.onFallback?.();
  };

  const gl = canvas.getContext('webgl2', { antialias: false, alpha: true, powerPreference: 'low-power' });
  if (!gl) {
    fail();
    return () => {};
  }

  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, opts.frag);
  if (!vs || !fs) {
    fail();
    return () => {};
  }

  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('[shader] link error:', gl.getProgramInfoLog(prog));
    fail();
    return () => {};
  }
  gl.useProgram(prog);

  // Fullscreen triangle
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  const uTime = gl.getUniformLocation(prog, 'uTime');
  const uRes = gl.getUniformLocation(prog, 'uRes');
  const uScroll = gl.getUniformLocation(prog, 'uScroll');
  const customLocs = new Map<string, WebGLUniformLocation | null>();
  for (const name of Object.keys(opts.uniforms ?? {})) {
    customLocs.set(name, gl.getUniformLocation(prog, name));
  }

  const maxDpr = opts.maxDpr ?? 1.5;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      gl!.viewport(0, 0, w, h);
    }
  }

  function applyCustomUniforms() {
    for (const [name, value] of Object.entries(opts.uniforms ?? {})) {
      const loc = customLocs.get(name);
      if (loc == null) continue;
      if (typeof value === 'number') gl!.uniform1f(loc, value);
      else if (value.length === 2) gl!.uniform2f(loc, value[0], value[1]);
      else if (value.length === 3) gl!.uniform3f(loc, value[0], value[1], value[2]);
      else gl!.uniform4f(loc, value[0], value[1], value[2], value[3]);
    }
  }

  function draw(time: number) {
    resize();
    if (uTime) gl!.uniform1f(uTime, time);
    if (uRes) gl!.uniform2f(uRes, canvas.width, canvas.height);
    if (uScroll) {
      const rect = canvas.getBoundingClientRect();
      const progress = 1 - Math.min(1, Math.max(0, rect.top / Math.max(1, window.innerHeight)));
      gl!.uniform1f(uScroll, progress);
    }
    applyCustomUniforms();
    gl!.drawArrays(gl!.TRIANGLES, 0, 3);
  }

  let raf = 0;
  let visible = true;
  const start = performance.now();

  function loop() {
    raf = 0;
    if (!visible) return;
    draw((performance.now() - start) / 1000);
    raf = requestAnimationFrame(loop);
  }

  const io = new IntersectionObserver(
    (entries) => {
      visible = entries[0]?.isIntersecting ?? true;
      if (visible && !reduced && !raf) raf = requestAnimationFrame(loop);
    },
    { rootMargin: '80px' },
  );
  io.observe(canvas);

  const onLost = (e: Event) => {
    e.preventDefault();
    if (raf) cancelAnimationFrame(raf);
    fail();
  };
  canvas.addEventListener('webglcontextlost', onLost);

  if (reduced) {
    draw(0);
  } else {
    raf = requestAnimationFrame(loop);
  }

  return () => {
    io.disconnect();
    canvas.removeEventListener('webglcontextlost', onLost);
    if (raf) cancelAnimationFrame(raf);
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
  };
}
