# Sounio Handoff — 2026-03-08

## Branch
`codex/ci-signal-recovery-20260307`

## Objetivo atual (Sprint 37 — próximo)
Epistemic auto-viz: `Knowledge<T>` → renderização automática de incerteza.
Dual-path: compilador gera viz default + API manual explícita.

---

## O que foi concluído nesta sessão

### Sprint 34 — Render effect + tipos gráficos (COMPLETO — 34/34)
- `self-hosted/check/effects.sio`: Render effect adicionado (ID=12)
- `stdlib/render/types.sio`: Color, Vec2/3/4, Vertex, Triangle, Canvas
- `stdlib/render/scene.sio`: Camera, Light, Scene
- `stdlib/render/pipeline.sio`: RenderBackend enum, RenderConfig
- `stdlib/render/epistemic.sio`: UncertainVertex, ConfidenceBand, UncertaintyVizMode
- `examples/render/`: 5 exemplos de check
- Gate: `scripts/sprint34_render_types_gate.sh` → 34/34

### Sprint 35 — Software rasterizer + PPM output (COMPLETO — 25/25)
- `stdlib/render/framebuffer.sio`: Framebuffer struct, fb_new, fb_set_pixel, fb_emit_ppm
- `stdlib/render/rasterizer.sio`: Barycentric triangle fill, Bresenham line, z-buffer
- `examples/render/triangle_ppm.sio`: Triangle com interpolação de cor → PPM stdout
- `examples/render/uncertainty_ppm.sio`: Heatmap epistêmico 16x16 → PPM stdout
- Gate: `scripts/sprint35_rasterizer_gate.sh` → 25/25

### Sprint 36 — GPU render pipeline: SPIR-V/Metal/WGSL/PTX (COMPLETO — 37/37)
- `self-hosted/gpu/spirv_render.sio`: SPIR-V binary emitter, Vertex(0)+Fragment(3) shaders
- `self-hosted/gpu/metal_render.sio`: MSL text emitter, vertex_main + fragment_main
- `self-hosted/gpu/wgsl_render.sio`: WGSL text emitter, @vertex vs_main + @fragment fs_main
- `self-hosted/gpu/ptx_render.sio`: PTX kernel emitter, SM_75 pixel-fill com bounds check
- `self-hosted/gpu/spirv.sio`: Adicionados DECORATION_LOCATION=30, spv_emit_decorate_location, spv_emit_execution_mode
- Gate: `scripts/sprint36_gpu_render_gate.sh` → 37/37

---

## O que está em andamento

**Nada** — Sprint 36 commitado com sucesso. Working tree tem arquivos modificados não relacionados ao render pipeline (bootstrap/poseidon, alguns artifacts). Esses não foram tocados nesta sessão.

### Arquivos modificados (não desta sessão — NÃO commitar sem investigar):
```
M  bootstrap/poseidon/Makefile
M  bootstrap/poseidon/loader.c  vm.c  vm.h  main.c  loader.h
M  bootstrap/poseidon/tests/gen_test_soir.py  run_tests.sh
D  bootstrap/poseidon/tests/add.soir branch.soir call.soir loop.soir return.soir
?? bootstrap/poseidon/poseidon.c  poseidon.h
?? bootstrap/poseidon/rust/
?? bootstrap/poseidon/tests/fixtures/
M  self-hosted/check/refinement.sio
M  scripts/poseidon_gate.sh  poseidon_compat_matrix.txt
M  artifacts/omega/ sprint25/ sprint30/ sprint32/ sprint33/ sprint34/ sprint35/
```
Estes são de outra thread de trabalho (Poseidon VM + bootstrap). **Não reverter.**

---

## Próximos 3 passos exatos

### Passo 1: Sprint 37 — Epistemic auto-viz foundation
```bash
# Criar stdlib/render/epistemic_viz.sio com:
# - fn uncertainty_to_color(u: f64) -> Color  (heatmap azul→vermelho)
# - fn render_knowledge_heatmap(...)           (grid de pixels)
# - fn render_confidence_band(...)             (faixa de confiança)
target/debug/souc check stdlib/render/epistemic_viz.sio
```

### Passo 2: Sprint 37 — Exemplo epistêmico rodável
```bash
# examples/render/knowledge_uncertainty.sio
# Usa Knowledge<f64> simulado → heatmap → PPM stdout
target/debug/souc run examples/render/knowledge_uncertainty.sio > knowledge.ppm
```

### Passo 3: Sprint 37 — Gate + commit
```bash
bash scripts/sprint37_epistemic_viz_gate.sh   # alvo: ~20 checks
git add ...
git commit -m "feat(render): epistemic auto-viz — Knowledge<T> uncertainty rendering — Sprint 37"
```

---

## Blockers e Riscos

| Risco | Severidade | Mitigação |
|-------|-----------|-----------|
| Pinned binary não conhece `Render` effect | Baixo | Usar `with IO` nos exemplos run-pass |
| `Knowledge<T>` é generic — pinned binary pode não suportar | Médio | Usar f64 com campo `uncertainty: f64` explícito |
| spirv_render.sio usa `RenderSpvBuf` separado de `SpvBuf` | Info | Decisão deliberada para standalone check |
| bootstrap/poseidon modificados por outra thread | Info | Não reverter, não commitar junto |

---

## Arquivos principais desta sessão

```
self-hosted/gpu/spirv_render.sio    ← NOVO — SPIR-V vertex+fragment
self-hosted/gpu/metal_render.sio    ← NOVO — MSL vertex+fragment
self-hosted/gpu/wgsl_render.sio     ← NOVO — WGSL vs_main + fs_main
self-hosted/gpu/ptx_render.sio      ← NOVO — PTX fill kernel SM_75
self-hosted/gpu/spirv.sio           ← MODIFICADO — +3 funções render
stdlib/render/framebuffer.sio       ← Sprint 35
stdlib/render/rasterizer.sio        ← Sprint 35
stdlib/render/types.sio             ← Sprint 34
stdlib/render/scene.sio             ← Sprint 34
stdlib/render/pipeline.sio          ← Sprint 34
stdlib/render/epistemic.sio         ← Sprint 34 (base para Sprint 37)
examples/render/triangle_ppm.sio    ← Sprint 35 (roda: > triangle.ppm)
examples/render/uncertainty_ppm.sio ← Sprint 35 (roda: > uncertainty.ppm)
scripts/sprint36_gpu_render_gate.sh ← Gate Sprint 36
artifacts/sprint36/gpu_render_gate.v1.json
```

---

## Comandos para retomar

```bash
# 1. Entrar no repo
cd /home/demetrios/work/sounio
git status   # confirmar branch: codex/ci-signal-recovery-20260307

# 2. Verificar último commit
git log --oneline -3

# 3. Regredir todos os gates anteriores (sanity check)
bash scripts/sprint36_gpu_render_gate.sh
bash scripts/sprint35_rasterizer_gate.sh

# 4. Ver os renders rodando
target/debug/souc run examples/render/triangle_ppm.sio > /tmp/triangle.ppm
target/debug/souc run examples/render/uncertainty_ppm.sio > /tmp/uncertainty.ppm
# Abrir com: display /tmp/triangle.ppm  ou  eog /tmp/triangle.ppm

# 5. Começar Sprint 37
# Ver: stdlib/render/epistemic.sio (base já existente)
# Criar: stdlib/render/epistemic_viz.sio
```

---

## Testes rodados e resultado

| Gate | Resultado | Comando |
|------|-----------|---------|
| sprint34_render_types_gate.sh | 34/34 ✓ | `bash scripts/sprint34_render_types_gate.sh` |
| sprint35_rasterizer_gate.sh | 25/25 ✓ | `bash scripts/sprint35_rasterizer_gate.sh` |
| sprint36_gpu_render_gate.sh | 37/37 ✓ | `bash scripts/sprint36_gpu_render_gate.sh` |

---

## Arquitetura — Decisões relevantes

- **Render effect**: ID=12, em `self-hosted/check/effects.sio`
- **spirv_render.sio é standalone**: usa `RenderSpvBuf` (não `SpvBuf`) para evitar dependência de spirv.sio
- **metal_render.sio + wgsl_render.sio + ptx_render.sio**: usam `str_len` / `str_char_at` (builtins Sounio) para push de strings
- **Framebuffer**: 65536 pixels max (256×256) no stdlib; 16384 (128×128) nos exemplos (VM limit)
- **Pinned binary** (`target/debug/souc`): não reconhece `Render` effect nem generics epistêmicos — exemplos usam `with IO`
- **Namespace**: self-hosted é FLAT — sem `module`, sem `use`

---

## Processos longos rodando
Nenhum. Sem tmux sessions relevantes desta sessão.
