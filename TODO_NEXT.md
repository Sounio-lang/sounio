# TODO_NEXT — Próxima ação mínima executável

**Data:** 2026-03-08
**Branch:** `codex/ci-signal-recovery-20260307`
**Sprint:** 37

---

## ▶ AÇÃO AGORA

Criar `stdlib/render/epistemic_viz.sio`:

```bash
cd /home/demetrios/work/sounio

# 1. Criar o arquivo
cat stdlib/render/epistemic.sio   # ver base já existente

# 2. Implementar em stdlib/render/epistemic_viz.sio:
#    - fn uncertainty_to_color(u: f64, r: &!i64, g: &!i64, b: &!i64) with Mut
#    - fn render_knowledge_heatmap(values: ..., uncertainties: ..., fb: ...) with Mut, Panic, Div
#    - fn render_confidence_band(center_y: f64, sigma: f64, ...) with Mut, Panic, Div

# 3. Verificar
target/debug/souc check stdlib/render/epistemic_viz.sio

# 4. Criar examples/render/knowledge_uncertainty.sio e rodar
target/debug/souc run examples/render/knowledge_uncertainty.sio > /tmp/knowledge.ppm
```

---

## Contexto mínimo para retomar

- Sprints 34-36 completos e commitados
- Stack render: PPM software (S35) + SPIR-V/Metal/WGSL/PTX (S36)
- Próximo: epistemic rendering (S37) — `Knowledge<T>` → heatmap automático
- Gate alvo: `scripts/sprint37_epistemic_viz_gate.sh` ~20 checks
