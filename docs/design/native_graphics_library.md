<!-- docs:meta
topic_id: repo.docs.design.native-graphics-library
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.design.native-graphics-library
-->

# Design: Native Graphics Library for Sounio ("Biblioteca Gráfica Nativa")

**Status:** Draft — In Planning  
**Owner:** Grok (with user review)  
**Date:** May 2026  
**Related Branch:** `sounio-pure/python-extermination-phase7`
**Integration Map:** `docs/design/native_graphics_integration_map.md`

## 0. Implementation Sync — Grok + Kimi + Codex

This document remains the architecture brief owned by Grok. The live scaffold
has now moved from plan to working Sounio modules and should be read as a
three-lane integration:

- **Grok / architecture:** this design file defines the native graphics target:
  pixel graphics, epistemic visualization, PNG/SVG export, tiling, and eventual
  RenderTarget-style unification.
- **Kimi / scaffold:** the current `stdlib/graphics/*` surface provides the
  base implementation: `drawing`, `surface`, `plot`, `scatter`, `heatmap`,
  `epistemic`, `svg`, `png`, `tile`, `tiled_plot`, `plot_png`, `animate`, and
  `view`.
- **Codex / quality companion:** `stdlib/graphics/quality.sio` adds the
  publication-oriented scientific layer: antialiased line and heatmap paths,
  contour, colorbar, uncertainty band, markers, error bars, histogram, ECDF,
  boxplot, and violin density.

Validation is split deliberately:

- `scripts/ci/graphics_scaffold_gate.sh` is the owner gate for the base scaffold
  and now also checks `graphics::quality` when that module exists.
- `scripts/ci/graphics_companion_gate.sh` is the broader integrated graphics
  gate. It runs the scaffold gate first, then exercises companion, export,
  raster, quality, SVG, tile, PNG, and tiled-export smokes.

Current architectural truth as of this sync:

- The original FFI-first PNG recommendation has been superseded in the live
  scaffold by a **pure-Sounio PNG encoder using stored DEFLATE blocks**. This
  preserves the purity goal earlier than planned, at the cost of larger files.
- The tiling strategy from section 4.3 is implemented as `TiledSurface` with a
  current 3x3 tile limit.
- The SOTA visualization lane should continue as layered additions over
  `graphics::quality`, while keeping `graphics::plot` as the compact
  compatibility layer.

---

## 1. Executive Summary

Sounio currently lacks a first-class, self-contained way to produce publication-quality raster graphics (PNG) and advanced scientific visualizations directly from Sounio code.

The existing `stdlib/plot/` stack is excellent for terminal/Unicode visualization and has good SVG export, but raster output still depends on external tools or Python (matplotlib). The lower-level `stdlib/render/` rasterizer is capable but not integrated with high-level scientific plotting.

This design proposes a **native graphics library** that:

- Enables direct, high-quality PNG (and improved SVG) output from Sounio without Python.
- Treats epistemic visualization (`Knowledge<T>`, uncertainty, confidence, provenance) as a first-class citizen.
- Unifies the current character-based and pixel-based worlds under a clean architecture.
- Provides a realistic path to eliminate Python figure generation in the dissertation workflow (especially `bbb_dissertation_figures.py` and similar).

**Primary recommendation (MVP):** Adopt a minimal, well-isolated FFI to `stb_image_write` for PNG output in Phase 1, while designing the architecture to allow a pure-Sounio path later.

---

## 2. Current State & Gap Analysis

### Existing Components

- **`stdlib/plot/`**: Character-based canvas with Unicode blocks + Braille, line/scatter/bar/heatmap plots, basic epistemic error bars, SVG export (via external conversion for PNG/PDF).
- **`stdlib/render/`**: Software triangle rasterizer with barycentric interpolation, z-buffer, color interpolation, multiple backends (PPM primary, SPIR-V/Metal planned).
- **`stdlib/image/`**: Very basic `Image` type (256×256 max) + FFI stubs that currently return `false`.
- **`stdlib/compress/`**: `deflate.sio` only implements stored blocks (BTYPE=00) — no real compression.
- **GPU path**: kretikos exists for PTX/Metal/SPIR-V but is not connected to the plotting stack.

### The Core Problem

There are **two disconnected worlds**:

1. High-level scientific plotting lives in the character canvas world.
2. Real pixel rendering lives in the rasterizer world.

Publication-quality figures for the dissertation (high DPI, nice styling, epistemic elements) currently require Python + matplotlib because neither world alone is sufficient.

---

## 3. Vision, Goals and Scope (Phased)

(See approved plan for full goals — summarized here for the document)

**MVP Goal**: A usable path to generate dissertation-grade figures (PNG + epistemic visualization) directly from Sounio, enabling significant reduction of Python in the dissertation pipeline.

**Explicitly Deferred**: Full windowing/GUI, production typography, photorealistic 3D, replacing every matplotlib use in the repo.

---

## 4. Key Tensions and Trade-off Analysis

### 4.1 PNG Output Path — Detailed Comparative Analysis

**Options evaluated:**

- **Pure full DEFLATE in Sounio** (LZ77 + Huffman)
  - Effort: Very High (1200–2000+ LOC, complex testing)
  - File size: Good
  - Reproducibility: High risk of non-determinism
  - Recommendation: **Fase 3** (long-term purity goal)

- **PNG with stored blocks only** (current deflate)
  - Effort: Medium
  - File size: Poor (often 5–15× larger)
  - Recommendation: Acceptable as temporary prototype only

- **Minimal FFI to `stb_image_write`** (header-only C library)
  - Effort: Low-Medium
  - File size: Good
  - Integration: Requires careful FFI design and build support
  - **Recommended for Phase 1 (MVP)**

- **External lightweight conversion** (current SVG path)
  - Effort: None (already exists)
  - Recommendation: Keep for vector, deprecate as primary raster path for dissertation

**Decision**: Proceed with minimal `stb_image_write` FFI for MVP, with clear isolation and a documented path toward pure implementation.

### 4.2 Unification of Character-based and Pixel-based Worlds (Deepened)

**Problema de raiz**:  
Hoje temos dois sistemas de desenho quase ortogonais:

- `plot::canvas` → grade de caracteres + Unicode (braille, blocos, etc.). Fácil de usar, mas limitado em qualidade vetorial/raster.
- `render::rasterizer` + `Framebuffer` → rasterizador real de triângulos com z-buffer e interpolação. Mais poderoso, mas sem API de alto nível para plots científicos.

**Modelo recomendado (com mais detalhe)**: **RenderTarget + DrawingCommand**

Proposta de abstração:

```sounio
// drawing.sio (nova camada intermediária)

enum DrawingCommand {
    Line { p1: Point2D, p2: Point2D, style: LineStyle },
    Polygon { points: [Point2D; N], fill: FillStyle, stroke: StrokeStyle },
    Text { pos: Point2D, text: string, style: TextStyle },
    EpistemicBand { 
        center: [Point2D; N], 
        lower: [Point2D; N], 
        upper: [Point2D; N], 
        style: BandStyle 
    },
    // ... outros comandos
}

trait RenderTarget {
    fn submit(self: &!Self, cmd: DrawingCommand) with Mut;
    fn set_transform(self: &!Self, transform: Transform2D) with Mut;
    fn finish(self) -> OutputHandle with IO, Mut;   // PNG, SVG, terminal buffer, etc.
}
```

**Vantagens deste modelo**:
- Os módulos de alto nível (`plot::line`, `plot::epistemic`, `plot::scatter`) só geram comandos.
- Backends podem ser adicionados independentemente:
  - `RasterTarget` (usa o rasterizador atual de triângulos)
  - `SVGTarget` (evolui o export atual)
  - `TerminalTarget` (degrada comandos vetoriais para caracteres/Unicode)
  - Futuro: `GPU Target` via kretikos

**Desafios em Sounio**:
- Como representar `enum DrawingCommand` com dados de tamanho variável (hoje usamos arrays fixos).
- Como passar estilos complexos sem closures.
- Solução pragmática inicial: usar structs grandes com flags ou arrays de tamanho razoável.

**Alternativa considerada**: "Dual Path" (manter plot/ separado e criar `plot::raster` paralelo).  
Decisão: **Modelo de abstração é preferível** para longo prazo, mas podemos começar com uma versão simplificada (só 2-3 comandos principais) para o MVP.

### 4.3 Estratégias de Resolução e Memória (Deepened)

Limite atual: `Framebuffer` e rasterizador usam arrays de 65536 elementos → **256×256 pixels máximo prático**.

**Análise das estratégias**:

1. **Tiling (Recomendada para Fase 1-2)**
   - Renderizar em tiles de 256×256 ou 512×512.
   - Camada `TiledSurface` gerencia composição, z-buffer global e bordas.
   - Vantagens: mantém código do rasterizador quase intacto, escala para 4K, facilita GPU por tile.
   - Desafios: anti-aliasing e stroking entre tiles.

2. **Scanline Streaming**
   - Renderizar linha por linha, enviando direto para o encoder.
   - Excelente para memória, mas exige reescrita significativa do rasterizador atual (que é bounding-box based).

3. **Aumento de arrays fixos + careful management**
   - Criar `FramebufferHD { [i64; 2_073_600] }` para 1920×1080.
   - Simples, mas incha o binário e não escala indefinidamente.

**Decisão atual**:
Começar com **TiledSurface** como abstração principal. O rasterizador baixo nível continua com tamanho pequeno. O encoder PNG recebe tiles e compõe.

Isso também prepara o terreno para um backend GPU futuro (cada tile pode ser um dispatch independente).

### 4.4 Linguagem Visual Epistêmica de Primeira Classe (Deepened)

Além de barras de erro simples, propomos que a biblioteca ofereça primitivas visuais que explorem o que Sounio tem de único:

**Visualizações concretas que queremos suportar bem:**

- **Confidence Decay along a curve**  
  A opacidade ou saturação da banda diminui conforme a confiança decai ao longo de operações (ex.: simulação PBPK ao longo do tempo).

- **Multi-source Uncertainty**  
  Diferentes padrões (hachura, pontilhado, linhas diagonais) para diferentes fontes de incerteza (parâmetro vs modelo vs medição).

- **Provenance-linked elements**  
  Pequenos ícones ou marcadores que conectam visualmente uma curva a sua origem epistêmica.

- **Gate Visualization**  
  Regiões do gráfico marcadas quando uma predição cruza um `Confidence(N)` threshold.

- **Second-order / P-box rendering**  
  Áreas entre lower e upper bounds de distribuições de segundo ordem.

**Exemplo de uso desejado na dissertação:**

```sounio
use plot::epistemic::{epistemic_time_series, ConfidenceBandStyle};

let series = EpistemicTimeSeries { ... }; // com Knowledge ou Epistemic values

epistemic_time_series(
    &mut ctx,
    &series,
    EpistemicPlotConfig {
        show_confidence_decay: true,
        uncertainty_style: UncertaintyStyle::MultiSource,
        ...
    }
);
```

Isso permite que as figuras da dissertação mostrem não só "valor ± incerteza", mas **por que** a incerteza existe e como ela evolui — algo difícil de fazer de forma natural em matplotlib.

---

## 5. Recommended Architecture

**Modelo em camadas proposto:**

1. **Drawing Primitives** (`drawing/`)
   - `Point2D`, `Path`, `StrokeStyle`, `FillStyle`, `Transform2D`
   - `DrawingCommand` enum

2. **Scientific Plot Types** (`plot/`)
   - `line`, `scatter`, `bar`, `heatmap`, `epistemic`
   - Esses módulos geram DrawingCommands

3. **Epistemic Extensions** (`plot/epistemic`)
   - Tipos e funções específicas para `Knowledge<T>`, decaimento de confiança, multi-source uncertainty

4. **Render Targets** (`render/targets/`)
   - `RasterTarget` (usa o rasterizador + future tiled surface)
   - `SVGTarget`
   - `TerminalTarget` (caractere)

5. **Image / Export** (`image/`, `plot/export`)
   - PNG (via FFI inicial), SVG aprimorado, PPM

**Evolução do código existente**:
- `plot/canvas.sio` e `plot/line.sio` etc. serão gradualmente reescritos para emitirem DrawingCommands em vez de desenharem direto no Canvas.
- O Canvas atual vira um dos backends (`TerminalTarget`).
- O `render/rasterizer` é preservado e encapsulado pelo `RasterTarget`.

---

## 6. API Ergonomics & Module Structure (Rascunho Inicial)

```sounio
// Uso típico desejado

use plot::{line, epistemic};
use plot::raster::RasterTarget;
use image::png;

fn main() with IO, Mut {
    let mut target = RasterTarget::new(1200, 800);
    
    let data = ...; // Knowledge ou f64 + uncertainty
    
    epistemic::time_series(&mut target, &data, EpistemicConfig { ... });
    
    let png = target.finish().to_png();   // ou via FFI por enquanto
    // salvar png
}
```

Módulos públicos sugeridos:
- `plot::line`
- `plot::epistemic`
- `plot::raster` (backend)
- `plot::svg`
- `drawing` (primitivas de baixo nível, para usuários avançados)

---

(Seções 7 a 11 serão preenchidas na próxima iteração: GPU strategy, Roadmap detalhado, Risks, Verification, Appendices com mais sketches.)

**Status do documento**: Estrutura sólida + tensões aprofundadas. Arquitetura e API em desenvolvimento.

---

**O que foi feito agora (1-2)**:
- Continuei preenchendo o documento (seções de Arquitetura e API).
- Aprofundei significativamente as tensões 4.2 (Unification), 4.3 (Resolução) e 4.4 (Linguagem Epistêmica), com sketches de código e exemplos concretos.

Quer que eu continue preenchendo as seções seguintes (especialmente Roadmap + Risks + Verification) ou prefere que eu aprofunde ainda mais em algum ponto específico (ex.: mais detalhes do FFI para PNG, esboço completo do `DrawingCommand` enum, exemplos visuais da dissertação, etc.)?

- Keep the existing rasterizer and framebuffer working with 256×256 or 512×512 tiles.
- Add a higher-level `Surface` / `TiledImage` that manages tiles, global z-buffer, and final composition before PNG encoding.
- This scales to 1920×1080+ without changing the low-level rasterizer.

Alternative strategies (scanline streaming, direct-to-encoder) are noted as higher-complexity options for later phases.

### 4.4 Linguagem Visual Epistêmica de Primeira Classe

Beyond simple error bars, the library should support:

- Confidence decay visualization (opacity or color intensity along a curve)
- Multi-source uncertainty (different patterns/textures for different uncertainty origins)
- Provenance-linked glyphs
- Visual representation of compile-time confidence gates
- Second-order uncertainty (p-box style areas)

Concrete API direction and visual examples will be expanded in later sections.

---

## 5–11. Remaining Sections (To Be Populated)

The following sections will be completed in the next iteration:

- **5. Recommended Architecture** (layers, core types, evolution of existing modules)
- **6. API Ergonomics & Module Structure**
- **7. GPU Path Integration Strategy**
- **8. Phased Roadmap & Milestones**
- **9. Risks, Constraints & Open Questions**
- **10. Verification & Success Criteria**
- **11. Appendices** (code snippets, draft APIs, references)

---

**Document Status**: Structure defined and approved. Content population in progress.

*This document follows the structure agreed in the planning session.*
