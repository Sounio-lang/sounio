# Stats + Plot — demos de release

Programas Sounio puros (sem Python/matplotlib) para validar estatística e plotagem nativa.

## Compilar e correr

```bash
SOUC=./artifacts/self-hosted/souc-self-hosted-x86_64

$SOUC examples/stats_viz/stats_release_showcase.sio /tmp/showcase.elf
chmod +x /tmp/showcase.elf && /tmp/showcase.elf

$SOUC examples/stats_viz/regression_plot_demo.sio /tmp/reg_plot.elf
chmod +x /tmp/reg_plot.elf && /tmp/reg_plot.elf

$SOUC examples/stats_viz/descriptive_boxplot_demo.sio /tmp/hist.elf
chmod +x /tmp/hist.elf && /tmp/hist.elf

$SOUC examples/stats_viz/bootstrap_epistemic_lens_demo.sio /tmp/bca_lens.elf
chmod +x /tmp/bca_lens.elf && /tmp/bca_lens.elf > /tmp/sounio_bootstrap_band.svg
```

## Saídas

| Demo | Estatística | Plotagem | Ficheiro |
|------|-------------|----------|----------|
| `stats_release_showcase.sio` | Spearman ρ, Holm–Bonferroni, OLS | Banda epistémica + terminal `graphics::view` | `/tmp/sounio_stats_showcase_band.png` |
| `regression_plot_demo.sio` | Regressão linear (`stats::regression::linear`) | Banda epistémica + terminal | `/tmp/sounio_regression_band.png` |
| `descriptive_boxplot_demo.sio` | Média e desvio-padrão inline | Histograma raster PNG | `/tmp/sounio_descriptive_hist.png` |
| `bootstrap_epistemic_lens_demo.sio` | Bootstrap BCa (`stats::epistemic::bootstrap`) | Epistemic lens + SVG publicação | `/tmp/sounio_bootstrap_lens.png`, SVG em stdout |

## Módulos usados

- `stats::inferential`, `stats::multiple_testing`, `stats::regression::linear`, `stats::epistemic::bootstrap`
- `graphics::epistemic`, `graphics::view`, `graphics::export`, `graphics::surface`, `graphics::svg`

## Gotchas

- Usar `./artifacts/self-hosted/souc-self-hosted-x86_64` (o wrapper `bin/souc` pode estar desactualizado neste checkout).
- Indexar arrays com `i as usize`, não `i64` directo.
- Evitar `model.r2()` — o parser lê como `model.r`; usar `model.r_squared`.
- `model.predict()` pode falhar no bundle actual; usar `coefficients[0] + coefficients[1] * x`.
