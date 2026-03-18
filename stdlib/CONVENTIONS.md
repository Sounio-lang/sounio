# Convenções da Standard Library (stdlib) do Sounio

Este documento define os padrões **rigorosos** para o desenvolvimento e manutenção de toda a standard library do Sounio. Seu cumprimento é **obrigatório** para contribuições.

## 1. Nomenclatura

- **Funções e variáveis**: `snake_case`
  - Exemplo: `compute_mean`
- **Tipos/structs**: `PascalCase`
  - Exemplo: `EpistemicMatrix`
- **Constantes**: `SCREAMING_SNAKE_CASE`
  - Exemplo: `PI`

## 2. Estrutura de Módulos

Todo módulo `MODULO` em `stdlib/` deve ter exatamente esta estrutura:

- `mod.sio`: **Apenas** exports `pub use` e declarações `pub fn` (sem implementações).
- `lib.sio`: Implementações privadas (`fn` sem `pub`) + testes inline.
- `README.md`: Visão geral do módulo + exemplos executáveis simples.
- `EXAMPLES.md`: Casos de uso avançados com código compilável.
- `BENCHMARKS.md` (se aplicável): Metas de performance vs SciPy/NumPy, com benchmarks executáveis.

## 3. Documentação

Todo `pub fn` **deve** ter docstring `///` completa, incluindo:

* **Parâmetros**: Descrição detalhada de cada `param`.
* **Retorno**: Tipo exato, destacando `Knowledge<T>` se numérico.
* **Complexidade**: Notação O(?), ex: `O(n log n)`.
* **Referências bibliográficas**: DOIs, papers ou standards (NIST, GUM, etc.).

**Exemplos compiláveis** diretamente no docstring:

```sio
/// Calcula a média aritmética com propagação epistêmica.
///
/// # Parâmetros
/// - `data`: Vetor de medidas com incerteza.
///
/// # Retorno
/// `Knowledge<f64>` com média e confidence propagado.
///
/// # Complexidade
/// O(n)
///
/// # Exemplo
/// ```sio
/// let data = [knowledge(1.0, 0.01), knowledge(2.0, 0.02)];
/// let mean = compute_mean(data);
/// assert(mean.confidence > 0.95);
/// ```
///
/// # Referências
/// - GUM 2008, JCGM 100:2008
pub fn compute_mean(data: &[Knowledge<f64>]) -> Knowledge<f64>
```

## 4. Integração Epistêmica

- **Todo resultado numérico**: Retornar `Knowledge<T>` ou [`GUMUncertainty`](epistemic/gum.sio).
- **Propagação automática**: Incerteza deve propagar em todas ops matemáticas (`+`, `*`, etc.).
- **Confidence degradation**: Composições complexas degradam confidence automaticamente.

Exemplo mínimo em todo `pub fn` numérico:

```sio
pub fn add_epistemic(a: Knowledge<f64>, b: Knowledge<f64>) -> Knowledge<f64> {
    knowledge(a.value + b.value, sqrt(a.u**2 + b.u**2))
}
```

## 5. Testes

- **Unit tests**: Inline em `lib.sio`, usando `assert` e `assert_eq!`.
- **E2E tests**: Diretório `tests/stdlib/MODULO/` com cenários reais.
- **Validação vs referências**: Comparar resultados com SciPy, NIST datasets, etc.

Exemplo de test inline:

```sio
fn test_compute_mean() {
    let data = [knowledge(1.0, 0.0), knowledge(3.0, 0.0)];
    let mean = compute_mean(&data);
    assert_eq!(mean.value, 2.0);
    assert!(mean.confidence >= 1.0);
}
```
