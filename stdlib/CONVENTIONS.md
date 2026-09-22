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
* **Retorno**: Tipo exato, destacando `Epistemic` se numérico.
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
/// `Epistemic` com média e confidence propagado (confidence é i64, 0..1000).
///
/// # Complexidade
/// O(n)
///
/// # Exemplo
/// ```sio
/// use epistemic::knowledge::{ep_measured, ep_is_credible}
/// let data = [ep_measured(1.0, 0.01), ep_measured(2.0, 0.02)]
/// let mean = compute_mean(&data)
/// assert(ep_is_credible(&mean, 900))
/// ```
///
/// # Referências
/// - GUM 2008, JCGM 100:2008
pub fn compute_mean(data: &[Epistemic]) -> Epistemic
```

## 4. Integração Epistêmica

- **Todo resultado numérico**: Retornar `Epistemic` (`stdlib/epistemic/knowledge.sio`) ou `GUMResult` ([`gum.sio`](epistemic/gum.sio)).
- **Propagação**: Incerteza propaga pelas funções livres `ep_add`, `ep_sub`, `ep_mul`, `ep_div` (método GUM delta, entradas não correlacionadas). Não há sobrecarga de `+`/`*` para `Epistemic` na superfície checada.
- **Confidence**: `i64` na escala 0..1000. Combinações preservam o mínimo das confidences de entrada — nunca aumenta.

Exemplo mínimo em todo `pub fn` numérico:

```sio
use epistemic::knowledge::{Epistemic, ep_add}

pub fn add_measurements(a: Epistemic, b: Epistemic) -> Epistemic {
    ep_add(&a, &b)
}
```

## 5. Testes

- **Unit tests**: Inline em `lib.sio`, usando `assert`.
- **E2E tests**: Diretório `tests/stdlib/MODULO/` com cenários reais.
- **Validação vs referências**: Comparar resultados com SciPy, NIST datasets, etc.

Exemplo de test inline:

```sio
use epistemic::knowledge::{ep_certain, ep_val, ep_confidence}

fn test_compute_mean() {
    let data = [ep_certain(1.0), ep_certain(3.0)]
    let mean = compute_mean(&data)
    assert(ep_val(&mean) == 2.0)
    assert(ep_confidence(&mean) == 1000)
}
```
