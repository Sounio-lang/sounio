# Ataque à Conjecture 6.8 (Guterman–Zhilina, arXiv:2608.26903) — log 2026-08-31

**Alvo:** diam Γ_C^Z(𝕊) = 3 (provado 3 ≤ d ≤ 4 no paper; o 3 é conjecturado a partir
de experimentos Mathematica em ponto flutuante). Contexto e ranking:
`open_problems_scan_2026-08-31.md`.

## A redução (o que tornou a conjectura computável)

1. Comutação depende só da parte imaginária ⟹ vértices reduzem a ZDs puros.
2. Para ZD puro x: Im C(x) = ℝx ⊕ O(x), dim 5 (Lemma 6.2(2) + Moreno dim 4).
3. d=1 ⟺ [x,x′]=0 (exato); d=2 ⟺ (ℝx⊕O(x)) ∩ (ℝx′⊕O(x′)) ≠ 0 (posto exato);
   d=3 ⟺ o kernel racional exato do mapa bilinear T: U⊗V → Im 𝕊, (u,w) ↦ [u,w]
   contém um tensor real de posto 1 (k mᵀ). Contagem dimensional ingênua diz que
   genericamente NÃO deveria existir (dim ker ≥ 9 em ℝ⁵ˣ⁵; posto-1 tem codim
   demais) — quando existe sempre, é estrutura octoniônica, não acaso.
4. Um par sem tensor posto-1, com U∩V=0 e [x,x′]≠0, é contraexemplo (diam 4).

## Perna primária (Sounio sob Madaros, ADR-008)

`tests/run-pass/sedenion_conj68_basis_probe.sio` — canônico x₀=(e₁,e₂) vs swap
(e₂,e₁) + os 84 ZDs de base (aproveitando: no subgrafo de base, O(x′) = span dos
4 parceiros de base = vizinhos do double hexagon; VERIFICADO por aniquilação
bilateral exata i64, independência por posto de Bareiss inteiro).

Resultado (Madaros v0.80.0 COMMITTED, ELF md5=bf1fe608):

```
PARTNERS4 OK          # 84 ZDs de base, exatamente 4 parceiros cada (Descr. 4.18)
URANK5 OK             # dim(ℝx₀⊕O(x₀)) = 5 exato
D1 4  D2 37  D3W 43  D3U 0
SWAP_PROP67_AND_WITNESS OK   # Im C(x)∩Im C(x̃)=0 exato (Prop 6.7) + witness d=3
CONJ68_BASIS_SWEEP OK
```

**Todos os 85 alvos têm caminho de comprimento ≤ 3.** A conjectura sobrevive ao
sweep de base completo com certificados exatos nos degraus lineares.

## As três classes de witness descobertas (a estrutura do problema)

1. **Combinatória (mesma componente de Γ_O, paridade ímpar)** — e.g. o swap e
   (e₁,−e₂): o witness é um caminho puro de ortogonalidade dentro do double
   hexagon; aparece como fatia (i,j) inteiramente ZERO do tensor inteiro
   [uᵢ,wⱼ] — detectada exatamente, sem float. A iteração numérica cega tem
   bacia minúscula aqui (por isso os "unresolved" das primeiras rodadas).
2. **Contínua genérica** — witness (k,m) denso, achado pela iteração alternante
   (potência inversa em AᵀA 5×5) + polimento Gauss–Newton.
3. **Cruzada não-ortogonal (a família dura)** — pares como (e₁,e₂) vs (e₂,±e₃)
   (componente octoniônica compartilhada): o witness tem k₀ = m₀ = 0, i.e.
   u ∈ O(x₀), w ∈ O(x′) — ZDs de **componentes distintas de Γ_O** (invariantes
   e₃ vs ∓e₁) que **comutam sem serem ortogonais** (caso w = λu + q impossível
   de ortogonalidade: componentes distintas ⟹ uw ≠ 0). A busca z-parametrizada
   de witnesses ortogonais (caso-(i), u = √n(z)·x₀ + L(z) afim em z) devolve
   residual 2.6 — confirmando que NÃO há witness ortogonal para esses pares.
   Resolvido na perna souc com fase `lock0` (subespaço k₀=m₀=0 imposto
   exatamente via linhas-identidade nos sistemas normais).
   Nota: os witnesses de (e₂,+e₃) e (e₂,−e₃) são permutação/negação um do outro
   com o MESMO m — forte indício de família uniforme sob a simetria do hexágono.

## Oráculo diferencial (busca apenas; não define pass/fail — ADR-008)

`scripts/research/sedenion_commutativity_diameter_probe.py` — mesma redução em
Python/numpy; reproduziu D1/D2 idênticos e forneceu os witnesses numéricos da
classe 3 que orientaram o `lock0`. Convenção de multiplicação verificada
idêntica à Table 1 do paper e ao `cd_sigma` do repo (par canônico
(e₁+e₁₀)(e₇+e₁₂)=0 ✓).

## Próximos degraus

1. **Sweep racional geral** (além da base): amostragem densa de pares ZD
   racionais via b = a·t, t ⊥ 1,a com n(t) quadrado (parametrização completa da
   variedade ZD) — perna souc dedicada fora do run-pass (custo maior).
2. **Construção uniforme**: minerar a estrutura dos witnesses classe 3
   (mesma-m, permutação-k) → candidato a fórmula fechada u(x,x′), w(x,x′);
   se fechar, a prova da conjectura vira verificação de identidades algébricas.
3. **Lean**: identidades da construção uniforme sobre a recursão CD (pipeline
   EpistemicEffectsNS: `lean --threads=1`, sem `decide` em não-Decidable).
4. **Witnesses exatos da classe 3**: identificar algebricidade de k
   (0,−0.7794,−0.1183,0.3879,0.4776) — PSLQ/candidatos quadráticos.

## Risco de corrida

Conjectura dos próprios autores (grupo ativo, Moscou). Nosso diferencial:
certificados exatos nos degraus lineares + perna mecanizável em Lean. Mesmo se
a prova clássica deles sair primeiro, a infraestrutura ataca o alvo #2 do scan
(componentes de Γ_O(M₅), genuinamente aberto).
