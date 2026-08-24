# Fronteira: Ancoragem lógica (EL) do oráculo de conflitos em alinhamentos

**Slug:** `el-grounding`
**Status:** em ataque
**Data:** 2026-08-02

## O problema aberto

Na fronteira `epistemic-alignment-repair`, o reparo epistêmico de
alinhamentos recebia os conflitos entre mappings de um **oráculo
codificado à mão**: uma função `conflicts(a, b)` que listava
explicitamente os pares de mapping ids incompatíveis ({m0-m1}, {m2-m3}).
Isso deixa aberta a pergunta central: **de onde vêm os conflitos?** Sem
uma derivação lógica, o reparo — com todas as suas garantias formais de
consistência, maximalidade e propagação de confiança — repousa sobre uma
relação de conflito que pode estar errada ou incompleta. O problema
aberto é *ancorar* (ground) o oráculo de conflitos na semântica da
própria ontologia: derivar incompatibilidades de axiomas de subsunção e
disjunção, com prova formal de que a derivação é correta.

## Evidência na literatura

1. **Bayoudhi, Sassi, Jaziri (2018).** *Expert Systems*, DOI:
   `10.1111/exsy.12355`.
   - Abordagens de correspondência (matching) entre ontologias que
     exploram a estrutura lógica das ontologias — e não apenas
     similaridade lexical — como fundamento para decidir correspondências;
     reforça a tese de que a qualidade do alinhamento depende de ancorar
     as decisões na estrutura formal das ontologias envolvidas.

2. **Jiménez-Ruiz, Cuenca Grau, Horrocks (2011).** "Logic-based
   assessment of the compatibility of UMLS ontology sources." *Journal of
   Biomedical Semantics* 2(Suppl 1):S2. DOI: `10.1186/2041-1480-2-s1-s2`.
   - Mostra que fontes do UMLS (SNOMED CT, FMA, NCI) contêm erros
     detectáveis *logicamente*: conflitos entre mappings podem e devem ser
     derivados da união lógica das ontologias alinhadas, não declarados
     ad hoc. É a âncora teórica direta desta fronteira — e também do
     cenário de reparo herdado da fronteira anterior.

## A aposta Sounio

Derivar o oráculo de conflitos de uma **mini-TBox EL**:
subsumpções `C ⊑ D` e disjunções `Disj(C, D)`. A regra de derivação é:

> mappings `a`, `b` que afirmam a **mesma entidade** sob classes `c_a` e
> `c_b` conflitam sse existem `d₁`, `d₂` com `c_a ⊑* d₁`, `c_b ⊑* d₂` e
> `Disj(d₁, d₂)`, onde `⊑*` é o fecho reflexivo-transitivo da subsunção.

O fecho é computado por um ponto fixo (laço `while`) sobre uma matriz
booleana 8×8 achatada. O que é novo: o oráculo deixa de ser uma entrada
não verificada e passa a ser um **teorema** — a correção da derivação
(fecho correto + regra de conflito correta) é provada pelo artefato
irmão `formal/OntologyELReasoner.lean`, escrito nesta mesma rodada por
um agente irmão. No protótipo, o oráculo derivado é confrontado com o
oráculo hardcoded original na instância compartilhada (mesmos 5 mappings,
confianças 0.30/0.06/0.95/0.40/0.80) e produz **exatamente** os conflitos
{0-1} e {2-3} e os mesmos sobreviventes {m0, m2, m4} após o reparo
guloso — equivalência empírica entre oráculo derivado e oráculo
hardcoded.

## Artefatos

- `el_conflict_demo.sio` — protótipo executável: mini-TBox EL com 8
  classes (lymphokine(0) ⊑ protein(1) ⊑ molecule(2); heart(3) ⊑
  organ(4); organ(4) disjunto de muscleonly(5); protein(1) disjunto de
  drugclass(6)), fecho de subsunção por ponto fixo, derivação dos pares
  de conflito, reparo epistêmico guloso com o oráculo derivado e
  verificação de equivalência com o cenário hardcoded original
  (imprime `ALL PASS`).
- `formal/OntologyELReasoner.lean` (agente irmão, mesma rodada) —
  formalização do raciocinador EL em miniatura e prova de que o fecho
  computado é correto e de que a regra de derivação de conflitos só
  produz pares genuinamente incompatíveis (soundness).

## Lacunas e riscos

- A TBox é proposicionalmente pequena (8 classes, 3 subsumpções, 2
  disjunções); EL completo inclui conjunção e restrições existenciais
  (`∃r.C`), ainda não modeladas — o algoritmo de classificação EL++
  (completion rules) é trabalho futuro.
- A equivalência oráculo-derivado ≡ oráculo-hardcoded é verificada apenas
  na instância compartilhada de 5 mappings; a prova Lean cobre a regra
  geral, mas a ponte mecanizada entre a regra provada e o protótipo `.sio`
  ainda é por inspeção/manual.
- Disjointness aqui é derivada apenas por alcance via fecho; propagação
  adicional de disjunção (ex.: se `Disj(X,Y)` e `Z ⊑ X` então
  `Disj(Z,Y)`) é consequência da regra de derivação, mas não é enumerada
  explicitamente como axioma saturado.
