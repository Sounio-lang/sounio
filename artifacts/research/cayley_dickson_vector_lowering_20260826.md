# Multiplicacao de Cayley-Dickson: medida de codigo gerado e proposta de baixa vetorial

Data: 2026-08-26. Ambiente: Xeon Gold 6148 (AVX-512F/BW/CD/DQ/VL).

## Resumo

A multiplicacao de sedenion e uma **algebra de grupo torcida sobre Z2^4**, com sigma
como cociclo de ordem 2. Reescrita na forma canonica dessa estrutura, ela vira 16
reducoes horizontais sobre vetores de 16 lanes -- forma que o backend pode emitir
diretamente e que hoje nao emite.

Numeros medidos, nao estimados:

| | Rust/LLVM (`-C target-cpu=native`) | Madaros v0.80.0 (build 2026-08-26 22:55) |
|---|---|---|
| instrucoes | 477 (so `Sedenion::mul`) | 2221 (binario inteiro) |
| aritmetica FP | 32 `vmulps`, 29 `vaddps`, 29 `vsubps` | 15 `mulsd`, 12 `addsd` |
| embaralhamento | 154 (`vpermi2ps`/`vblendps`/`vshufps`) | 0 |
| registradores | 753 `ymm`, 205 `xmm`, **zero `zmm`** | 143 `xmm` |
| movimentacao | 79 `vmovaps` | 1162 `mov` |
| correcao | -- | `exit 228` = -28, correto |

Duas leituras:

O LLVM **nao chega a AVX-512**. A tese de "16 x f32 = 64 bytes = um registrador zmm,
100% de utilizacao" nao se realiza no codigo emitido: ele fica em 256 bits e gasta
154 das 477 instrucoes remexendo lanes, porque recebeu 256 termos colados e teve que
inferir a estrutura.

O Madaros emite **escalar correto**. Razao movimentacao/aritmetica de 43 para 1. O
codigo esta certo -- `exit 228` confere com r[0] = 2*(1 + 15*(-1)) = -28 -- mas nao
vetoriza.

## A estrutura

Definicao em `stdlib/algebra/cayley_dickson.sio`:

    r[i^j] += sigma(i,j) * a[i] * b[j]

Fixando o deslocamento `d = i^j`, o destino `i^j` **e o proprio d**. Logo todos os
pares de deslocamento d escrevem numa unica lane:

    r[d] = SOMA_i  sigma(i, i^d) * a[i] * b[i^d]

Verificado numericamente contra a definicao: erro maximo 1.78e-15.

Isso e uma reducao horizontal, nao uma acumulacao vertical. Por lane de saida:
uma permutacao XOR de `b`, uma aplicacao de sinal por vetor constante, uma
multiplicacao vetorial, uma reducao horizontal. Ordem de 7 instrucoes por lane,
~112 no total -- contra 477.

## Propriedades de sigma (calculadas, bits=4)

    sigma = -1 : 120 pares (46.9%)
    sigma = +1 : 136 pares (53.1%)
    sigma =  0 : 0 pares

**Sigma nunca zera.** Nao ha esparsidade a explorar -- os 256 termos sao todos
necessarios. (Registro: eu havia suposto o contrario; o calculo desmentiu.)

**Sigma nao e funcao de `i^j` apenas.** Se fosse, isso seria convolucao XOR pura e a
transformada de Walsh-Hadamard daria O(n log n). Nao e: e algebra de grupo *torcida*,
e o cociclo quebra a fatoracao.

**Sigma nao e separavel** -- `sigma(i,j) != f(i)*g(j)` em 120 dos 256 pares.

Se existe aceleracao tipo WHT para o caso torcido e questao matematica em aberto
aqui, nao afirmacao. O que esta verificado e a forma de reducao horizontal.

## Regularidade digna de nota

Vetores de sinal `SIGN_d[i] = sigma(i, i^d)`:

| d | SIGN_d (i = 0..15) | negativos |
|---|---|---|
| 0 | `+---------------` | 15 |
| 1 | `+++-+--++--+-++-` | 7 |
| 2 | `+-++++--++----++` | 7 |
| 3 | `++-++-+-+-+--+-+` | 7 |
| 4 | `+---++++++++----` | 7 |
| 5 | `++-+-+-++-+-+-+-` | 7 |
| 6 | `+++--++-+--++--+` | 7 |
| 7 | `+-++--++++--++--` | 7 |
| 8 | `+-------++++++++` | 7 |
| 9 | `++-+-++--+-+-++-` | 7 |
| 10 | `+++---++-++---++` | 7 |
| 11 | `+-++-+-+--++-+-+` | 7 |
| 12 | `+++++----++++---` | 7 |
| 13 | `+-+-+++---+-+++-` | 7 |
| 14 | `+--++-++---++-++` | 7 |
| 15 | `++--++-+-+--++-+` | 7 |

`d=0` tem 15 negativos -- e a norma: `e_i^2 = -1` para i>0.

**Todos os quinze `d` nao-triviais tem exatamente 7 negativos.** Sete e o numero de
retas do plano de Fano, e ha `stdlib/algebra/fano.sio` no repo. A conexao merece ser
investigada -- pode ser coincidencia numerica, pode ser a estrutura de fundo.

## Proposta

Reconhecer a forma no backend nao por casamento de padrao fragil, mas por **contrato
declarado no tipo**, na linha dos efeitos que a linguagem ja tem:

    fn cd_mul(a: CDElement, b: CDElement) -> CDElement
        with XorConvolution(bits = 4, cocycle = cd_sigma)

O contrato informa: destino indexado por `i^j`, coeficiente funcao pura de `(i,j)`.
Com isso o backend emite permutacao + sinal + reducao por construcao.

A generalizacao e o ganho: a mesma forma cobre `clifford.sio`, `fano.sio`,
`associator_field.sio`, Walsh-Hadamard (sigma == 1), convolucao XOR e FFT sobre GF(2).
Um mecanismo, a familia inteira.

Nenhum compilador de uso geral trata produto de algebra de grupo como construcao de
primeira classe -- todos esperam que o vetorizador redescubra a estrutura a partir de
aritmetica escalar. A medida acima mostra que ele redescobre mal.

## O que esta verificado e o que nao esta

Verificado: os numeros de instrucao dos dois compiladores; a equivalencia da forma de
reducao horizontal; as propriedades de sigma; a correcao do binario do Madaros.

Nao verificado: o custo real das ~112 instrucoes (e estimativa, nao medicao); se
`vpermps` cobre os 16 padroes de XOR ou se algum exige `vpermi2ps`; se ha aceleracao
sub-quadratica para o caso torcido.

## Reproducao

    ./bin/souc build probe.sio -o out          # NAO chamar o ELF cru
    MADAROS_RAW_BIN=<elf-novo> ./bin/souc build ...

`artifacts/self-hosted/madaros` estava em 5-ago enquanto builds novos saiam em
`.wt/*/bin/`. Conferir a data do binario antes de medir -- tres medicoes foram
descartadas nesta analise por esse motivo.

