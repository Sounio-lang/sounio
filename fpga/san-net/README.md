# SAN catastrophe-scan alimentado pela rede (Alveo U250)

Variante do kernel aceito (T3_GREEN) em que a coorte entra pela **fibra de
100G** em vez do DMA sobre PCIe. O host deixa de participar do caminho de
dados: ele carrega a LUT de custo, arma o kernel e lê os resultados, que são
pequenos.

**Baseline a bater:** 511 Msamples/s via DMA. A hipótese é ganho de
*latência* ponta a ponta, não de throughput — o throughput já era limitado
pelo barramento.

## Topologia

```
fibra 100G ─▶ cmac_1 ─▶ networklayer_1 ─▶ krnl_san_scan_net_1 ─▶ DDR ─▶ host
                            ▲
                       krnl_mm2s_1  (fecha o lado TX; o v++ exige toda
                                     porta de stream conectada)
```

`cmac_1`/`networklayer_1`, instância **1 e não 0**: medido pelo BAR PCIe, a
gaiola QSFP do `cmac_0` está sem cabo.

## Peças

| Arquivo | Onde roda | Papel |
|---|---|---|
| `krnl_san_scan_net.cpp` | FPGA | kernel; 4 mudanças sobre o aceito, entrada vira `hls::stream<ap_axiu<512,0,0,16>>` |
| `tb_san_scan_net.cpp` | host (csim) | golden model preservado; só a entrega mudou |
| `san_net.cfg` + `link_san_net.sh` | builder | composição e link v++ |
| `ctl_san.cpp` | **dl380** (XRT) | configura CMAC + tabela de sockets, arma o kernel, lê resultados |
| `inject_san.c` | outro nó do fabric | envia a coorte em UDP |
| `check_pack.c`, `loopback_test.py` | qualquer | provam o empacotamento |

## Empacotamento (idêntico ao artefato aceito, byte a byte)

- 1 amostra = 128 bits, `n_conf` campos Q0.15 de 15 bits, campo *k* em
  `[15k+14 : 15k]`
- 1 beat = 512 bits = 4 amostras

`check_pack.c` confere isso contra uma reimplementação em `uint64` (4000
casos, incluindo os campos que cruzam a fronteira de 64 bits);
`loopback_test.py` captura o datagrama de verdade na rede e confere campo a
campo contra um LCG reimplementado do zero.

## Ordem de execução

```bash
# 1. no dl380 — configura e arma; imprime a linha exata do passo 2
./ctl_san san_net.xclbin 10.100.100.2 50000 62781 7 100003

# 2. no nó injetor, com o ctl_san já armado
./inject_san 10.100.100.50 62781 100003 7 140 50000
```

A porta de origem do injetor é **fixa** e tem de casar com `theirPort` na
tabela de sockets — com porta efêmera o UDP do VNx descarta em silêncio.

## Duas armadilhas que custaram tempo

1. **Passo da tabela de sockets é 8 bytes, não 4** (Makefile do VNx:
   `OFFSET + 8*MAX_SOCKETS` entre colunas), embora cada campo tenha 4 bytes.
   Um passo de 4 escreve o socket 1 no padding do socket 0 e o tráfego some
   sem erro nenhum.
2. **Endereçar por nome, nunca por base deduzida.** `xrt::ip` resolve
   `cmac_1:cmac_1`; deduzir base custou horas configurando a gaiola sem cabo.

## Diagnóstico quando não chega nada

`ctl_san` lê os contadores do network layer no timeout:

- `eth_in` delta **0** → nenhum quadro chegou (cabo / VLAN / rota)
- `eth_in` delta **> 0** → chegou quadro e o UDP descartou (tabela de
  sockets / porta)

## Estado medido (2026-08-13)

Correção provada duas vezes, de forma limpa, em 4.000.003 amostras: zero
perda de pacote, histograma/catástrofes/FLOPs bit-exatos contra o golden
model. Isso é o resultado de arquitetura — publicável como está.

**Throughput ainda não medido corretamente.** `sendto` de thread única
satura em ~104 Msamples/s (limite do host, não da placa). Tentei elevar isso
com `sendmmsg` (lotes de 64 datagramas por chamada): a taxa de envio subiu
para 146,6 Msamples/s, mas **97,7% dos pacotes se perderam** — achado real:
o `networklayer` do VNx tem FIFOs pequenas e nenhum controle de fluxo, e uma
rajada nesse ritmo estoura o buffer.

### A rajada deixou a FPGA travada — `xrt-smi program` sozinho NÃO recupera

Depois da perda de pacotes, nenhuma corrida completava mais — nem revertendo
para `sendto`, nem com coortes de 8 amostras, mesmo após `xrt-smi program`
duas vezes. Isolado com cuidado antes de concluir: `tcpdump` no injetor
mostrou o pacote saindo certo (IP/porta corretos); a tabela de sockets da
FPGA conferia na leitura de volta; mesmo assim o kernel nunca recebia o
stream. O quadro chegava ao CMAC (contador sobia) mas nunca atravessava o
`networklayer` até o kernel — reprogramar só a região dinâmica não limpa
esse estado.

**Recuperação que funcionou:** `systemctl restart u250-vnx.service` (recarrega
o bitstream de BASE do VNx e refaz o bring-up do CMAC do zero), seguido de
`xrt-smi program --user san_net.xclbin` por cima. Confirmado por duas
corridas limpas depois disso (1000 e 4.000.003 amostras, ambas bit-exatas).

**Para a próxima tentativa de medir throughput real:** não repetir a rajada
sem controle. Um injetor com *pacing* (intervalo entre lotes, sweep
crescente) é o próximo passo — ainda não escrito.
