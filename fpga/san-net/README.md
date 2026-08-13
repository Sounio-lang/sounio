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

## Segunda rodada: `sendmmsg` removido, throughput é INTERMITENTE (2026-08-13, tarde)

Depois da recuperação acima, reescrevi `inject_san.c` para nunca mais usar
`sendmmsg` — só `sendto`, um pacote por chamada de sistema, sempre, com
*pacing* opcional por atraso entre chamadas (token bucket). Confirmado
correto localmente (`pace_test.py`) antes de tocar na FPGA.

**Não existe um teto limpo dependente de taxa.** O que a varredura mostrou:

| tentativa | taxa | resultado |
|---|---|---|
| `sendto` sem pace (ritmo natural) | 12,6–13,4 Gbit/s | bit-exato, 3 vezes seguidas |
| `sendto` sem pace (ritmo natural, variou) | 17,2 Gbit/s | **travou** |
| `sendto` com pace | 8,0 Gbit/s | bit-exato, 2 vezes seguidas |
| `sendto` com pace | 12,0 Gbit/s | bit-exato, 1 vez |
| `sendto` com pace | **8,0 Gbit/s de novo** | **travou** |

A mesma taxa (8 Gbit/s), com o mesmo código, a mesma coorte, o mesmo padrão
de envio (um pacote por syscall, sem lote): passou duas vezes e falhou na
terceira. Isso não é um limiar determinístico de banda — é um comportamento
**intermitente** do `networklayer` do VNx (ou de algo na cadeia CMAC →
networklayer → kernel) que uma taxa mais baixa não elimina, só reduz a
frequência.

**Conclusão honesta:** não há um número de throughput sustentável e
reprodutível para reportar. O que existe é:
- correção bit-exata, robusta, reproduzida repetidas vezes em 1000 e em
  4.000.003 amostras — isso é o resultado publicável
- uma taxa de transmissão que *às vezes* funciona até ~13 Gbit/s e falha
  de forma imprevisível mesmo bem abaixo disso
- uma recuperação sempre efetiva (`systemctl restart u250-vnx.service` +
  reprogramação), usada ~6 vezes nesta sessão sem exceção

**Não investigado:** a causa raiz da intermitência. Candidatos plausíveis,
nenhum confirmado: efeito cumulativo de reprogramações repetidas na mesma
sessão sem power-cycle real da placa; estado da tabela ARP/socket do
networklayer degradando com uso; alguma condição de corrida no próprio
`networklayer` de terceiros (VNx) que só aparece sob re-uso intenso. Medir
isso exigiria uma bancada dedicada (não compartilhada) e provavelmente
capturas de tráfego mais longas — fora do escopo de hoje.

## Terceira rodada: a causa era ARP indo STALE, não o networklayer (2026-08-13, final)

`ip neigh show 10.100.100.50` no injetor mostrava a entrada em **STALE**
depois de qualquer hiato (editar um script, rodar `git commit`, etc.). Teste
direto confirmou: `ping -c1` isolado, um por vez com 1s de intervalo, perdeu
1 em cada poucos; um stream contínuo de 40 pings a 0,3s não perdeu nenhum.

Isso bate exatamente com o padrão de falha observado: as três corridas que
passaram seguidas (8, 8, 12 Gbit/s) ficaram próximas no tempo; a que falhou
veio depois de eu editar e reimplantar `run_measurement.sh` — minutos de
hiato no meio, tempo suficiente para a entrada ARP envelhecer.

**Correção:** `run_measurement.sh` agora manda 3 pings (`ping -c3 -i0.2`)
para a FPGA imediatamente antes da rajada de UDP. O ICMP echo-reply confirma
o vizinho dos dois lados (kernel do injetor e `arp_server` da FPGA) antes do
tráfego de dados começar.

**Efeito colateral achado no caminho:** a primeira tentativa com o
aquecimento falhou — mas não por causa do ARP. O script antigo usava
`set -euo pipefail`; quando um passo falhava depois de armar, o `ctl_san`
já lançado em segundo plano ficava **órfão**, segurando o contexto do
dispositivo (`xrt::device`) aberto. A tentativa seguinte falhava com
`ERRO: failed to open cu context: Invalid argument` — um sintoma totalmente
diferente que mascarava a causa raiz. Corrigido com `trap` que mata o
`ctl_san` armado se qualquer passo posterior falhar.

**Confirmado 3× seguidas** com o aquecimento de ARP no lugar, taxa de
8 Gbit/s, todas bit-exatas, zero perda real (`eth_in` delta ~7150-7190
contra 7143 pacotes enviados, dentro do ruído de fundo). Ainda não é uma
prova de que a intermitência acabou de vez — só três corridas — mas a causa
agora tem um mecanismo plausível e testável, não é mais um mistério.
