// ctl_san.cpp — host de controle da variante em rede do SAN scan (roda no dl380).
//
// Divisao de trabalho:
//   ctl_san    (dl380, XRT)  configura a FPGA, arma o kernel, le os resultados
//   inject_san (outro no)    manda a coorte pela fibra em UDP
//
// Enderecamento por NOME via xrt::ip — nunca por base address deduzida. Ja
// custou horas nesta placa configurar o cmac_0/networklayer_0, que sao a
// gaiola QSFP SEM cabo; o caminho vivo e o par _1.
//
// build e execucao (no dl380) — o setup.sh e' obrigatorio nos DOIS passos:
// sem ele o link acha o header mas o binario nao acha libxrt_coreutil.so.2.
//   source /opt/xilinx/xrt/setup.sh
//   g++ -std=c++17 -O2 -Wno-deprecated-declarations -o ctl_san ctl_san.cpp
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib
//       -lxrt_coreutil -lpthread -luuid
//
// cluster-ops 2026-08-13
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <arpa/inet.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
// xrt::ip — acesso a registrador por NOME de CU. Na XRT 2.23 instalada no
// dl380 este header vive sob experimental/, nao direto em xrt/.
#include "xrt/experimental/xrt_ip.h"

// ---- CMAC: offsets do kernel.xml do VNx (Ethernet/kernel.xml) ----
static const uint32_t CMAC_RESET      = 0x0004;
static const uint32_t CMAC_RX_STATUS  = 0x0204;
static const uint32_t CMAC_RSFEC_IND  = 0x1000;
static const uint32_t CMAC_RSFEC_EN   = 0x107C;

// ---- NetworkLayer: offsets do kernel.xml CONCRETO gerado com MAX_SOCKETS=16.
// A tabela de sockets tem passo de 8 BYTES por entrada (Makefile do VNx:
// OFFSET + 8*MAX_SOCKETS entre as colunas), embora cada campo tenha 4 bytes.
// Um passo de 4 escreveria o socket 1 no padding do socket 0 e o trafego
// sumiria em silencio.
static const uint32_t NL_MAC_L     = 0x0010;
static const uint32_t NL_MAC_H     = 0x0014;
static const uint32_t NL_IP        = 0x0018;
static const uint32_t NL_GW        = 0x001C;
static const uint32_t NL_NSOCK     = 0x0810;
static const uint32_t NL_THEIR_IP  = 0x0820;
static const uint32_t NL_THEIR_PT  = 0x08A0;
static const uint32_t NL_MY_PT     = 0x0920;
static const uint32_t NL_VALID     = 0x09A0;
static const uint32_t NL_SOCK_STEP = 8;
// contadores de pacote do network layer, para provar que os beats chegaram
static const uint32_t NL_ETH_IN    = 0x0410;
static const uint32_t NL_ARP_IN    = 0x0440;
static const uint32_t NL_ICMP_IN   = 0x0470;

// Tamanhos vindos do kernel (krnl_san_scan_net.cpp), nao inventados:
//   MAX_POINTS 8, HIST_BINS = MAX_POINTS, flop_t = ap_uint<64>
static const uint32_t MAX_POINTS = 8;
static const uint32_t HIST_BINS  = MAX_POINTS;

// Golden model em software, para a corrida se auto-verificar. Reproduz o
// mesmo LCG do inject_san e a mesma semantica do testbench aceito: o primeiro
// ponto de saida cuja confianca alcanca q_delta vence; quem nao assenta cai no
// head final e conta como catastrofe.
struct Golden {
    std::vector<uint32_t> hist;
    uint32_t cat = 0;
    uint64_t flops = 0;
};

static Golden golden(uint32_t n_samples, uint32_t n_points, uint32_t q_delta,
                     const uint64_t *lut) {
    Golden g;
    g.hist.assign(HIST_BINS, 0);
    uint32_t st = 12345u;
    const uint32_t n_conf = n_points - 1;
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t idx = n_points - 1;
        bool achou = false;
        for (uint32_t k = 0; k < n_conf; k++) {
            st = st * 1103515245u + 12345u;
            uint32_t q = (st >> 17) & 0x7FFF;   // igual ao injetor
            if (!achou && q >= q_delta) { idx = k; achou = true; }
        }
        if (idx == n_points - 1) g.cat++;
        g.hist[idx]++;
        g.flops += lut[idx];
    }
    return g;
}

static uint32_t ip2i(const char *s) {
    in_addr a;
    if (inet_pton(AF_INET, s, &a) != 1) throw std::runtime_error(std::string("ip invalido: ") + s);
    return ntohl(a.s_addr);
}

int main(int argc, char **argv) {
    // uso: ctl_san <xclbin> <ip-injetor> <porta-origem> <porta-fpga> <n_points> <n_samples> [q_delta]
    if (argc < 7) {
        fprintf(stderr,
            "uso: %s <xclbin> <ip-injetor> <porta-origem> <porta-fpga> <n_points> <n_samples> [q_delta]\n"
            "ex:  %s san_net.xclbin 10.100.100.2 50000 62781 7 100003 328\n", argv[0], argv[0]);
        return 2;
    }
    const char *xclbin   = argv[1];
    uint32_t inj_ip      = ip2i(argv[2]);
    uint32_t inj_port    = (uint32_t)atoi(argv[3]);
    uint32_t fpga_port   = (uint32_t)atoi(argv[4]);
    uint32_t n_points    = (uint32_t)atoi(argv[5]);
    uint32_t n_samples   = (uint32_t)atol(argv[6]);
    uint32_t q_delta     = (argc > 7) ? (uint32_t)atoi(argv[7]) : 328;

    const uint32_t FPGA_IP = ip2i("10.100.100.50");
    const uint32_t FPGA_GW = ip2i("10.100.100.1");
    if (n_points < 2 || n_points > MAX_POINTS) {
        fprintf(stderr, "n_points fora de 2..%u\n", MAX_POINTS);
        return 2;
    }

    try {
        xrt::device dev(0);
        auto uuid = dev.load_xclbin(xclbin);

        // --- 1. CMAC do lado CABEADO, com RS-FEC (padrao do fabric) ---
        xrt::ip cmac(dev, uuid, "cmac_1:cmac_1");
        cmac.write_register(CMAC_RSFEC_EN, 0x3);
        cmac.write_register(CMAC_RSFEC_IND, 7);
        cmac.write_register(CMAC_RESET, 0xC0000000);
        cmac.write_register(CMAC_RESET, 0x0);
        // O alinhamento com RS-FEC NAO e' instantaneo — ler o status logo apos
        // tirar o reset da sempre "sem sinal". O u250-vnx-init.sh, ja validado,
        // espera 3 s; aqui sondamos ate 15 s, que tolera um link mais lento sem
        // pagar o custo fixo quando ele sobe rapido.
        uint32_t st = 0;
        for (int tent = 0; tent < 60; tent++) {
            st = cmac.read_register(CMAC_RX_STATUS);
            if (st & 0x1) { printf("cmac_1 alinhou em ~%.1f s\n", tent * 0.25); break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        printf("cmac_1 rx_status=0x%X (%s)\n", st, (st & 0x1) ? "ALINHADO" : "SEM SINAL");
        if (!(st & 0x1)) {
            fprintf(stderr, "ERRO: o link nao alinhou em 15 s; sem isso nenhum beat chega.\n");
            return 3;
        }

        // --- 2. NetworkLayer: identidade + tabela de sockets ---
        xrt::ip nl(dev, uuid, "networklayer:networklayer_1");
        nl.write_register(NL_MAC_L, 0x350DD7EAu);
        nl.write_register(NL_MAC_H, 0x000Au);
        nl.write_register(NL_IP,    FPGA_IP);
        nl.write_register(NL_GW,    FPGA_GW);
        nl.write_register(NL_NSOCK, 16);
        // socket 0 = o injetor. Os outros ficam invalidos.
        nl.write_register(NL_THEIR_IP + 0 * NL_SOCK_STEP, inj_ip);
        nl.write_register(NL_THEIR_PT + 0 * NL_SOCK_STEP, inj_port);
        nl.write_register(NL_MY_PT    + 0 * NL_SOCK_STEP, fpga_port);
        nl.write_register(NL_VALID    + 0 * NL_SOCK_STEP, 1);
        for (uint32_t i = 1; i < 16; i++)
            nl.write_register(NL_VALID + i * NL_SOCK_STEP, 0);
        // leitura de volta: escrita em AXI-Lite falha calada se o offset erra
        uint32_t rb_ip = nl.read_register(NL_THEIR_IP), rb_pt = nl.read_register(NL_MY_PT),
                 rb_vl = nl.read_register(NL_VALID);
        printf("socket[0] theirIP=0x%08X myPort=%u valid=%u\n", rb_ip, rb_pt, rb_vl);
        if (rb_ip != inj_ip || rb_pt != fpga_port || rb_vl != 1) {
            fprintf(stderr, "ERRO: tabela de sockets nao confere na leitura de volta.\n");
            return 4;
        }

        // --- 3. Kernel: LUT de custo, escalares, buffers de saida ---
        // Assinatura lida do PROPRIO artefato (xclbinutil --info), nao suposta:
        //   0 samples_in (stream, AXIS)   1 lut          2 q_delta
        //   3 n_points                    4 n_samples    5 hist_out
        //   6 catastrophe_out             7 flops_out
        // A porta de stream OCUPA o indice 0, entao os argumentos nao podem ser
        // passados posicionalmente — tem de ser set_arg com indice explicito.
        xrt::kernel k(dev, uuid, "krnl_san_scan_net:{krnl_san_scan_net_1}");
        auto bo_lut  = xrt::bo(dev, MAX_POINTS * sizeof(uint64_t), k.group_id(1));
        auto bo_hist = xrt::bo(dev, HIST_BINS  * sizeof(uint32_t), k.group_id(5));
        auto bo_cat  = xrt::bo(dev, sizeof(uint32_t),              k.group_id(6));
        auto bo_flop = xrt::bo(dev, sizeof(uint64_t),              k.group_id(7));

        // LUT identica a do testbench aceito: prefixos exatos de MAC, 64 bits,
        // MAX_POINTS entradas com o indice saturado em n_points-1.
        auto *lut = bo_lut.map<uint64_t *>();
        for (uint32_t k_i = 0; k_i < MAX_POINTS; k_i++) {
            uint32_t idx = (k_i < n_points) ? k_i : n_points - 1;
            lut[k_i] = 1000000ULL * (idx + 1) * (idx + 1);
        }
        bo_lut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memset(bo_hist.map<uint32_t *>(), 0, HIST_BINS * sizeof(uint32_t));
        bo_hist.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        *bo_cat.map<uint32_t *>()  = 0; bo_cat.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        *bo_flop.map<uint64_t *>() = 0; bo_flop.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // O kernel BLOQUEIA no stream ate chegarem n_samples. Arma e volta:
        // quem alimenta e o inject_san, do outro lado da fibra.
        uint32_t eth0 = nl.read_register(NL_ETH_IN);
        xrt::run run(k);
        run.set_arg(1, bo_lut);
        run.set_arg(2, q_delta);
        run.set_arg(3, n_points);
        run.set_arg(4, n_samples);
        run.set_arg(5, bo_hist);
        run.set_arg(6, bo_cat);
        run.set_arg(7, bo_flop);
        run.start();
        printf("ARMADO — rode agora, no outro no:\n");
        printf("  ./inject_san 10.100.100.50 %u %u %u 140 %u\n",
               fpga_port, n_samples, n_points, inj_port);
        fflush(stdout);

        // 120 s: a 100G a coorte inteira leva ms; se estourar, nao chegou nada.
        auto estado = run.wait(std::chrono::seconds(120));
        uint32_t eth1 = nl.read_register(NL_ETH_IN);
        if (estado == ERT_CMD_STATE_TIMEOUT) {
            fprintf(stderr,
                "TIMEOUT: o kernel nao completou. eth_in %u -> %u (delta %u), arp_in=%u icmp_in=%u\n"
                "  delta 0  => nenhum quadro chegou ao network layer (cabo/VLAN/rota)\n"
                "  delta > 0 => chegou quadro mas o UDP descartou (tabela de sockets/porta)\n",
                eth0, eth1, eth1 - eth0,
                nl.read_register(NL_ARP_IN), nl.read_register(NL_ICMP_IN));
            return 5;
        }

        bo_hist.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_cat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_flop.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint32_t cat  = *bo_cat.map<uint32_t *>();
        uint64_t flop = *bo_flop.map<uint64_t *>();
        auto *hist = bo_hist.map<uint32_t *>();
        uint64_t soma = 0;
        for (uint32_t i = 0; i < HIST_BINS; i++) soma += hist[i];

        // --- 4. Confronto com o golden model ---
        Golden g = golden(n_samples, n_points, q_delta, lut);
        bool bate = (cat == g.cat) && (flop == g.flops);
        for (uint32_t i = 0; i < HIST_BINS; i++) if (hist[i] != g.hist[i]) bate = false;

        printf("SAN_NET n_samples=%u n_points=%u q_delta=%u\n", n_samples, n_points, q_delta);
        printf("  fpga   cat=%-8u flops=%-16llu hist=", cat, (unsigned long long)flop);
        for (uint32_t i = 0; i < HIST_BINS; i++) printf("%u ", hist[i]);
        printf("\n  golden cat=%-8u flops=%-16llu hist=", g.cat, (unsigned long long)g.flops);
        for (uint32_t i = 0; i < HIST_BINS; i++) printf("%u ", g.hist[i]);
        printf("\n  soma=%llu (esperado %u)  quadros eth_in delta=%u\n",
               (unsigned long long)soma, n_samples, eth1 - eth0);

        if (soma != n_samples) {
            fprintf(stderr, "FALHA: histograma nao fecha — houve perda de pacote UDP.\n");
            return 6;
        }
        printf("SAN_NET_%s\n", bate ? "BIT_EXATO" : "DIVERGENTE");
        return bate ? 0 : 7;
    } catch (const std::exception &e) {
        fprintf(stderr, "ERRO: %s\n", e.what());
        return 1;
    }
}
