#!/bin/bash
# Retoma so o LINK: o krnl_san_scan_net.xo ja foi compilado com sucesso
# (COMPILE rc=0) na tentativa anterior, que morreu por erro de sintaxe no
# san_net.cfg ('defaultTool' em vez de 'defaultFreqHz').
export XILINXD_LICENSE_FILE=$HOME/.Xilinx/Xilinx.lic:$HOME/.Xilinx/Xilinx-cmac.lic
export XILINX_XRT=$HOME/xrt-stub
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
source /opt/amd/2025.1/Vitis/settings64.sh

cd "$HOME/sanet"
XPFM=/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm
VNX="$HOME/vnx"

[ -f krnl_san_scan_net.xo ] || { echo "SEM .xo — recompilar antes"; exit 1; }
date -u +"INICIO %Y-%m-%dT%H:%M:%SZ"

HLSIP="$VNX/NetLayers/100G-fpga-network-stack-core/synthesis_results_noHBM"
# o networklayer.xo e um kernel COMPOSTO: seu block design depende de sub-IPs
# HLS (arp_server, icmp_server, udp, packet_handler) que vivem neste repo.
# Sem --user_ip_repo_paths o generate_target falha com "definition for
# subcore dependency xilinx.com:hls:arp_server:1.0 is not available".
v++ -l -t hw --platform "$XPFM" --config san_net.cfg --save-temps \
    --user_ip_repo_paths "$HLSIP" \
    --temp_dir ./_link -o san_net.xclbin \
    krnl_san_scan_net.xo \
    "$VNX/Ethernet/_x.xilinx_u250_gen3x16_xdma_4_1_202210_1/cmac_1.xo" \
    "$VNX/NetLayers/_x.xilinx_u250_gen3x16_xdma_4_1_202210_1/networklayer.xo" \
    "$VNX/Basic_kernels/_x.xilinx_u250_gen3x16_xdma_4_1_202210_1/krnl_mm2s.xo"
rc=$?
echo "LINK rc=$rc"
date -u +"FIM %Y-%m-%dT%H:%M:%SZ rc=$rc"
ls -la san_net.xclbin 2>/dev/null || echo "SEM XCLBIN"
