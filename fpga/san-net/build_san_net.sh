#!/bin/bash
# Build da variante em rede: compila o kernel e linka com CMAC + networklayer
# do VNx no mesmo xclbin. ~2h.
#
# Roda como devsounio: a autorizacao do Vitis esta em ~/.Xilinx (root nao tem).
# XILINX_XRT aponta para o stub, porque o Makefile do VNx exige a variavel
# setada mas o wrapper do xclbinutil so usa o loader interno quando ela esta
# vazia — o stub reconcilia os dois.
export XILINXD_LICENSE_FILE=$HOME/.Xilinx/Xilinx.lic:$HOME/.Xilinx/Xilinx-cmac.lic
export XILINX_XRT=$HOME/xrt-stub
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
source /opt/amd/2025.1/Vitis/settings64.sh

cd "$HOME/sanet"
XPFM=/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm
VNX="$HOME/vnx"
B=basic.intf3.xilinx_u250_gen3x16_xdma_4_1_202210_1

date -u +"INICIO %Y-%m-%dT%H:%M:%SZ"

# 1) compila o kernel novo -> .xo
v++ -c -t hw --platform "$XPFM" -k krnl_san_scan_net \
    krnl_san_scan_net.cpp -o krnl_san_scan_net.xo --save-temps
rc=$?
echo "COMPILE rc=$rc"
[ $rc -ne 0 ] && { date -u +"FIM %Y-%m-%dT%H:%M:%SZ rc=$rc"; exit $rc; }

# 2) linka com os .xo do VNx ja construidos (cmac + networklayer)
v++ -l -t hw --platform "$XPFM" --config san_net.cfg --save-temps \
    --temp_dir ./_link -o san_net.xclbin \
    krnl_san_scan_net.xo \
    "$VNX/Ethernet/_x.xilinx_u250_gen3x16_xdma_4_1_202210_1/cmac_1.xo" \
    "$VNX/NetLayers/_x.xilinx_u250_gen3x16_xdma_4_1_202210_1/networklayer.xo" \
    -j 8
rc=$?
echo "LINK rc=$rc"
date -u +"FIM %Y-%m-%dT%H:%M:%SZ rc=$rc"
ls -la san_net.xclbin 2>/dev/null || echo "SEM XCLBIN"
