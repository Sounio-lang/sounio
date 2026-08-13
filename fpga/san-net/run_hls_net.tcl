# csynth da variante em rede do SAN scan.
# Mesmo part e mesmo clock do fluxo original (250 MHz), para o resultado
# ser comparavel com o kernel via DMA ja aceito (T3_GREEN).
open_project -reset prj_san_scan_net
set_top krnl_san_scan_net
add_files krnl_san_scan_net.cpp
open_solution -reset -flow_target vitis sol_u250
set_part {xcu250-figd2104-2L-e}
create_clock -period 4 -name default
csynth_design
exit
