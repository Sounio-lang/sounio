# csim da variante em rede contra o golden model independente.
# Prova de equivalencia bit-exata: mesmas coortes, mesmo golden, mesmo
# empacotamento do artefato aceito (T3_GREEN). So o transporte mudou.
open_project -reset prj_san_net_csim
set_top krnl_san_scan_net
add_files krnl_san_scan_net.cpp
add_files -tb tb_san_scan_net.cpp
open_solution -reset -flow_target vitis sol_u250
set_part {xcu250-figd2104-2L-e}
create_clock -period 4 -name default
csim_design
exit
