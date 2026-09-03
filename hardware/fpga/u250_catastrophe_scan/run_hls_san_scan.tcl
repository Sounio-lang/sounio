# run_hls_san_scan.tcl — Vitis HLS flow for krnl_san_scan (U250).
# Usage (on the DL380 once Vitis is installed):
#   cd hardware/fpga/u250_catastrophe_scan
#   vitis_hls -f run_hls_san_scan.tcl        # csim + csynth + cosim
# csim/cosim acceptance: TB_SAN_SCAN_PASS (bit-exact vs independent golden).
open_project -reset prj_san_scan
set_top krnl_san_scan
add_files krnl_san_scan.cpp
add_files -tb tb_san_scan.cpp
open_solution -reset "sol_u250" -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 250MHz -name default
# csim: C-level bit-exactness vs the independent golden
csim_design -clean
# synthesis
csynth_design
# cosim: RTL vs C (same testbench)
cosim_design -rtl verilog
exit
