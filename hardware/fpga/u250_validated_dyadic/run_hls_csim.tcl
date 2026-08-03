open_project -reset validated_dyadic_kat
set_top validated_dyadic_kat
add_files kernel.cpp
add_files -tb testbench.cpp
open_solution -reset solution -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 4.0 -name default
csim_design -argv "$::env(KAT_INPUTS) $::env(KAT_EXPECTED)"
exit
