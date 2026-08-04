set script_dir [file dirname [file normalize [info script]]]
if {![info exists ::env(CHAIN41_INPUTS)] || ![info exists ::env(CHAIN41_EXPECTED)]} {
    error "CHAIN41_INPUTS and CHAIN41_EXPECTED are required"
}
open_project -reset target23_chained_taylor41_csim
set_top target23_chained_taylor41
add_files "$script_dir/kernel.cpp"
add_files -tb "$script_dir/testbench.cpp"
open_solution -reset solution -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 10.0 -name default
csim_design -argv "$::env(CHAIN41_INPUTS) $::env(CHAIN41_EXPECTED)"
close_project
exit
