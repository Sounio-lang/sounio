set script_dir [file dirname [file normalize [info script]]]
if {![info exists ::env(TARGET23_INPUTS)] || ![info exists ::env(TARGET23_EXPECTED)]} {
    error "TARGET23_INPUTS and TARGET23_EXPECTED are required"
}
open_project -reset target23_batch_csim
set_top target23_batch
add_files "$script_dir/kernel.cpp"
add_files -tb "$script_dir/testbench.cpp"
open_solution -reset solution1 -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 4.0 -name default
csim_design -argv "$::env(TARGET23_INPUTS) $::env(TARGET23_EXPECTED)"
close_project
exit
