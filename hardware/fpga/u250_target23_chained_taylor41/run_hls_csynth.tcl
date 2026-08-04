set script_dir [file dirname [file normalize [info script]]]
open_project -reset target23_chained_taylor41_synth
set_top target23_chained_taylor41
add_files "$script_dir/kernel.cpp"
open_solution -reset solution -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 10.0 -name default
csynth_design
close_project
exit
