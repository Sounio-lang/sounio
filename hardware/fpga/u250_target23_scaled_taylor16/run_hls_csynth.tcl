open_project -reset target23_scaled_taylor16_synth
set_top target23_scaled_taylor16
add_files kernel.cpp
open_solution -reset solution -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 4.0 -name default
csynth_design
exit
