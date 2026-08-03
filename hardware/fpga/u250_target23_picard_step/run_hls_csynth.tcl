open_project -reset target23_picard_step_synth
set_top target23_picard_step
add_files kernel.cpp
open_solution -reset solution -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 4.0 -name default
csynth_design
exit
