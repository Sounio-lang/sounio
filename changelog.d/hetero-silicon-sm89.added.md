`stdlib/hetero/silicon` builds sm_89 cubins directly from measured instruction
encodings, with no PTX and no ptxas. It provides a single-block and a checked
multiblock GEMV (`si_gemv_sm89`, `si_gemv_grid_sm89`). `hetero::cuda_exec`
gains `ce_module_image` to load such an image. A CPU-only test pins every image
by size and byte digest.
