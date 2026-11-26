`molecular::host_guest`: one-site guests in a rigid host framework. Builds the sII clathrate
lattice from the Fd-3m generators, evaluates the guest-host field with quadratic Feynman-Hibbs
and its gradients in the guest's Lennard-Jones parameters, and runs grand-canonical trials at a
given fugacity with cage assignment. The fluctuation sensitivities of the cage occupancy to the
H2 parameters reproduce predictions written before the run (`examples/molecular/h2_sii_gcmc.sio`,
12 of 12 within 3σ).
