# Same compiler and inputs, different gate outcome

The successful source CI run 34424138904 and failed source CI run 34432396152 archived byte-identical Madaros executables, boundary_main.sio and cap_dep.sio. The successful artifact zip was downloaded and verified against GitHub artifact digest 8449269af5d72d1fd44c94ca8a4c250331dafdb850f636f6aaef1af817b087c6. The failed artifact was already independently pinned. The successful boundary log and comparison hashes are retained here.

The failed log reaches merged IR and then reports status 124 from the wrapper, whose native compilation command has a 300-second timeout. Matching binaries and inputs exclude a difference in those bytes as the explanation for these two outcomes. They do not distinguish runtime nondeterminism, machine load, memory pressure, scheduling, or other environment differences. No elapsed-stage or resource measurements establish the cause yet. No timeout or compiler implementation was changed.
