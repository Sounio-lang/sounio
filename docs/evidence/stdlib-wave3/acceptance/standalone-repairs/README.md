# Resolve BigFrame logarithm imports and EISA argument borrowing

Four BigFrame extension modules called ln without importing its real definition. They now import math::pure::ln. EISA previously passed an exclusive borrow of b and read b.n_code in the same argument list; it now computes the identical destination before the exclusive borrow. Numerical algorithms and assertions are unchanged. The local-model syntax proposal failed compilation and was not adopted; corrections use existing repository syntax.

Slurm19324 replays exactly six original activated tests with raw826d9c18: all six check and compile successfully; NP, SP, EISA backend and EISA bridge also execute with exit0. ML and PD still execute with exit139. Aggregate4pass/2fail/0skip, no selection/source identity errors. Their native failures remain open pending the fresh short-circuit compiler; the import repair is not called complete runtime acceptance.
