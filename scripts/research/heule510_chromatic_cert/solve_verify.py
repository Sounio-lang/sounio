from pysat.formula import CNF
from pysat.solvers import Solver
cnf=CNF(from_file='heule510_4col.cnf')
s=Solver(name='glucose3', bootstrap_with=cnf, with_proof=True)
print("4-colorable?", s.solve())
pr=s.get_proof()
open('heule510_4col.drat','w').write('\n'.join(pr)+'\n')
print("DRAT lines:",len(pr))
s.delete()
