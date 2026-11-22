`formal/Bootstrap/TMR.lean` adds the vote kernel of the triple-modular-redundancy interpreter. Three copies of the state step independently on faulty silicon. Their results are voted, the majority is written back to the copy that lost, and with no majority the interpreter refuses.
- `tri_err`: one voted step goes wrong with probability at most 3·q², where q is a copy's fault rate.
- `trun_err`: n voted steps go wrong with probability at most n·3·q², against n·q unhardened.
- `witness`: at rates in [5·10⁻⁸, 2·10⁻⁷], a million voted steps go wrong with probability at most 1.2·10⁻⁷, where the unhardened bound is 0.2.

This module votes whole states and assumes an exact voter. Voting only what a step wrote, together with the exposure of words that are only read, and faults in the voter itself, come in later modules.
