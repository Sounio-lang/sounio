<!-- docs:meta
topic_id: repo.docs.audit.epistemic-merge-2026-10-24.readme
authority: repo_only
audience: users
last_validated: 2026-10-24
validated_by: peelle.cpp
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-merge-2026-10-24.readme
-->

# Peelle's Pertinent Puzzle — witness for ADR-016 decision 8

Two measurements of one quantity, `1.5` and `1.0`, each with 10 % independent uncertainty and a shared 20 % normalisation uncertainty.

```
/usr/bin/clang++ -std=c++2b -O2 -Wall -Wextra peelle.cpp -o peelle && ./peelle
```

Output (Apple clang, macOS arm64, 2026-10-24):

```
additive GLS : 0.8824 +- 0.2183  (w1 = -0.2353, w2 = +1.2353)
log-domain   : 1.2247 +- 0.2598  (rel 21.21%, w1 = +0.5000, w2 = +0.5000)
```

The additive model evaluates the relative normalisation component at the measured values, and the generalised-least-squares estimate falls below both inputs with a negative weight. Combining in the log domain, where the shared factor is additive, gives equal weights and the geometric mean.

## Claims forbidden

- That the log-domain result is correct for every shared relative component: it requires a positive quantity (ADR-016 decision 9).
- That the additive GLS is wrong as an estimator: it is the BLUE for its covariance model; the model is what is wrong here.
