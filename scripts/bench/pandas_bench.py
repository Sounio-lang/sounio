#!/usr/bin/env python3
# pandas side of the bigframe vs pandas benchmark: identical ops over 1,000,000 rows,
# timed with perf_counter around the operation only (data built once outside the timer).
import numpy as np, pandas as pd, time
n = 1_000_000
df = pd.DataFrame({'a': np.arange(n, dtype='float64'),
                   'k': (np.arange(n) % 10).astype('float64'),
                   'v': np.ones(n)})
def bench(fn, K):
    fn()
    t = time.perf_counter()
    for _ in range(K): fn()
    return (time.perf_counter() - t) / K * 1000.0
print(f"col_sum {bench(lambda: df['a'].sum(), 200):.4f}")
print(f"filter_count {bench(lambda: int((df['a'] > 499999.5).sum()), 200):.4f}")
print(f"groupby_sum {bench(lambda: df.groupby('k')['v'].sum(), 50):.4f}")
