`print(k)` and `println(k)` on a `Knowledge<T>` value are refused with the new
`E310`. Madaros printed the nominal value and dropped the uncertainty and the
confidence; lean_single printed the handle address. Print the field you mean,
for example `println(k.value)`, which needs `with Epistemic`.

Six tests now project the field explicitly. Three of them used to give different
output on the two engines (`covid_2020_kernel`, `epsilon_comparison_valid`,
`med/vancomycin_full_propagation`); they now print the same and leave the
engine-parity baseline.
