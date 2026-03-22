<!-- docs:meta
topic_id: repo.docs.training.rosetta.08-matrix-multiply
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.training.rosetta.08-matrix-multiply
-->

# Matrix Multiplication

## Python
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
print(C)  # [[19 22] [43 50]]
```

## Sounio
```sio
// 2x2 matrix stored as flat [f64; 4] array (row-major)
// Index: mat[row * 2 + col]

fn mat2_mul(a: &[f64; 4], b: &[f64; 4], out: &![f64; 4]) with Mut, Panic {
    // C[i][j] = sum_k A[i][k] * B[k][j]
    (*out)[0] = a[0] * b[0] + a[1] * b[2]   // C[0][0]
    (*out)[1] = a[0] * b[1] + a[1] * b[3]   // C[0][1]
    (*out)[2] = a[2] * b[0] + a[3] * b[2]   // C[1][0]
    (*out)[3] = a[2] * b[1] + a[3] * b[3]   // C[1][1]
}

// General NxN (up to 16x16) matrix multiply
fn matmul(a: &[f64; 256], b: &[f64; 256], out: &![f64; 256], n: i64) with Mut, Panic {
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            var sum = 0.0
            var k: i64 = 0
            while k < n {
                sum = sum + a[i * n + k] * b[k * n + j]
                k = k + 1
            }
            (*out)[i * n + j] = sum
            j = j + 1
        }
        i = i + 1
    }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let a: [f64; 4] = [1.0, 2.0, 3.0, 4.0]
    let b: [f64; 4] = [5.0, 6.0, 7.0, 8.0]
    var c: [f64; 4] = [0.0; 4]

    mat2_mul(&a, &b, &! c)

    // Expected: [19, 22, 43, 50]
    print("C = [")
    print(c[0])
    print(", ")
    print(c[1])
    print(", ")
    print(c[2])
    print(", ")
    print(c[3])
    println("]")

    assert(c[0] == 19.0)
    assert(c[1] == 22.0)
    assert(c[2] == 43.0)
    assert(c[3] == 50.0)
    0
}
```

## Key Differences
- **No numpy** — manual matrix operations with flat arrays
- **Fixed-size arrays** — `[f64; 4]` for 2x2, `[f64; 256]` for up to 16x16
- **Row-major indexing** — `mat[row * n + col]`
- **Explicit deref** — `(*out)[i] = val` for bare array `&!` mutation
- **Three nested loops** for general matmul (classic O(n^3))
- Sounio's stdlib has `stdlib/linalg/` for more sophisticated operations
