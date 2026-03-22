<!-- docs:meta
topic_id: repo.docs.training.rosetta.02-binary-search
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.training.rosetta.02-binary-search
-->

# Binary Search

## Python
```python
def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search(data, 23))   # 5
print(binary_search(data, 99))   # -1
```

## Sounio
```sio
fn binary_search(arr: &[i64; 256], len: i64, target: i64) -> i64 with Mut, Panic, Div {
    var lo: i64 = 0
    var hi: i64 = len - 1
    while lo <= hi {
        let mid = (lo + hi) / 2
        if arr[mid] == target { return mid }
        if arr[mid] < target {
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    0 - 1
}

fn main() -> i32 with IO, Mut, Panic, Div {
    var data: [i64; 256] = [0; 256]
    data[0] = 2
    data[1] = 5
    data[2] = 8
    data[3] = 12
    data[4] = 16
    data[5] = 23
    data[6] = 38
    data[7] = 56
    data[8] = 72
    data[9] = 91

    let idx = binary_search(&data, 10, 23)
    print("Found 23 at index: ")
    print(idx)
    println("")

    let idx2 = binary_search(&data, 10, 99)
    print("Found 99 at index: ")
    print(idx2)
    println("")
    0
}
```

## Key Differences
- Fixed-size array `[i64; 256]` with explicit length param vs Python `list`
- `0 - 1` instead of `-1` (no unary minus)
- Effects: `Div` for division, `Panic` for array bounds, `Mut` for variables
- Array passed by reference `&[i64; 256]`
- No `elif` — use `if/else if` or nested `if`
