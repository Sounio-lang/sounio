<!-- docs:meta
topic_id: repo.docs.training.rosetta.03-bubble-sort
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.training.rosetta.03-bubble-sort
-->

# Bubble Sort

## Python
```python
def bubble_sort(arr: list[int]) -> list[int]:
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
```

## Sounio
```sio
struct SortBuf { data: [i64; 256], len: i64 }

fn bubble_sort(buf: &! SortBuf) with Mut, Panic {
    var i: i64 = 0
    while i < buf.len {
        var j: i64 = 0
        while j < buf.len - i - 1 {
            if buf.data[j] > buf.data[j + 1] {
                let temp = buf.data[j]
                buf.data[j] = buf.data[j + 1]
                buf.data[j + 1] = temp
            }
            j = j + 1
        }
        i = i + 1
    }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    var buf = SortBuf { data: [0; 256], len: 7 }
    buf.data[0] = 64
    buf.data[1] = 34
    buf.data[2] = 25
    buf.data[3] = 12
    buf.data[4] = 22
    buf.data[5] = 11
    buf.data[6] = 90

    bubble_sort(&! buf)

    var i: i64 = 0
    while i < buf.len {
        print(buf.data[i])
        print(" ")
        i = i + 1
    }
    println("")
    0
}
```

## Key Differences
- Struct wrapper `SortBuf` to work around bare array `&!` mutation bug
- `&! SortBuf` for mutable reference (not `&mut`)
- No tuple swap `a, b = b, a` — use temp variable
- `while` loops instead of `for i in range(n)` (while is safer for nested mutation)
- Explicit `.len` field instead of `len()` method
