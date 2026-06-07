# Native V2 Release Showcase

These single-file programs dogfood the verified native subset through the
self-hosted compiler built for this lane. They are import-free and integer-only.

- `prime_sieve_100.sio`: Sieve of Eratosthenes up to 100. It validates the
  prime count, prime sum, last prime, and selected sieve cells. Expected exit:
  `25`.
- `tiny_register_vm.sio`: Tiny bytecode interpreter using `enum Op`, `match`,
  structs, arrays, and mutable machine state. It computes and stores `42`, then
  validates the interpreter state. Expected exit: `42`.

Validation shape for each file:

```sh
/tmp/sc_mc.elf --check examples/showcase/<file>.sio
/tmp/sc_mc.elf --native-v2-compile examples/showcase/<file>.sio -o /tmp/<file>.elf
chmod +x /tmp/<file>.elf
/tmp/<file>.elf
echo $?
```
