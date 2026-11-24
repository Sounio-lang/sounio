Madaros no longer starts a module-level global at zero when it ignores its
initializer.

- A fixed-array global with a closed initializer is now written before
  `main`, element by element. That covers literals, struct, array and tuple
  literals, `None` and `Type::Variant`. Before, an array of structs such as
  `var TURNS: [Oct; 5] = [Oct { .. }, ..]` was dropped: it held null handles,
  and a program indexing it died with SIGILL. The conversational O-SSM
  examples ran their five turns on that.
- `None` and a unit enum variant now count as closed, so a one-word global
  initialized with them is evaluated instead of left zero.
- Anything else that does not fold is refused with the new `E309` instead of
  being dropped. That is an initializer calling a fn, or naming a global, that
  the parser cannot fold in its own module. `var A: [i64; 3] = [impure(), 20, 30]`
  used to read `0 0 0` and exit 0. A call initializer with effects never ran.
  To fix it, declare the global without an initializer and assign it in `main`.

The global-init identity gate now pins the refusal for the shadow and
ambiguous fixtures and for the three scoping-off controls that used to read a
zero.

`stdlib/physics/em.sio` no longer declares a module-level `C`. Its
initializer called `speed_of_light_approx()`, which does not fold when
`constants::physical` is parsed after `em`. In that case `C` read the
same-named global of `physics::sr` by name, which happened to hold the
same value. The module now calls the constant where it is used.
