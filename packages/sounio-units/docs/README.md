# sounio-units

Canonical package for SI dimensional analysis and metrological calibration.

## Import

```sounio
use sounio_units::*
```

## Stdlib compatibility

Legacy imports remain valid via shims:

- `use units::lib::*` → `stdlib/units/lib.sio`
- `use units::mod::*` → `stdlib/units/mod.sio`
- `use metrology::calibration::*` → `stdlib/metrology/calibration.sio`

## References

- JCGM 100:2008 (GUM)
- JCGM 200:2012 (VIM)
- ISO 17025:2017