# stdlib/heliobiology

Space-weather indices, solar-event classification, units, and published
associations between geomagnetic activity and cardiovascular outcomes.

Sourcing rule (#2835): every coefficient or threshold in this module is either
a published definition (NOAA Space Weather Scales, Mursula et al. 2022 Dst
classes, IAU 2012 astronomical unit) or a value copied from a table of a cited
peer-reviewed paper, with its standard uncertainty derived from that paper's
95 % confidence interval. A number without a source does not belong here.

- `indices.sio` -- NOAA G and S scales, Dst storm classes, IMF helpers.
- `solar.sio` -- GOES flare classes, NOAA R scale, ballistic CME transit.
- `units.sio` -- magnetic, radiometric, particle-flux and dosimetry units.
- `effects.sio` -- Zilli Vieira et al. 2019 (mortality per SD of daily Kp,
  263 U.S. cities) and Zilli Vieira et al. 2022 (HRV during intense
  geomagnetic disturbance), as `Epistemic` values.

Test: `tests/stdlib/heliobiology/test_heliobiology.sio` checks every effect
estimate against an independent transcription of the source tables.
