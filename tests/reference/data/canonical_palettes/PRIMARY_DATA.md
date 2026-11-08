# Canonical palette primary data

These are pinned primary data assets, not executable Python dependencies.
The C++23 lexical reader extracts numeric lists without executing their code.

- BIDS/colormap commit `bc549477db0c12b54a5928087552ad2cf274980f`:
  [colormaps.py](https://github.com/BIDS/colormap/blob/bc549477db0c12b54a5928087552ad2cf274980f/colormaps.py)
  and [LICENSE.txt](https://github.com/BIDS/colormap/blob/bc549477db0c12b54a5928087552ad2cf274980f/LICENSE.txt).
  Nathaniel J. Smith, Stéfan van der Walt, and Eric Firing (viridis); CC0.
  `BIDS-colormaps.py.txt` is byte-identical to the source, SHA-256
  `b7478d10e2ccfdb22128867e137f16f55b687e9c225065e91bcbd6bb9e6c0352`.
  License SHA `30419703377136667f2c8a426d396874a8dadf53699d277fc8ad759d05092799`.
- Matplotlib commit `39b27ddaafe636f06776a2044b5df05d07648dc4`:
  [_cm_listed.py](https://github.com/matplotlib/matplotlib/blob/39b27ddaafe636f06776a2044b5df05d07648dc4/lib/matplotlib/_cm_listed.py)
  and [license](https://github.com/matplotlib/matplotlib/blob/39b27ddaafe636f06776a2044b5df05d07648dc4/LICENSE/LICENSE).
  `Matplotlib-cm-listed.py.txt.gz` uses deterministic gzip without a timestamp;
  gzip SHA `679ea1847a643bc691970022f1c7945bff5fb57e7adb67fbd3f46d8efeba3d5b`.
  Its byte-identical expanded source SHA is
  `ddad3698f5129ceb1792a445371286c08bc9080298e657b3054aea19c9659ef9`;
  retained license SHA `5a1a81ea301728c8bba2933da832c0cd62229daf20893a024ab3d53244468dbc`.

Runtime bytes are generated as `floor(255 * published channel)`. The primary
floating source remains separate and immutable; exact decimal and binary64
conversion paths agree on all 1,024 RGB triplets. Both primary sources agree on
all 3,072 represented binary64 channels. Palette selection uses listed bins,
not interpolation between these entries. The original eight-anchor/polynomial
fits are retained under explicitly approximate names rather than relabeled as
canonical data. No D27 image golden is generated or replaced.
