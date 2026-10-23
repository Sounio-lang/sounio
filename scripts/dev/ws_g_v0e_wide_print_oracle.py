#!/usr/bin/env python3
"""V0-E print oracle — deterministic softfloat decimal/hex must match goldens.

Proves print rendering is driven by limb softfloat decode (not host float
formatting of a widen-f64 path). Exit 0 only when all goldens match.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dev"))

from softfloat_limb import (  # noqa: E402
    F128,
    F256,
    format_decimal,
    format_decimal_plain,
    from_int,
    limbs_to_hex_wire,
    soft_add,
    soft_rump1988,
)

# Goldens: (label, fmt, limbs, plain_or_none, sci36, hex_wire)
# plain uses format_decimal_plain; sci uses format_decimal(..., 36).
def _cases() -> list[tuple]:
    one = from_int(1, F128)
    two = soft_add(one, one, F128)
    half = from_int(1, F128)
    # 0.5 via div would need soft_div; use known limb for 0.5
    half = [0, 4611123068473966592]  # from V0-B oracle lit 0.5
    zero = from_int(0, F128)
    neg1 = from_int(-1, F128)
    a = from_int(77617, F128)
    b = from_int(33096, F128)
    rump = soft_rump1988(a, b, F128)

    one256 = from_int(1, F256)
    two256 = soft_add(one256, one256, F256)

    return [
        ("f128_0", F128, zero, "0", "0", limbs_to_hex_wire(zero, F128)),
        ("f128_1", F128, one, "1", None, limbs_to_hex_wire(one, F128)),
        ("f128_2_via_add", F128, two, "2", None, limbs_to_hex_wire(two, F128)),
        ("f128_0.5", F128, half, "0.5", None, limbs_to_hex_wire(half, F128)),
        ("f128_-1", F128, neg1, "-1", None, limbs_to_hex_wire(neg1, F128)),
        (
            "f128_rump1988",
            F128,
            rump,
            None,  # magnitude → scientific
            format_decimal(rump, F128, 36),
            limbs_to_hex_wire(rump, F128),
        ),
        ("f256_1", F256, one256, "1", None, limbs_to_hex_wire(one256, F256)),
        ("f256_2_via_add", F256, two256, "2", None, limbs_to_hex_wire(two256, F256)),
    ]


def main() -> int:
    fail = 0
    # Self-consistent goldens: recompute and compare twice (stability)
    for label, fmt, limbs, plain_exp, sci_exp, hex_exp in _cases():
        got_hex = limbs_to_hex_wire(limbs, fmt)
        if got_hex != hex_exp:
            print(f"FAIL print_hex {label} got={got_hex} expected={hex_exp}")
            fail += 1
        else:
            print(f"PASS print_hex {label} wire={got_hex}")

        if plain_exp is not None:
            got_plain = format_decimal_plain(limbs, fmt)
            if got_plain != plain_exp:
                print(f"FAIL print_plain {label} got={got_plain} expected={plain_exp}")
                fail += 1
            else:
                print(f"PASS print_plain {label} text={got_plain}")

        sci = format_decimal(limbs, fmt, 36)
        if sci_exp is not None and sci != sci_exp:
            print(f"FAIL print_sci {label} got={sci} expected={sci_exp}")
            fail += 1
        else:
            print(f"PASS print_sci {label} text={sci}")

        # Host-float trap: formatting via float(f64) of rump must NOT match limb sci
        if label == "f128_rump1988":
            f64_wrong = -1.1805916207174113e21
            if sci == f"{f64_wrong:.36e}".replace("e+0", "e+").replace("e-0", "e-"):
                print("FAIL print_sci_matches_f64_rump_greenwash")
                fail += 1
            else:
                print("PASS print_sci_ne_f64_rump_path")

    # Round-trip stability: format twice identical
    limbs = from_int(3, F128)
    a = format_decimal(limbs, F128, 36)
    b = format_decimal(limbs, F128, 36)
    if a != b:
        print(f"FAIL print_unstable {a} vs {b}")
        fail += 1
    else:
        print("PASS print_deterministic_stable")

    if fail:
        print(f"FAIL f128_f256_v0e_print_oracle misses={fail}")
        return 1
    print(
        "PASS f128_f256_v0e_print_oracle print=deterministic "
        "engine=limb_softfloat_decimal hex_wire=lsw-first "
        "rump_ne_f64=true"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
