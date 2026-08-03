#!/usr/bin/env python3
"""Independent Decimal/RK4 center-orbit Liouville replay for one frozen leaf."""

from __future__ import annotations

import argparse
import hashlib
import platform
from decimal import Decimal, localcontext
from pathlib import Path


ZS = "22.3274637391"
ORIGIN_X = "15.186446520640786"
ORIGIN_Y = "10.908543194765466"
UNSTABLE_X = "-0.67430316214199759"
UNSTABLE_Y = "-0.73845463335624273"
STABLE_X = "-0.94170446778164518"
STABLE_Y = "0.33644122125579123"
RADIUS_U = "0.004"
RADIUS_S = "0.3"
MAX_STEPS = 20000
EVENT_BISECTIONS = 48


def fail(message: str) -> None:
    raise SystemExit(f"decimal center worker error: {message}")


def center(depth_u: int, index_u: int, depth_s: int, index_s: int) -> list[Decimal]:
    ru, rs = Decimal(RADIUS_U), Decimal(RADIUS_S)
    u = -ru + (Decimal(index_u) + Decimal("0.5")) * (2 * ru) / Decimal(2**depth_u)
    s = -rs + (Decimal(index_s) + Decimal("0.5")) * (2 * rs) / Decimal(2**depth_s)
    return [
        Decimal(ORIGIN_X) + Decimal(UNSTABLE_X) * u + Decimal(STABLE_X) * s,
        Decimal(ORIGIN_Y) + Decimal(UNSTABLE_Y) * u + Decimal(STABLE_Y) * s,
        Decimal(0), Decimal(0),
    ]


def field(state: list[Decimal]) -> list[Decimal]:
    x, y, w, _ell = state
    zs = Decimal(ZS)
    return [
        2 * y * y - x * y,
        x * y - y * (w + zs) / 2,
        x * y - w - zs,
        x - y - (w + zs) / 2 - 1,
    ]


def combine(base: list[Decimal], *terms: tuple[Decimal, list[Decimal]]) -> list[Decimal]:
    return [base[i] + sum((scale * vector[i] for scale, vector in terms), Decimal(0)) for i in range(4)]


def rk4(state: list[Decimal], step: Decimal) -> list[Decimal]:
    k1 = field(state)
    k2 = field(combine(state, (step / 2, k1)))
    k3 = field(combine(state, (step / 2, k2)))
    k4 = field(combine(state, (step, k3)))
    return combine(state, (step / 6, k1), (step / 3, k2), (step / 3, k3), (step / 6, k4))


def localize_event(left: list[Decimal], step: Decimal) -> tuple[Decimal, list[Decimal]]:
    low, high = Decimal(0), step
    high_state = rk4(left, high)
    for _ in range(EVENT_BISECTIONS):
        middle = (low + high) / 2
        middle_state = rk4(left, middle)
        if middle_state[2] < 0:
            low = middle
        else:
            high, high_state = middle, middle_state
    return high, high_state


def integrate(depth_u: int, index_u: int, depth_s: int, index_s: int,
              precision: int, step_power: int) -> dict[str, Decimal | int]:
    with localcontext() as context:
        context.prec = precision
        state = center(depth_u, index_u, depth_s, index_s)
        initial_normal = state[0] * state[1] - Decimal(ZS)
        step = Decimal(1) / Decimal(2**step_power)
        time = Decimal(0)
        armed = False
        events: list[tuple[Decimal, list[Decimal]]] = []
        steps = 0
        while len(events) < 2 and steps < MAX_STEPS:
            following = rk4(state, step)
            if following[2] < 0:
                armed = True
            if armed and state[2] < 0 <= following[2]:
                local_time, event_state = localize_event(state, step)
                events.append((time + local_time, event_state))
                armed = False
            state = following
            time += step
            steps += 1
        if len(events) != 2:
            fail(f"expected two minus-plus returns, got {len(events)}")
        second_time, second = events[1]
        final_normal = second[0] * second[1] - Decimal(ZS)
        q0_area = (
            Decimal(UNSTABLE_X) * Decimal(STABLE_Y)
            - Decimal(STABLE_X) * Decimal(UNSTABLE_Y)
        ) * Decimal(RADIUS_U) * Decimal(RADIUS_S)
        determinant = second[3].exp() * initial_normal / final_normal * q0_area
        return {
            "steps": steps, "return1_time": events[0][0], "return2_time": second_time,
            "ell": second[3], "initial_normal": initial_normal,
            "final_normal": final_normal, "q0_area": q0_area,
            "determinant": determinant,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("u_depth", type=int)
    parser.add_argument("u_index", type=int)
    parser.add_argument("s_depth", type=int)
    parser.add_argument("s_index", type=int)
    parser.add_argument("challenge")
    parser.add_argument("attempt_binding")
    args = parser.parse_args()
    for depth, index, name in ((args.u_depth, args.u_index, "u"), (args.s_depth, args.s_index, "s")):
        if depth < 1 or depth > 30 or index < 0 or index >= 2**depth:
            fail(f"invalid {name} coordinate")
    for value, name in ((args.challenge, "challenge"), (args.attempt_binding, "attempt binding")):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            fail(f"invalid {name}")
    coarse = integrate(args.u_depth, args.u_index, args.s_depth, args.s_index, 50, 9)
    fine = integrate(args.u_depth, args.u_index, args.s_depth, args.s_index, 80, 10)
    with localcontext() as comparison_context:
        comparison_context.prec = 100
        delta = abs(coarse["determinant"] - fine["determinant"])
    self_consistent = (
        coarse["initial_normal"] > 0 and coarse["final_normal"] > 0
        and fine["initial_normal"] > 0 and fine["final_normal"] > 0
        and coarse["determinant"] < 0 and fine["determinant"] < 0
        and delta <= Decimal("1E-16")
    )
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print("SCHEMA=sounio.cs6.v7b-target23-decimal-center-worker.v1")
    print(f"WORKER_SOURCE_SHA256={source_sha}")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"PYTHON_IMPLEMENTATION={platform.python_implementation()}")
    print("DECIMAL_IMPLEMENTATION=stdlib-decimal")
    print(f"RUN_CHALLENGE={args.challenge}")
    print(f"ATTEMPT_BINDING={args.attempt_binding}")
    print(f"U_DEPTH={args.u_depth}")
    print(f"U_INDEX={args.u_index}")
    print(f"S_DEPTH={args.s_depth}")
    print(f"S_INDEX={args.s_index}")
    for prefix, result in (("COARSE", coarse), ("FINE", fine)):
        for key in ("steps", "return1_time", "return2_time", "ell", "initial_normal", "final_normal", "q0_area", "determinant"):
            print(f"{prefix}_{key.upper()}={result[key]}")
    print(f"ABSOLUTE_DETERMINANT_DELTA={delta}")
    print(f"CENTER_REPLAY_SELF_CONSISTENT={str(self_consistent).lower()}")
    print("CAPD_USED_BY_INTEGRATOR=false")
    print("RIGOROUS_INTERVAL_CERTIFICATE=false")
    print("POINTWISE_FALSIFICATION_ONLY=true")
    if not self_consistent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
