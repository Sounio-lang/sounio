#!/usr/bin/env python3
"""Executable certificate mirroring formal/lean4/SounioFoCssSurfaceParity.lean.

Algebraic FO Css surface independence (residual §5.4 mathematical half).
Exact rational freezes match R1–R4 annex. Zero floating-point tolerance on
primary freezes (fractions compared via fractions.Fraction).

Exit 0 and print FO_CSS_SURFACE_PARITY_CERT_OK on success.
"""
from __future__ import annotations

from fractions import Fraction


def infusion_rate(F, Dose, tau):
    return F * Dose / tau


def clearance(CL0, e_eta):
    return CL0 * e_eta


def css_import(F, Dose, tau, CL0, e_eta):
    return infusion_rate(F, Dose, tau) / clearance(CL0, e_eta)


def css_site(F, Dose, tau, CL0, e_eta):
    return (F * Dose / tau) / (CL0 * e_eta)


def css_method(F, Dose, tau, CL0, e_eta):
    return css_site(F, Dose, tau, CL0, e_eta)


def css_call_result(F, Dose, tau, CL0, e_eta):
    return css_method(F, Dose, tau, CL0, e_eta)


def main() -> int:
    F0 = Fraction(4, 5)
    Dose0 = Fraction(500)
    tau0 = Fraction(12)
    CL0 = Fraction(5)
    V0 = Fraction(50)
    eEta0 = Fraction(1)

    sigF = Fraction(1, 20)
    sigDose = Fraction(10)
    sigCL0 = Fraction(3, 10)
    sigEta = Fraction(1, 10)
    sigV0 = Fraction(2)

    checks: list[tuple[str, bool]] = []

    # Surface definitional equality at seeds
    ci = css_import(F0, Dose0, tau0, CL0, eEta0)
    cs = css_site(F0, Dose0, tau0, CL0, eEta0)
    cm = css_method(F0, Dose0, tau0, CL0, eEta0)
    cr = css_call_result(F0, Dose0, tau0, CL0, eEta0)
    checks.append(("surfaces_agree", ci == cs == cm == cr))
    checks.append(("css_point", ci == Fraction(20, 3)))
    checks.append(("rate_point", infusion_rate(F0, Dose0, tau0) == Fraction(100, 3)))
    checks.append(("cl_point", clearance(CL0, eEta0) == Fraction(5)))

    # FO Var(Css)
    sF = Dose0 / (tau0 * CL0)
    sD = F0 / (tau0 * CL0)
    sC = -(F0 * Dose0) / (tau0 * CL0 * CL0)
    sE = -ci
    fo_var_css = (
        sF * sF * sigF * sigF
        + sD * sD * sigDose * sigDose
        + sC * sC * sigCL0 * sigCL0
        + sE * sE * sigEta * sigEta
    )
    checks.append(("var_css", fo_var_css == Fraction(191, 240)))

    fo_var_cl = (Fraction(1) * Fraction(1) * sigCL0 * sigCL0) + (
        CL0 * CL0 * sigEta * sigEta
    )
    checks.append(("var_cl", fo_var_cl == Fraction(17, 50)))

    sRF = Dose0 / tau0
    sRD = F0 / tau0
    fo_var_rate = sRF * sRF * sigF * sigF + sRD * sRD * sigDose * sigDose
    checks.append(("var_rate", fo_var_rate == Fraction(689, 144)))

    def fo_var_E(rho: Fraction) -> Fraction:
        sCL = V0
        sV = CL0
        sE1 = CL0 * V0
        sE2 = CL0 * V0
        return (
            sCL * sCL * sigCL0 * sigCL0
            + sV * sV * sigV0 * sigV0
            + sE1 * sE1 * sigEta * sigEta
            + sE2 * sE2 * sigEta * sigEta
            + Fraction(2) * sE1 * sE2 * sigEta * sigEta * rho
        )

    checks.append(("var_E_0", fo_var_E(Fraction(0)) == Fraction(1575)))
    checks.append(("var_E_half", fo_var_E(Fraction(1, 2)) == Fraction(2200)))
    checks.append(("var_E_1", fo_var_E(Fraction(1)) == Fraction(2825)))
    checks.append(
        (
            "var_E_law",
            fo_var_E(Fraction(0)) == Fraction(1575) + Fraction(1250) * Fraction(0)
            and fo_var_E(Fraction(1, 2))
            == Fraction(1575) + Fraction(1250) * Fraction(1, 2)
            and fo_var_E(Fraction(1)) == Fraction(1575) + Fraction(1250) * Fraction(1),
        )
    )

    kel_shared = (CL0 * eEta0) / (V0 * eEta0)
    kel_peeled = CL0 / V0
    checks.append(("kel_point", kel_shared == kel_peeled == Fraction(1, 10)))

    sKCL = Fraction(1) / V0
    sKV = -CL0 / (V0 * V0)
    fo_var_kel = sKCL * sKCL * sigCL0 * sigCL0 + sKV * sKV * sigV0 * sigV0
    checks.append(("var_kel", fo_var_kel == Fraction(13, 250000)))

    def var_css_at_tau(tau: Fraction) -> Fraction:
        sF_ = Dose0 / (tau * CL0)
        sD_ = F0 / (tau * CL0)
        sC_ = -(F0 * Dose0) / (tau * CL0 * CL0)
        sE_ = -(F0 * Dose0) / (tau * CL0)
        return (
            sF_ * sF_ * sigF * sigF
            + sD_ * sD_ * sigDose * sigDose
            + sC_ * sC_ * sigCL0 * sigCL0
            + sE_ * sE_ * sigEta * sigEta
        )

    checks.append(("var_tau8", var_css_at_tau(Fraction(8)) == Fraction(573, 320)))
    checks.append(("var_tau12", var_css_at_tau(Fraction(12)) == Fraction(191, 240)))
    checks.append(("var_tau24", var_css_at_tau(Fraction(24)) == Fraction(191, 960)))
    checks.append(
        (
            "tau_scale",
            var_css_at_tau(Fraction(8)) == fo_var_css * Fraction(9, 4)
            and var_css_at_tau(Fraction(24)) == fo_var_css * Fraction(1, 4),
        )
    )

    failed = [(n, ok) for n, ok in checks if not ok]
    for name, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")

    if failed:
        print(f"FO_CSS_SURFACE_PARITY_CERT_FAIL {len(failed)}/{len(checks)}")
        return 1

    print(f"FO_CSS_SURFACE_PARITY_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
