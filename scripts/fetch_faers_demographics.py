#!/usr/bin/env python3
"""Pull per-report demographics for each of the 35 CYP-triple cocktails.

For each report: age (in years; normalize across units), sex (0=unk,
1=male, 2=female), reporter qualification (1=physician, 2=pharmacist,
3=other HCP, 4=lawyer, 5=consumer), country (US vs other).

Aggregate per triple:
    n_reports, mean_age, frac_female, frac_hcp, frac_us

Output: data/faers_demographics.csv

The Sounio analysis (examples/oct_cohort_test.sio) will compute η²
stratified by octonion associator direction m for each of these 4
demographic variables and compare to the |asymmetry| η² = 0.421 from
oct_associator_expanded.sio. If any demographic η² is comparable to
0.421, the algebra finding is likely a cohort-structure proxy. If all
demographic η² are much smaller (< 0.15), the finding is not explained
by obvious demographic confounders.
"""
import urllib.request
import urllib.error
import json
import itertools
import time
import csv
import os

DRUGS = {
    1: ("CYP1A2",  "theophylline"),
    2: ("CYP2C9",  "warfarin"),
    3: ("CYP2C8",  "repaglinide"),
    4: ("CYP2B6",  "bupropion"),
    5: ("CYP2C19", "omeprazole"),
    6: ("CYP2D6",  "codeine"),
    7: ("CYP3A4",  "simvastatin"),
}

API = "https://api.fda.gov/drug/event.json"
PAGE = 100
CAP = 1000


def fetch_page(d1, d2, d3, skip):
    q = (f"search=patient.drug.medicinalproduct:{d1}"
         f"+AND+patient.drug.medicinalproduct:{d2}"
         f"+AND+patient.drug.medicinalproduct:{d3}"
         f"&limit={PAGE}&skip={skip}")
    try:
        r = urllib.request.urlopen(f"{API}?{q}", timeout=20).read()
        return json.loads(r).get("results", [])
    except Exception:
        return []


def extract_demographics(record):
    """Returns (age_years_or_None, sex_int_or_0, qual_int_or_0, is_us_bool)."""
    patient = record.get("patient", {})
    # Age — normalize to years. patientonsetageunit: 800=decade, 801=year,
    # 802=month, 803=week, 804=day, 805=hour (per ICH E2B convention).
    age = None
    raw_age = patient.get("patientonsetage")
    unit = patient.get("patientonsetageunit")
    if raw_age is not None:
        try:
            a = float(raw_age)
            if unit == "801" or unit == "800":       # year or decade→ close enough
                age = a if unit == "801" else a * 10
            elif unit == "802":
                age = a / 12.0
            elif unit == "803":
                age = a / 52.0
            elif unit == "804":
                age = a / 365.0
            elif unit == "805":
                age = a / (365.0 * 24.0)
            else:
                age = a    # fall back
        except Exception:
            age = None
    sex_raw = patient.get("patientsex")
    try:
        sex = int(sex_raw) if sex_raw is not None else 0
    except Exception:
        sex = 0
    qual = 0
    ps = record.get("primarysource")
    if isinstance(ps, dict):
        try:
            qual = int(ps.get("qualification", 0) or 0)
        except Exception:
            qual = 0
    country = (record.get("primarysourcecountry") or "").strip().upper()
    is_us = country == "US" or country == "USA" or country == "UNITED STATES"
    return age, sex, qual, is_us


def main():
    triples = list(itertools.combinations(range(1, 8), 3))
    rows = []
    t0 = time.time()
    for idx, (i, j, k) in enumerate(triples):
        cyp_a, drug_a = DRUGS[i]
        cyp_b, drug_b = DRUGS[j]
        cyp_c, drug_c = DRUGS[k]
        ages = []
        n_female = 0
        n_sexed = 0
        n_hcp = 0
        n_with_qual = 0
        n_us = 0
        n_total = 0
        skip = 0
        while skip < CAP:
            records = fetch_page(drug_a, drug_b, drug_c, skip)
            if not records:
                break
            for rec in records:
                n_total += 1
                age, sex, qual, is_us = extract_demographics(rec)
                if age is not None and 0 < age < 120:
                    ages.append(age)
                if sex == 1 or sex == 2:
                    n_sexed += 1
                    if sex == 2:
                        n_female += 1
                if qual > 0:
                    n_with_qual += 1
                    if qual in (1, 2, 3):
                        n_hcp += 1
                if is_us:
                    n_us += 1
            skip += PAGE
            if len(records) < PAGE:
                break
            time.sleep(0.25)
        mean_age = sum(ages) / len(ages) if ages else -1
        frac_female = n_female / n_sexed if n_sexed > 0 else -1
        frac_hcp = n_hcp / n_with_qual if n_with_qual > 0 else -1
        frac_us = n_us / n_total if n_total > 0 else -1
        rows.append({
            "cyp_a": cyp_a, "cyp_b": cyp_b, "cyp_c": cyp_c,
            "drugs": f"{drug_a}+{drug_b}+{drug_c}",
            "n_reports": n_total,
            "n_with_age": len(ages),
            "mean_age": round(mean_age, 2),
            "n_sexed": n_sexed,
            "frac_female": round(frac_female, 4),
            "n_with_qual": n_with_qual,
            "frac_hcp": round(frac_hcp, 4),
            "frac_us": round(frac_us, 4),
        })
        elapsed = time.time() - t0
        print(f"[{idx+1:2d}/35] ({i},{j},{k}) n={n_total:4d}  "
              f"age={mean_age:5.1f}  F={frac_female:.2f}  "
              f"HCP={frac_hcp:.2f}  US={frac_us:.2f}  [{elapsed:.0f}s]")

    out = "data/faers_demographics.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
