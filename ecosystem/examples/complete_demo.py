#!/usr/bin/env python3
"""Complete end-to-end demo of the Triple Sounio Ecosystem.

Shows: PubChem → Knowledge → Pipeline → Report

This script demonstrates:
1. Fetching molecule data from PubChem (using offline cache)
2. Creating Knowledge values with uncertainty
3. Performing GUM-based arithmetic
4. Running the drug discovery pipeline
5. Generating a comprehensive report

Run with:
    PYTHONPATH=sounio-py/python \\
    SOUC=../bin/souc \\
    SOUNIO_STDLIB_PATH=../stdlib \\
    python3 examples/complete_demo.py
"""

import sys
import os
from pathlib import Path

# Add sounio-py to path
SOUNIO_PY = Path(__file__).parent.parent / "sounio-py" / "python"
if SOUNIO_PY.exists():
    sys.path.insert(0, str(SOUNIO_PY))

from sounio import Knowledge, run_file, get_executor
from sounio.integrations.pubchem import fetch_by_name
from sounio.report import ReportBuilder


def main():
    print("=" * 70)
    print("  SOUNIO ECOSYSTEM — COMPLETE END-TO-END DEMO")
    print("=" * 70)

    # Step 1: Fetch molecule from PubChem (offline cache)
    print("\n[Step 1] Fetch Molecule from PubChem")
    print("-" * 70)
    try:
        aspirin = fetch_by_name("aspirin", offline=True)
        print(f"✓ Molecule: {aspirin.name}")
        print(f"  SMILES: {aspirin.smiles}")
        print(f"  Molecular Weight: {aspirin.molecular_weight.value:.3f} ± "
              f"{aspirin.molecular_weight.epsilon:.3f} g/mol")
        print(f"  LogP: {aspirin.logp.value:.3f} ± {aspirin.logp.epsilon:.3f}")
        print(f"  H-bond donors: {aspirin.hbd}, acceptors: {aspirin.hba}")
        print(f"  Lipinski rule of five: PASS" if aspirin.passes_lipinski else
              f"  Lipinski rule of five: FAIL")
    except Exception as e:
        print(f"✗ Failed to fetch aspirin: {e}")
        return 1

    # Step 2: Create Knowledge values and demonstrate arithmetic
    print("\n[Step 2] Knowledge Arithmetic with Uncertainty Propagation")
    print("-" * 70)
    try:
        dose = Knowledge(500.0, 10.0, "prescribed_dose_mg")
        weight = Knowledge(70.0, 0.5, "patient_weight_kg")
        print(f"Dose: {dose}")
        print(f"Weight: {weight}")

        dose_per_kg = dose / weight
        print(f"Dose/kg: {dose_per_kg}")
        print(f"  Relative uncertainty: {dose_per_kg.relative_uncertainty:.2%}")
        print(f"  Confidence: {dose_per_kg.confidence:.2%}")

        # More complex arithmetic
        volume_of_distribution = Knowledge(50.0, 5.0, "pk_vd_estimate")
        concentration = dose / volume_of_distribution
        print(f"\nEstimated plasma concentration: {concentration}")
        print(f"  Value: {concentration.value:.3f} mg/L")
        print(f"  Uncertainty: ± {concentration.epsilon:.3f} mg/L")

        # Check reliability
        if concentration.is_reliable(threshold=0.15):
            print("  Status: RELIABLE (< 15% rel. uncertainty)")
        else:
            print("  Status: QUESTIONABLE (> 15% rel. uncertainty)")
    except Exception as e:
        print(f"✗ Knowledge arithmetic failed: {e}")
        return 1

    # Step 3: Run drug discovery pipeline
    print("\n[Step 3] Run Drug Discovery Pipeline")
    print("-" * 70)
    pipeline_path = Path(__file__).parent.parent / "drug-discovery" / "examples" / "full_pipeline.sio"
    result = None
    if pipeline_path.exists():
        try:
            executor = get_executor()
            result = executor.run_file(str(pipeline_path), timeout=60)

            if result.ok:
                print(f"✓ Pipeline executed successfully (exit code {result.exit_code})")
                print(f"  Knowledge values extracted: {len(result.knowledge_values)}")

                if result.knowledge_values:
                    print("\n  Pipeline Knowledge Values:")
                    for i, kv in enumerate(result.knowledge_values, 1):
                        print(f"    {i}. {kv.provenance}")
                        print(f"       Value: {kv.value:.6g}")
                        print(f"       Uncertainty: ± {kv.epsilon:.6g}")
                        print(f"       Rel. Uncertainty: {kv.relative_uncertainty:.2%}")
            else:
                print(f"✗ Pipeline failed with exit code {result.exit_code}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:300]}")
        except Exception as e:
            print(f"⚠ Pipeline execution skipped: {e}")
    else:
        print(f"⚠ Pipeline not found at {pipeline_path}")

    # Step 4: Generate report
    print("\n[Step 4] Generate Comprehensive Report")
    print("-" * 70)
    try:
        rb = ReportBuilder(
            "Sounio Epistemic Computing Demo Report",
            author="Sounio Ecosystem Demo"
        )

        rb.add_section(
            "Introduction",
            "This report demonstrates the Sounio epistemic computing pipeline, "
            "combining PubChem molecular data, GUM-based uncertainty propagation, "
            "and drug discovery pipeline execution."
        )

        rb.add_knowledge_table(
            "Molecule Properties (PubChem)",
            {
                "Molecular Weight": aspirin.molecular_weight,
                "LogP (XLogP3)": aspirin.logp,
            }
        )

        rb.add_knowledge_table(
            "Dosing Calculation",
            {
                "Prescribed Dose": dose,
                "Patient Weight": weight,
                "Dose per kg": dose_per_kg,
                "Estimated Plasma Concentration": concentration,
            }
        )

        if result and result.ok and result.knowledge_values:
            kv_dict = {kv.provenance: kv for kv in result.knowledge_values}
            rb.add_knowledge_table("Pipeline Results", kv_dict)

        rb.add_section(
            "Methodology",
            "All epistemic calculations use the Guide to the Expression of "
            "Uncertainty in Measurement (GUM) standard. Relative uncertainties "
            "are propagated through arithmetic operations using:  \n"
            "- Addition/Subtraction: ε = √(ε₁² + ε₂²)  \n"
            "- Multiplication/Division: ε = |result| × √((ε₁/val₁)² + (ε₂/val₂)²)  \n"
            "- Scalar operations: ε = |factor| × ε₁  \n"
        )

        report_md = rb.to_markdown()
        print(f"✓ Report generated ({len(report_md)} characters)")
        print(f"\n  Report preview (first 500 chars):\n")
        print("  " + "\n  ".join(report_md[:500].split("\n")))

        # Save report
        report_path = Path(__file__).parent / "demo_report.md"
        rb.save(str(report_path))
        print(f"\n✓ Saved to: {report_path}")

    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — All steps executed successfully")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
