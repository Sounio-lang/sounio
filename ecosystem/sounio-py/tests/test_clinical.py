"""Tests for clinical data import from CDISC/SDTM domains."""

import pathlib
import sys

import pytest

# Add python package to path
test_dir = pathlib.Path(__file__).parent
repo_root = test_dir.parent
python_pkg = repo_root / "python"

from sounio.integrations.clinical import (
    read_sdtm_domain,
    parse_dm,
    parse_ex,
    parse_pc,
    load_study,
)
from sounio.knowledge import Knowledge
from sounio.types import PatientData


# Fixtures
FIXTURES_DIR = test_dir / "fixtures" / "sdtm"


class TestReadSdtmDomain:
    """Test SDTM file reading."""

    def test_read_dm_csv(self):
        """Test reading Demographics domain from CSV."""
        dm_path = FIXTURES_DIR / "dm.csv"
        df = read_sdtm_domain(str(dm_path))
        assert len(df) == 5
        assert "USUBJID" in df.columns
        assert "WEIGHT" in df.columns

    def test_read_domain_auto_detect(self):
        """Test auto-detection of domain code from filename."""
        dm_path = FIXTURES_DIR / "dm.csv"
        df = read_sdtm_domain(str(dm_path))  # No domain parameter
        assert len(df) == 5

    def test_read_nonexistent_file(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            read_sdtm_domain(str(FIXTURES_DIR / "nonexistent.csv"))

    def test_read_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            read_sdtm_domain("test.txt")


class TestParseDM:
    """Test Demographics domain parsing."""

    def test_parse_dm_basic(self):
        """Test basic Demographics parsing."""
        dm_path = FIXTURES_DIR / "dm.csv"
        df = read_sdtm_domain(str(dm_path))
        patients = parse_dm(df)

        assert len(patients) == 5
        assert all(isinstance(p, PatientData) for p in patients)

    def test_parse_dm_patient_data(self):
        """Test that parsed patient data has correct attributes."""
        dm_path = FIXTURES_DIR / "dm.csv"
        df = read_sdtm_domain(str(dm_path))
        patients = parse_dm(df)

        p1 = patients[0]
        assert p1.patient_id == "SUBJ-001"
        assert p1.age == 45
        assert isinstance(p1.weight, Knowledge)
        assert p1.weight.value == 82.5

    def test_parse_dm_weight_uncertainty(self):
        """Test that weight uncertainty is set correctly."""
        dm_path = FIXTURES_DIR / "dm.csv"
        df = read_sdtm_domain(str(dm_path))
        patients = parse_dm(df, weight_epsilon=0.5)

        for p in patients:
            assert p.weight.epsilon == 0.5
            assert "DM" in p.weight.provenance

    def test_parse_dm_custom_weight_epsilon(self):
        """Test custom weight uncertainty."""
        dm_path = FIXTURES_DIR / "dm.csv"
        df = read_sdtm_domain(str(dm_path))
        patients = parse_dm(df, weight_epsilon=1.0)

        for p in patients:
            assert p.weight.epsilon == 1.0

    def test_parse_dm_missing_weight_column(self):
        """Test handling of missing WEIGHT column."""
        import pandas as pd
        df = pd.DataFrame({
            "USUBJID": ["SUBJ-001"],
            "AGE": [45],
            "SEX": ["M"],
            "RACE": ["WHITE"],
            "ACTARMCD": ["ACTIVE"],
        })
        patients = parse_dm(df)
        assert len(patients) == 1
        # Weight should be NaN
        import math
        assert math.isnan(patients[0].weight.value)

    def test_parse_dm_missing_columns(self):
        """Test error handling for missing required columns."""
        import pandas as pd
        df = pd.DataFrame({"USUBJID": ["SUBJ-001"]})

        with pytest.raises(KeyError, match="Missing required columns"):
            parse_dm(df)


class TestParseEX:
    """Test Exposure domain parsing."""

    def test_parse_ex_basic(self):
        """Test basic Exposure parsing."""
        ex_path = FIXTURES_DIR / "ex.csv"
        df = read_sdtm_domain(str(ex_path))
        ex_epistemic = parse_ex(df)

        assert len(ex_epistemic) == 5
        assert "dose" in ex_epistemic.measured_columns

    def test_parse_ex_dose_values(self):
        """Test that dose values are parsed correctly."""
        ex_path = FIXTURES_DIR / "ex.csv"
        df = read_sdtm_domain(str(ex_path))
        ex_epistemic = parse_ex(df)

        doses = ex_epistemic.get_knowledge_column("dose")
        assert len(doses) == 5
        assert doses[0].value == 500.0
        assert doses[2].value == 0.0

    def test_parse_ex_dose_uncertainty(self):
        """Test that dose uncertainty is calculated as dose * epsilon_pct."""
        ex_path = FIXTURES_DIR / "ex.csv"
        df = read_sdtm_domain(str(ex_path))
        ex_epistemic = parse_ex(df, dose_epsilon_pct=0.02)

        doses = ex_epistemic.get_knowledge_column("dose")
        # First dose: 500 mg * 0.02 = 10.0
        assert abs(doses[0].epsilon - 10.0) < 1e-6
        # Fourth dose: 250 mg * 0.02 = 5.0
        assert abs(doses[3].epsilon - 5.0) < 1e-6

    def test_parse_ex_zero_dose(self):
        """Test handling of zero dose (placebo)."""
        ex_path = FIXTURES_DIR / "ex.csv"
        df = read_sdtm_domain(str(ex_path))
        ex_epistemic = parse_ex(df, dose_epsilon_pct=0.02)

        doses = ex_epistemic.get_knowledge_column("dose")
        # Zero dose should have zero uncertainty
        assert doses[2].value == 0.0
        assert doses[2].epsilon == 0.0

    def test_parse_ex_custom_epsilon_pct(self):
        """Test custom dose epsilon percentage."""
        ex_path = FIXTURES_DIR / "ex.csv"
        df = read_sdtm_domain(str(ex_path))
        ex_epistemic = parse_ex(df, dose_epsilon_pct=0.05)

        doses = ex_epistemic.get_knowledge_column("dose")
        # First dose: 500 mg * 0.05 = 25.0
        assert abs(doses[0].epsilon - 25.0) < 1e-6

    def test_parse_ex_missing_columns(self):
        """Test error handling for missing columns."""
        import pandas as pd
        df = pd.DataFrame({"USUBJID": ["SUBJ-001"]})

        with pytest.raises(KeyError, match="Missing required columns"):
            parse_ex(df)


class TestParsePC:
    """Test Pharmacoconcentration domain parsing."""

    def test_parse_pc_basic(self):
        """Test basic Pharmacoconcentration parsing."""
        pc_path = FIXTURES_DIR / "pc.csv"
        df = read_sdtm_domain(str(pc_path))
        pc_epistemic = parse_pc(df)

        assert len(pc_epistemic) == 6
        assert "concentration" in pc_epistemic.measured_columns

    def test_parse_pc_concentration_values(self):
        """Test that concentration values are parsed correctly."""
        pc_path = FIXTURES_DIR / "pc.csv"
        df = read_sdtm_domain(str(pc_path))
        pc_epistemic = parse_pc(df)

        concs = pc_epistemic.get_knowledge_column("concentration")
        assert len(concs) == 6
        assert concs[0].value == 0.0
        assert concs[1].value == 12.5
        assert concs[2].value == 8.3

    def test_parse_pc_concentration_uncertainty(self):
        """Test that concentration uncertainty is calculated as conc * assay_cv."""
        pc_path = FIXTURES_DIR / "pc.csv"
        df = read_sdtm_domain(str(pc_path))
        pc_epistemic = parse_pc(df, assay_cv=0.15)

        concs = pc_epistemic.get_knowledge_column("concentration")
        # First conc: 0.0 ng/mL * 0.15 = 0.0
        assert abs(concs[0].epsilon - 0.0) < 1e-6
        # Second conc: 12.5 ng/mL * 0.15 = 1.875
        assert abs(concs[1].epsilon - 1.875) < 1e-6
        # Third conc: 8.3 ng/mL * 0.15 = 1.245
        assert abs(concs[2].epsilon - 1.245) < 1e-6

    def test_parse_pc_custom_assay_cv(self):
        """Test custom assay coefficient of variation."""
        pc_path = FIXTURES_DIR / "pc.csv"
        df = read_sdtm_domain(str(pc_path))
        pc_epistemic = parse_pc(df, assay_cv=0.20)

        concs = pc_epistemic.get_knowledge_column("concentration")
        # Second conc: 12.5 ng/mL * 0.20 = 2.5
        assert abs(concs[1].epsilon - 2.5) < 1e-6

    def test_parse_pc_provenance(self):
        """Test that provenance is set correctly."""
        pc_path = FIXTURES_DIR / "pc.csv"
        df = read_sdtm_domain(str(pc_path))
        pc_epistemic = parse_pc(df)

        concs = pc_epistemic.get_knowledge_column("concentration")
        assert "PC" in concs[0].provenance
        assert "T0" in concs[0].provenance
        assert "T1" in concs[1].provenance

    def test_parse_pc_missing_columns(self):
        """Test error handling for missing columns."""
        import pandas as pd
        df = pd.DataFrame({"USUBJID": ["SUBJ-001"]})

        with pytest.raises(KeyError, match="Missing required columns"):
            parse_pc(df)


class TestLoadStudy:
    """Test complete study loading."""

    def test_load_study_basic(self):
        """Test loading a complete study from directory."""
        study = load_study(str(FIXTURES_DIR))

        assert "dm" in study
        assert "ex" in study
        assert "pc" in study

        assert len(study["dm"]) == 5
        assert len(study["ex"]) == 5
        assert len(study["pc"]) == 6

    def test_load_study_dm_patients(self):
        """Test that DM patients are loaded correctly."""
        study = load_study(str(FIXTURES_DIR))

        patients = study["dm"]
        assert all(isinstance(p, PatientData) for p in patients)
        assert patients[0].patient_id == "SUBJ-001"
        assert patients[0].age == 45

    def test_load_study_ex_patients(self):
        """Test that EX exposures are loaded correctly."""
        study = load_study(str(FIXTURES_DIR))

        ex_df = study["ex"]
        subjects = ex_df.df["subject_id"].tolist()
        assert len(subjects) == 5
        assert "SUBJ-001" in subjects

    def test_load_study_pc_patients(self):
        """Test that PC concentrations are loaded correctly."""
        study = load_study(str(FIXTURES_DIR))

        pc_df = study["pc"]
        subjects = pc_df.df["subject_id"].tolist()
        assert len(subjects) == 6
        assert "SUBJ-001" in subjects

    def test_load_study_dose_merging(self):
        """Test that EX doses are merged into DM PatientData."""
        study = load_study(str(FIXTURES_DIR))

        patients = study["dm"]
        # SUBJ-001 dose should be 500 mg (from EX)
        assert patients[0].dose.value == 500.0
        # SUBJ-003 dose should be 0 mg (placebo)
        assert patients[2].dose.value == 0.0

    def test_load_study_missing_dm(self):
        """Test error handling when DM file is missing."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only EX and PC
            import shutil
            shutil.copy(FIXTURES_DIR / "ex.csv", tmpdir)
            shutil.copy(FIXTURES_DIR / "pc.csv", tmpdir)

            with pytest.raises(FileNotFoundError, match="DM domain file"):
                load_study(tmpdir)

    def test_load_study_missing_ex(self):
        """Test error handling when EX file is missing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            shutil.copy(FIXTURES_DIR / "dm.csv", tmpdir)
            shutil.copy(FIXTURES_DIR / "pc.csv", tmpdir)

            with pytest.raises(FileNotFoundError, match="EX domain file"):
                load_study(tmpdir)

    def test_load_study_missing_pc(self):
        """Test error handling when PC file is missing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            shutil.copy(FIXTURES_DIR / "dm.csv", tmpdir)
            shutil.copy(FIXTURES_DIR / "ex.csv", tmpdir)

            with pytest.raises(FileNotFoundError, match="PC domain file"):
                load_study(tmpdir)


class TestIntegrationEndToEnd:
    """Integration tests for the complete workflow."""

    def test_study_with_knowledge_propagation(self):
        """Test that knowledge (uncertainty) propagates correctly."""
        study = load_study(str(FIXTURES_DIR))

        # Patient 1: weight 82.5 ± 0.5 kg, dose 500 ± 10 mg
        p1 = study["dm"][0]
        assert p1.weight.value == 82.5
        assert p1.weight.epsilon == 0.5
        assert p1.dose.value == 500.0
        assert p1.dose.epsilon == 10.0  # 500 * 0.02

    def test_study_concentration_summary(self):
        """Test epistemic summary across concentration measurements."""
        study = load_study(str(FIXTURES_DIR))

        pc_df = study["pc"]
        summary = pc_df.epistemic_summary("concentration")

        # Summary should have mean and pooled uncertainty
        assert summary.value > 0.0
        assert summary.epsilon >= 0.0

    def test_study_describe_epistemic(self):
        """Test epistemic statistics computation."""
        study = load_study(str(FIXTURES_DIR))

        pc_stats = study["pc"].describe_epistemic()
        assert "concentration" in pc_stats.index
        assert "mean" in pc_stats.columns
        assert "mean_epsilon" in pc_stats.columns
        assert pc_stats.loc["concentration", "mean"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
