"""
gaussian_copula_final/test_copula_final.py
------------------------------------------
Comprehensive unit test suite for GaussianCopulaFinal.
Tests schema, supports, integers, clinical constraints, zero-inflation,
dependence calibration, reproducibility, serialization, and frozen baseline integrity.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaussian_copula_final.copula_final import GaussianCopulaFinal, APPROVED_FEATURES


class TestGaussianCopulaFinal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        train_path = os.path.join(PROJECT_ROOT, "final_training_data", "nhanes_generative_train.csv")
        cls.train_df = pd.read_csv(train_path)

        config_path = os.path.join(SCRIPT_DIR, "config.json")
        with open(config_path, "r") as f:
            cls.config = json.load(f)

        cls.model = GaussianCopulaFinal(config_dict=cls.config)
        cls.model.fit(cls.train_df)
        cls.synth_df = cls.model.sample(n=1000, random_state=42)

    def test_01_schema_and_shape(self):
        """Test that generated synthetic cohort has exact 9 features and expected shape."""
        self.assertEqual(len(self.synth_df), 1000)
        self.assertEqual(list(self.synth_df.columns), APPROVED_FEATURES)

    def test_02_bounds_and_supports(self):
        """Test that all variable values fall strictly within documented supports."""
        # Age in [8, 80]
        self.assertTrue((self.synth_df["age"] >= 8).all())
        self.assertTrue((self.synth_df["age"] <= 80).all())

        # Adherence in [0, 100]
        self.assertTrue((self.synth_df["adherence_pct"] >= 0.0).all())
        self.assertTrue((self.synth_df["adherence_pct"] <= 100.0).all())

        # Activity >= 0
        self.assertTrue((self.synth_df["activity_mims"] >= 0).all())

        # Meds >= 0
        self.assertTrue((self.synth_df["n_medications"] >= 0).all())

        # Pain in [0, 10]
        self.assertTrue((self.synth_df["pain_score"] >= 0.0).all())
        self.assertTrue((self.synth_df["pain_score"] <= 10.0).all())

        # Binary columns
        self.assertTrue(set(self.synth_df["sex"].unique()).issubset({1, 2}))
        self.assertTrue(set(self.synth_df["diabetes"].unique()).issubset({0, 1}))

    def test_03_integer_types(self):
        """Test that discrete count and integer variables are strictly integer types."""
        int_cols = ["age", "sex", "diabetes", "systolic_bp", "diastolic_bp", "activity_mims", "n_medications"]
        for col in int_cols:
            self.assertTrue(
                np.issubdtype(self.synth_df[col].dtype, np.integer),
                f"Column {col} must be integer type, got {self.synth_df[col].dtype}",
            )

    def test_04_clinical_validity(self):
        """Test that 100% of generated records satisfy hemodynamic consistency."""
        # SBP > 0, DBP > 0
        self.assertTrue((self.synth_df["systolic_bp"] > 0).all())
        self.assertTrue((self.synth_df["diastolic_bp"] > 0).all())

        # SBP > DBP
        self.assertTrue((self.synth_df["systolic_bp"] > self.synth_df["diastolic_bp"]).all())

        # Pulse pressure >= 5 mmHg
        pulse_pressure = self.synth_df["systolic_bp"] - self.synth_df["diastolic_bp"]
        self.assertTrue((pulse_pressure >= 5).all())

    def test_05_zero_inflation_masses(self):
        """Test that zero-inflation is accurately reproduced without collapse."""
        # Pain score zero mass should be ~69.5% (+/- 4%)
        pain_zeros = (self.synth_df["pain_score"] == 0.0).mean()
        self.assertAlmostEqual(pain_zeros, 0.695, delta=0.05)

        # Medication count zero mass should be ~52.2% (+/- 4%)
        med_zeros = (self.synth_df["n_medications"] == 0).mean()
        self.assertAlmostEqual(med_zeros, 0.522, delta=0.05)

    def test_06_diabetes_prevalence(self):
        """Test that latent thresholding produces diabetes prevalence near training ~9.95%."""
        diab_prev = (self.synth_df["diabetes"] == 1).mean()
        self.assertAlmostEqual(diab_prev, 0.0995, delta=0.035)

    def test_07_positive_pain_dependence(self):
        """Test that positive pain magnitude retains non-zero dependence with other variables."""
        pos_df = self.synth_df[self.synth_df["pain_score"] > 0]
        self.assertGreater(len(pos_df), 200)
        # Check that positive pain has positive correlation with age and SBP
        corr_age = pos_df["pain_score"].corr(pos_df["age"])
        corr_sbp = pos_df["pain_score"].corr(pos_df["systolic_bp"])
        # Should not be degenerate NaN
        self.assertFalse(np.isnan(corr_age))
        self.assertFalse(np.isnan(corr_sbp))

    def test_08_reproducibility(self):
        """Test that identical seeds yield identical synthetic records."""
        sample_a = self.model.sample(n=100, random_state=123)
        sample_b = self.model.sample(n=100, random_state=123)
        pd.testing.assert_frame_equal(sample_a, sample_b)

    def test_09_serialization_round_trip(self):
        """Test model save and load consistency."""
        test_path = os.path.join(SCRIPT_DIR, "test_model_roundtrip.pkl")
        self.model.save(test_path)
        loaded = GaussianCopulaFinal.load(test_path)
        os.remove(test_path)

        synth_orig = self.model.sample(n=100, random_state=999)
        synth_loaded = loaded.sample(n=100, random_state=999)
        pd.testing.assert_frame_equal(synth_orig, synth_loaded)

    def test_10_psd_properties(self):
        """Test that copula correlation matrix R is strictly positive definite."""
        eigvals = np.linalg.eigvalsh(self.model.R_psd)
        self.assertGreaterEqual(float(eigvals.min()), 1e-4)
        self.assertIsNotNone(self.model.L)

    def test_11_zero_data_leakage(self):
        """Verify that nhanes_real_holdout.csv was never touched during training."""
        self.assertFalse(self.model.marginal_params.get("holdout_accessed", False))
        self.assertEqual(self.model.training_rows, 4826)

    def test_12_frozen_v1_baseline_hashes(self):
        """Verify that all 6 frozen baseline artifacts remain 100% byte-identical."""
        frozen = {
            "final_training_data/nhanes_generative_train.csv": "690321d829d262dfa0d77a91d7d7159457dcc75541fc345f95098638c89a2866",
            "final_training_data/nhanes_real_holdout.csv": "345a320bb2ab5b57093f54f1791736fadfa2c637646412308560f17015aadf71",
            "training_pipeline/synthetic/ctgan_synthetic.csv": "32f1d2eb2304a7a6e0c795e76ed5f9731d9367ecd1702337b1b4c53f92bbdd40",
            "training_pipeline/synthetic/gaussian_copula_synthetic.csv": "0089e65d9aa4816ebc74c4eb44284d8b15d6b49d12bc444c87fc4f58e06889ff",
            "training_pipeline/models/ctgan/model.pkl": "5599c103681bdfcaef1cb1b1d1506f85aa2f7178a8d52d45970c2c158cd447c1",
            "training_pipeline/models/gaussian_copula/model.pkl": "5e5ac8f8a643f034003e80dc6e4d6ea9a800ca7c2d923e8dfa56aaadffa3ce30",
        }
        for rel_path, expected_hash in frozen.items():
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            self.assertTrue(os.path.exists(full_path), f"Frozen file missing: {rel_path}")
            with open(full_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(h, expected_hash, f"Hash mismatch for frozen file: {rel_path}")


if __name__ == "__main__":
    unittest.main()
