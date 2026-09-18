"""
gaussian_copula_final/train_final.py
------------------------------------
Production training and cohort generation script for GaussianCopulaFinal.

DATA HYGIENE GUARANTEE:
Strictly fits on final_training_data/nhanes_generative_train.csv (N=4,826).
The real holdout dataset (nhanes_real_holdout.csv) is NEVER read or imported.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaussian_copula_final.copula_final import GaussianCopulaFinal, APPROVED_FEATURES


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("SH-405: TRAINING FINAL HARDENED GAUSSIAN COPULA GENERATOR")
    print("=" * 80)

    # 1. Load configuration
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded configuration from {config_path}")

    # 2. Strict Zero-Leakage Data Hygiene: Only load training data
    train_path = os.path.join(PROJECT_ROOT, "final_training_data", "nhanes_generative_train.csv")
    assert os.path.exists(train_path), f"Training data missing at {train_path}"
    df_train = pd.read_csv(train_path)
    print(f"Loaded training cohort: {len(df_train)} rows, {len(df_train.columns)} columns")
    print(f"Columns: {list(df_train.columns)}")
    assert list(df_train.columns) == APPROVED_FEATURES

    # 3. Fit GaussianCopulaFinal
    print("\nFitting GaussianCopulaFinal on training data...")
    t0_fit = time.time()
    model = GaussianCopulaFinal(config_dict=config)
    model.fit(df_train)
    fit_duration = time.time() - t0_fit
    print(f"Model fitting completed in {fit_duration:.3f} seconds.")

    # 4. Report Copula PSD Diagnostics
    diag = model.psd_diagnostics
    print("\n--- COPULA DEPENDENCE & PSD CORRECTION DIAGNOSTICS ---")
    print(f"Min eigenvalue before PSD projection: {diag['min_eigenvalue_before']:.6f}")
    print(f"Number of negative eigenvalues before: {diag['n_negative_eigenvalues_before']}")
    print(f"Frobenius norm of PSD correction:      {diag['frobenius_norm_correction']:.6f}")
    print(f"Max element-wise correlation change:  {diag['max_element_correction']:.6f}")
    print(f"Min eigenvalue after PSD projection:   {diag['min_eigenvalue_after']:.6f}")

    # 5. Save model
    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")
    model.save(model_path)
    model_sha256 = compute_sha256(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Model SHA-256: {model_sha256}")

    # 6. Generate synthetic cohort
    n_synth = int(config.get("n_synthetic_samples", 4826))
    random_seed = int(config.get("random_seed", 42))
    print(f"\nGenerating {n_synth} synthetic records with seed {random_seed}...")
    t0_gen = time.time()
    df_synth = model.sample(n=n_synth, random_state=random_seed)
    gen_duration = time.time() - t0_gen
    print(f"Synthetic cohort generation completed in {gen_duration:.3f} seconds.")

    # 7. Save synthetic datasets
    synthetic_dir = os.path.join(SCRIPT_DIR, "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)
    csv_path = os.path.join(synthetic_dir, "gaussian_copula_final_synthetic.csv")
    parquet_path = os.path.join(synthetic_dir, "gaussian_copula_final_synthetic.parquet")

    df_synth.to_csv(csv_path, index=False)
    df_synth.to_parquet(parquet_path, index=False)
    csv_sha256 = compute_sha256(csv_path)
    parquet_sha256 = compute_sha256(parquet_path)

    print(f"\nSynthetic CSV saved to:     {csv_path}")
    print(f"Synthetic CSV SHA-256:     {csv_sha256}")
    print(f"Synthetic Parquet saved to: {parquet_path}")
    print(f"Synthetic Parquet SHA-256: {parquet_sha256}")

    # 8. Save metadata.json
    metadata = {
        "model_name": config["model_name"],
        "version": config["version"],
        "training_dataset": "final_training_data/nhanes_generative_train.csv",
        "training_rows": len(df_train),
        "holdout_accessed_during_training": False,
        "synthetic_rows": len(df_synth),
        "random_seed": random_seed,
        "training_runtime_seconds": round(fit_duration, 4),
        "generation_runtime_seconds": round(gen_duration, 4),
        "model_sha256": model_sha256,
        "synthetic_csv_sha256": csv_sha256,
        "synthetic_parquet_sha256": parquet_sha256,
        "psd_diagnostics": diag,
        "marginal_summary": {
            k: {pk: pv for pk, pv in v.items() if pk not in ["cdf", "values"]}
            for k, v in model.marginal_params.items()
        },
    }
    meta_path = os.path.join(models_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to:       {meta_path}")

    print("\n" + "=" * 80)
    print("TRAINING & COHORT GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
