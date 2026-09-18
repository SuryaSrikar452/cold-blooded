import os
import sys
import time
import json
import hashlib
import platform
import pandas as pd
import numpy as np
import scipy
import sklearn
import pyarrow
import sdv
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

RANDOM_SEED = 42

def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_environment_info():
    cpu_info = platform.processor() or "AMD64/x86_64"
    env_str = (
        f"Python Version: {sys.version.split()[0]}\n"
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"CPU: {cpu_info}\n"
        f"SDV Version: {sdv.__version__}\n"
        f"pandas Version: {pd.__version__}\n"
        f"numpy Version: {np.__version__}\n"
        f"scipy Version: {scipy.__version__}\n"
        f"scikit-learn Version: {sklearn.__version__}\n"
        f"pyarrow Version: {pyarrow.__version__}\n"
    )
    req_str = (
        f"python=={sys.version.split()[0]}\n"
        f"sdv=={sdv.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"numpy=={np.__version__}\n"
        f"scipy=={scipy.__version__}\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"pyarrow=={pyarrow.__version__}\n"
    )
    return env_str, req_str

def validate_dataframe(df, is_synthetic=False):
    expected_cols = [
        'age', 'sex', 'diabetes', 'systolic_bp', 'diastolic_bp',
        'activity_mims', 'n_medications', 'adherence_pct', 'pain_score'
    ]
    
    if list(df.columns) != expected_cols:
        return False, f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}"
    
    if df.isna().sum().sum() > 0:
        return False, f"NaN values detected: {df.isna().sum().to_dict()}"
    
    if np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0:
        return False, "Infinite values detected."
    
    invalid_reasons = []
    
    invalid_sex = ~df['sex'].isin([1, 2])
    if invalid_sex.sum() > 0:
        invalid_reasons.append(f"Invalid sex values count: {invalid_sex.sum()}")
        
    invalid_diabetes = ~df['diabetes'].isin([0, 1])
    if invalid_diabetes.sum() > 0:
        invalid_reasons.append(f"Invalid diabetes values count: {invalid_diabetes.sum()}")
        
    invalid_age = (df['age'] < 8) | (df['age'] > 80)
    if invalid_age.sum() > 0:
        invalid_reasons.append(f"Invalid age values count: {invalid_age.sum()}")
        
    invalid_sbp = df['systolic_bp'] <= 0
    if invalid_sbp.sum() > 0:
        invalid_reasons.append(f"SBP <= 0 count: {invalid_sbp.sum()}")
        
    invalid_dbp = df['diastolic_bp'] <= 0
    if invalid_dbp.sum() > 0:
        invalid_reasons.append(f"DBP <= 0 count: {invalid_dbp.sum()}")
        
    invalid_pulse = df['diastolic_bp'] >= df['systolic_bp']
    if invalid_pulse.sum() > 0:
        invalid_reasons.append(f"DBP >= SBP count: {invalid_pulse.sum()}")
        
    invalid_activity = df['activity_mims'] < 0
    if invalid_activity.sum() > 0:
        invalid_reasons.append(f"activity_mims < 0 count: {invalid_activity.sum()}")
        
    invalid_meds = df['n_medications'] < 0
    if invalid_meds.sum() > 0:
        invalid_reasons.append(f"n_medications < 0 count: {invalid_meds.sum()}")
        
    invalid_adh = (df['adherence_pct'] < 0) | (df['adherence_pct'] > 100)
    if invalid_adh.sum() > 0:
        invalid_reasons.append(f"adherence_pct out of [0, 100] count: {invalid_adh.sum()}")
        
    invalid_pain = (df['pain_score'] < 0) | (df['pain_score'] > 10)
    if invalid_pain.sum() > 0:
        invalid_reasons.append(f"pain_score out of [0, 10] count: {invalid_pain.sum()}")
        
    if invalid_reasons:
        return False, "; ".join(invalid_reasons)
        
    return True, "All validation assertions passed successfully."

def main():
    print("============================================================")
    print("SH-405 GAUSSIAN COPULA BASELINE: TRAINING & GENERATION")
    print("============================================================")
    
    # 1. Paths & Hash Pre-Verification (Relative paths for portability)
    train_path = 'final_training_data/nhanes_generative_train.csv'
    holdout_path = 'final_training_data/nhanes_real_holdout.csv'
    schema_path = 'final_training_data/final_schema.json'
    
    print(f"1. Training dataset path: {train_path}")
    print(f"2. Holdout dataset path:  {holdout_path}")
    
    train_hash_before = calculate_sha256(train_path)
    holdout_hash_before = calculate_sha256(holdout_path)
    
    print(f"3. SHA-256 Hash (Train):   {train_hash_before}")
    print(f"   SHA-256 Hash (Holdout): {holdout_hash_before}")
    
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)
    
    print(f"4. Row Counts - Train: {len(train_df)}, Holdout: {len(holdout_df)}")
    print(f"5. Columns: {list(train_df.columns)}")
    
    env_str, req_str = get_environment_info()
    with open('gaussian_copula_pipeline/environment.txt', 'w') as f:
        f.write(env_str)
    with open('gaussian_copula_pipeline/requirements.txt', 'w') as f:
        f.write(req_str)
    print("Recorded environment.txt and requirements.txt.")
    
    with open(schema_path, 'r') as f:
        schema_json = json.load(f)
        
    expected_features = set(schema_json['features'].keys())
    actual_features = set(train_df.columns)
    
    if expected_features != actual_features:
        print(f"ERROR: Schema mismatch! Expected {expected_features}, got {actual_features}")
        sys.exit(1)
    print("Schema verification against final_schema.json: PASSED.")
    
    valid_train, reason = validate_dataframe(train_df, is_synthetic=False)
    if not valid_train:
        print(f"ERROR: Training dataset pre-validation failed! Cause: {reason}")
        sys.exit(1)
    print("Pre-training validation of training CSV: PASSED.")
    
    overlap = pd.merge(train_df, holdout_df, how='inner')
    if len(overlap) > 0:
        print(f"ERROR: Train and holdout datasets have {len(overlap)} exact matching rows!")
        sys.exit(1)
    print(f"Train/Holdout separation check: PASSED (0 overlapping row profiles out of {len(holdout_df)} holdout rows).")
    
    metadata = Metadata.detect_from_dataframe(data=train_df, table_name='nhanes')
    metadata.update_column(column_name='sex', sdtype='categorical')
    metadata.update_column(column_name='diabetes', sdtype='categorical')
    
    config = {
        "model_type": "GaussianCopulaSynthesizer",
        "sdv_version": sdv.__version__,
        "random_seed": RANDOM_SEED,
        "features": list(train_df.columns),
        "continuous_columns": [
            "age", "systolic_bp", "diastolic_bp", "activity_mims",
            "n_medications", "adherence_pct", "pain_score"
        ],
        "categorical_columns": ["sex", "diabetes"],
        "default_distribution": "gaussian"
    }
    
    with open('gaussian_copula_pipeline/config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Starting Gaussian Copula training...")
    start_time = time.time()
    
    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    synthesizer.fit(train_df)
    
    end_time = time.time()
    training_duration = round(end_time - start_time, 4)
    print(f"Training completed in {training_duration} seconds.")
    
    model_dir = 'gaussian_copula_pipeline/models/gaussian_copula'
    synthesizer.save(os.path.join(model_dir, 'model.pkl'))
    
    model_metadata = {
        "model_name": "GaussianCopulaSynthesizer",
        "training_rows": len(train_df),
        "num_features": len(train_df.columns),
        "continuous_columns": config["continuous_columns"],
        "categorical_columns": config["categorical_columns"],
        "sdv_version": sdv.__version__,
        "random_seed": RANDOM_SEED,
        "training_duration_seconds": training_duration,
        "training_start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "training_end_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time))
    }
    
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=4)
    print("Saved model and model_metadata.json.")
    
    print(f"Generating {len(train_df)} synthetic rows...")
    gen_start = time.time()
    
    syn_df = synthesizer.sample(num_rows=len(train_df))
    gen_end = time.time()
    generation_duration = round(gen_end - gen_start, 4)
    print(f"Generation completed in {generation_duration} seconds.")
    
    syn_df = syn_df[list(train_df.columns)]
    
    csv_syn_path = 'gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.csv'
    parquet_syn_path = 'gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.parquet'
    
    syn_df.to_csv(csv_syn_path, index=False)
    syn_df.to_parquet(parquet_syn_path, index=False)
    print(f"Saved synthetic datasets to {csv_syn_path} and {parquet_syn_path}.")
    
    print("Performing Post-Generation Validation...")
    
    valid_mask = (
        syn_df['sex'].isin([1, 2]) &
        syn_df['diabetes'].isin([0, 1]) &
        (syn_df['age'] >= 8) & (syn_df['age'] <= 80) &
        (syn_df['systolic_bp'] > 0) &
        (syn_df['diastolic_bp'] > 0) &
        (syn_df['diastolic_bp'] < syn_df['systolic_bp']) &
        (syn_df['activity_mims'] >= 0) &
        (syn_df['n_medications'] >= 0) &
        (syn_df['adherence_pct'] >= 0) & (syn_df['adherence_pct'] <= 100) &
        (syn_df['pain_score'] >= 0) & (syn_df['pain_score'] <= 10) &
        (~syn_df.isna().any(axis=1))
    )
    
    valid_rows = int(valid_mask.sum())
    invalid_rows = int(len(syn_df) - valid_rows)
    validity_pct = round((valid_rows / len(syn_df)) * 100, 2)
    
    print(f"Post-Generation Validation Results:")
    print(f"  Total Synthetic Rows: {len(syn_df)}")
    print(f"  Valid Rows:           {valid_rows}")
    print(f"  Invalid Rows:         {invalid_rows}")
    print(f"  Validity Percentage:  {validity_pct}%")
    
    syn_hash = calculate_sha256(csv_syn_path)
    
    training_log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_file_path": train_path,
        "training_file_hash": train_hash_before,
        "holdout_file_path": holdout_path,
        "holdout_file_hash": holdout_hash_before,
        "training_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "synthetic_rows": len(syn_df),
        "feature_list": list(train_df.columns),
        "random_seed": RANDOM_SEED,
        "python_version": sys.version.split()[0],
        "sdv_version": sdv.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pyarrow_version": pyarrow.__version__,
        "model_configuration": config,
        "training_time_seconds": training_duration,
        "generation_time_seconds": generation_duration,
        "model_path": os.path.join(model_dir, 'model.pkl'),
        "synthetic_dataset_csv_path": csv_syn_path,
        "synthetic_dataset_parquet_path": parquet_syn_path,
        "synthetic_dataset_hash": syn_hash,
        "post_generation_valid_rows": valid_rows,
        "post_generation_invalid_rows": invalid_rows,
        "post_generation_validity_percentage": validity_pct,
        "holdout_hash_after_experiment": calculate_sha256(holdout_path)
    }
    
    with open('gaussian_copula_pipeline/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=4)
    print("Saved training_log.json.")
    print("TRAINING & GENERATION PHASE COMPLETE.")

if __name__ == '__main__':
    main()
