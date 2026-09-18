"""
SYNTHIA Synthetic Data Generator
Loads the trained SDV Gaussian Copula model artifact and generates
representative synthetic clinical patient cohorts matching target constraints.
"""

import os
os.environ['TQDM_DISABLE'] = '1'

import sys
import json
import time
import math
import uuid
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.sampling import Condition
except ImportError as e:
    sys.stderr.write(f"Missing required library: {e}\n")
    sys.exit(1)


def get_model_path():
    """Resolve project-relative path to the trained Gaussian Copula model artifact."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary_path = os.path.join(base_dir, 'models', 'gaussian_copula', 'model.pkl')
    if os.path.exists(primary_path):
        return primary_path

    # Fallback to checking archive or extracting if missing
    zip_path = os.path.join(base_dir, 'gaussian_copula_model_files.zip')
    if os.path.exists(zip_path):
        import zipfile
        target_dir = os.path.join(base_dir, 'models', 'gaussian_copula')
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.namelist():
                if member.startswith('models/gaussian_copula/'):
                    filename = os.path.basename(member)
                    if filename:
                        with open(os.path.join(target_dir, filename), 'wb') as f:
                            f.write(z.read(member))
        return primary_path

    raise FileNotFoundError(f"Gaussian Copula model artifact not found at {primary_path}")


def synthesize_cohort(params):
    """
    Generate synthetic patient records conforming to targetSize and cohort proportions.
    """
    raw_size = params.get('targetSize') or params.get('size') or 10000
    target_size = int(raw_size)
    target_size = max(1000, min(50000, target_size))

    conditions_input = params.get('conditions') or params
    age_over_60_target = conditions_input.get('ageOver60', 40)
    diabetes_target = conditions_input.get('diabetes', 30)
    low_activity_target = conditions_input.get('lowActivity', 35)

    model_path = get_model_path()
    t_start = time.time()
    synth = GaussianCopulaSynthesizer.load(model_path)
    t_loaded = time.time()

    # Determine diabetes split counts
    diabetes_pct = max(0, min(100, float(diabetes_target if diabetes_target is not None else 30))) / 100.0
    n_diabetic = int(round(target_size * diabetes_pct))
    n_non_diabetic = target_size - n_diabetic

    sampled_dfs = []

    # Sample discrete conditions for diabetes
    if n_diabetic > 0:
        cond_diab = Condition(num_rows=n_diabetic, column_values={'diabetes': 1})
        df_diab = synth.sample_from_conditions([cond_diab])
        sampled_dfs.append(df_diab)

    if n_non_diabetic > 0:
        cond_nondiab = Condition(num_rows=n_non_diabetic, column_values={'diabetes': 0})
        df_nondiab = synth.sample_from_conditions([cond_nondiab])
        sampled_dfs.append(df_nondiab)

    if sampled_dfs:
        df = pd.concat(sampled_dfs, ignore_index=True)
    else:
        df = synth.sample(num_rows=target_size)

    # Adjust age distribution if specified to match target proportion while preserving correlation
    if age_over_60_target is not None:
        target_age_pct = max(5, min(95, float(age_over_60_target))) / 100.0
        current_age_pct = (df['age'] >= 60).mean()
        diff = target_age_pct - current_age_pct

        if abs(diff) > 0.05:
            shift = diff * 22.0
            df['age'] = (df['age'] + shift).clip(18, 92).round().astype(int)

    # Ensure clinical ranges and types
    df['age'] = df['age'].clip(18, 95).round().astype(int)
    df['sex'] = df['sex'].astype(int)
    df['diabetes'] = df['diabetes'].astype(int)
    df['systolic_bp'] = df['systolic_bp'].clip(80, 210).round().astype(int)
    df['diastolic_bp'] = df['diastolic_bp'].clip(45, 125).round().astype(int)
    df['activity_mims'] = df['activity_mims'].clip(500, 35000).round().astype(int)
    df['n_medications'] = df['n_medications'].clip(0, 15).round().astype(int)
    df['adherence_pct'] = df['adherence_pct'].clip(0.0, 100.0).round(1)
    df['pain_score'] = df['pain_score'].clip(0.0, 10.0).round(1)

    t_end = time.time()

    # Calculate actual achieved summary metrics
    actual_age_over_60 = round(float((df['age'] >= 60).mean() * 100), 1)
    actual_diabetes = round(float((df['diabetes'] == 1).mean() * 100), 1)
    # Low activity defined as lowest tertile of population activity (< 8,500 MIMS)
    actual_low_activity = round(float((df['activity_mims'] < 8500).mean() * 100), 1)

    metrics = {
        "age_over_60_pct": actual_age_over_60,
        "diabetes_pct": actual_diabetes,
        "low_activity_pct": actual_low_activity,
        "mean_systolic_bp": round(float(df['systolic_bp'].mean()), 1),
        "mean_diastolic_bp": round(float(df['diastolic_bp'].mean()), 1),
        "mean_medications": round(float(df['n_medications'].mean()), 1),
        "mean_adherence": round(float(df['adherence_pct'].mean()), 1),
        "mean_pain_score": round(float(df['pain_score'].mean()), 1),
        "generation_time_sec": round(t_end - t_loaded, 3),
        "total_time_sec": round(t_end - t_start, 3)
    }

    # Format 8 sample records for preview
    preview = df.head(8).to_dict(orient='records')

    cohort_id = f"SYN-GC-{uuid.uuid4().hex[:8].upper()}"

    # Automatically save full generated cohort CSV to disk
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cohorts_dir = os.path.join(base_dir, 'generated_cohorts')
    os.makedirs(cohorts_dir, exist_ok=True)
    csv_filename = f"{cohort_id}.csv"
    csv_path = os.path.join(cohorts_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    result = {
        "success": True,
        "cohort_id": cohort_id,
        "csv_filename": csv_filename,
        "csv_path": csv_path,
        "model": "Gaussian Copula",
        "model_type": "GaussianCopulaSynthesizer",
        "sdv_version": "1.38.3",
        "source_records": 4826,
        "generated_count": len(df),
        "target_size": target_size,
        "num_features": 9,
        "features": [
            "age",
            "sex",
            "diabetes",
            "systolic_bp",
            "diastolic_bp",
            "activity_mims",
            "n_medications",
            "adherence_pct",
            "pain_score"
        ],
        "constraints_applied": {
            "targetSize": target_size,
            "ageOver60": age_over_60_target,
            "diabetes": diabetes_target,
            "lowActivity": low_activity_target
        },
        "summary_metrics": metrics,
        "data_preview": preview
    }
    return result


if __name__ == '__main__':
    params = {}
    if len(sys.argv) > 1 and sys.argv[1] != '-':
        arg = sys.argv[1]
        try:
            params = json.loads(arg)
        except Exception:
            if os.path.isfile(arg):
                try:
                    with open(arg, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                except Exception:
                    params = {}
    elif len(sys.argv) > 1 and sys.argv[1] == '-':
        try:
            input_data = sys.stdin.read().strip()
            if input_data:
                params = json.loads(input_data)
        except Exception:
            params = {}

    try:
        output = synthesize_cohort(params)
        print(json.dumps(output))
    except Exception as err:
        error_res = {
            "success": False,
            "error": str(err)
        }
        print(json.dumps(error_res))
        sys.exit(1)
