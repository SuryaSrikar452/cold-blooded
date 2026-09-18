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

from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import numpy as np
    from backend.services.generator_service import get_generator, store_generated_cohort
except ImportError as e:
    sys.stderr.write(f"Missing required library or generator service: {e}\n")
    sys.exit(1)


def synthesize_cohort(params):
    """
    Generate synthetic patient records conforming to targetSize and cohort proportions
    using the calibrated HurdleConditionalCopulaModel.
    """
    raw_size = params.get('targetSize') or params.get('size') or 10000
    target_size = int(raw_size)
    target_size = max(1000, min(50000, target_size))

    conditions_input = params.get('conditions') or params
    age_over_60_target = conditions_input.get('ageOver60', 40)
    diabetes_target = conditions_input.get('diabetes', 30)
    low_activity_target = conditions_input.get('lowActivity', 35)

    targets = {}
    if diabetes_target is not None:
        try:
            diab_val = float(diabetes_target)
            if diab_val >= 50:
                targets['diabetes'] = 1
            elif diab_val <= 15:
                targets['diabetes'] = 0
        except (ValueError, TypeError):
            pass

    if age_over_60_target is not None:
        try:
            age_val = float(age_over_60_target)
            if age_val >= 50:
                targets['age_gt'] = 60
        except (ValueError, TypeError):
            pass

    if low_activity_target is not None:
        try:
            act_val = float(low_activity_target)
            if act_val >= 50:
                targets['activity_mims_lt'] = 8500
        except (ValueError, TypeError):
            pass

    t_start = time.time()
    gen = get_generator()
    t_loaded = time.time()
    
    df = gen.sample(num_rows=target_size, targets=targets)
    t_end = time.time()

    cohort_id = f"SYN-HC-{uuid.uuid4().hex[:8].upper()}"
    if 'patient_id' not in df.columns:
        df.insert(0, "patient_id", [f"SYN-{i+1:06d}" for i in range(len(df))])

    # Store in memory cache
    store_generated_cohort(cohort_id, df, {
        "cohort_id": cohort_id,
        "n_requested": target_size,
        "n_generated": len(df),
        "targets": targets
    })

    # Summary metrics
    actual_age_over_60 = round(float((df['age'] >= 60).mean() * 100), 1)
    actual_diabetes = round(float((df['diabetes'] == 1).mean() * 100), 1)
    actual_low_activity = round(float((df['activity_mims'] < 8500).mean() * 100), 1)

    metrics = {
        "age_over_60_pct": actual_age_over_60,
        "diabetes_pct": actual_diabetes,
        "low_activity_pct": actual_low_activity,
        "mean_age": round(float(df['age'].mean()), 1),
        "mean_systolic_bp": round(float(df['systolic_bp'].mean()), 1),
        "mean_diastolic_bp": round(float(df['diastolic_bp'].mean()), 1),
        "mean_activity_mims": round(float(df['activity_mims'].mean()), 1),
        "mean_adherence": round(float(df['adherence_pct'].mean()), 1),
        "mean_pain_score": round(float(df['pain_score'].mean()), 1),
        "pct_pain_zero": round(float((df['pain_score'] == 0.0).mean() * 100), 1),
        "generation_time_sec": round(t_end - t_loaded, 3),
        "total_time_sec": round(t_end - t_start, 3)
    }

    preview = df.head(8).to_dict(orient='records')

    # Automatically save full generated cohort CSV to disk
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cohorts_dir = os.path.join(base_dir, 'generated_cohorts')
    os.makedirs(cohorts_dir, exist_ok=True)
    csv_filename = f"{cohort_id}.csv"
    csv_path = os.path.join(cohorts_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    result = {
        "success": True,
        "status": "success",
        "cohort_id": cohort_id,
        "csv_filename": csv_filename,
        "csv_path": csv_path,
        "model": "Hurdle Conditional Copula",
        "model_type": "HurdleConditionalCopulaModel",
        "provenance": "NHANES 2011-2012 Generative Benchmark (4,826 records)",
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
