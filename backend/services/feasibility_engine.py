from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = BASE_DIR / "final_training_data" / "nhanes_generative_train.csv"

_TRAIN_DF = None

def get_reference_df() -> pd.DataFrame:
    global _TRAIN_DF
    if _TRAIN_DF is None:
        if TRAIN_CSV.exists():
            _TRAIN_DF = pd.read_csv(TRAIN_CSV)
        else:
            raise FileNotFoundError(f"Training dataset not found at {TRAIN_CSV}")
    return _TRAIN_DF

def evaluate_feasibility(filters: Dict[str, Any]) -> Tuple[bool, float, str]:
    """
    Evaluates empirical joint feasibility of user-requested cohort constraints
    against the real training population (NHANES N=4,826).
    Returns (feasibility_warning: bool, estimated_joint_prevalence_pct: float, explanation: str).
    """
    df = get_reference_df()
    n_total = len(df)
    mask = pd.Series(True, index=df.index)

    constraints_described = []

    # 1. Diabetes
    if "diabetes" in filters:
        d_val = filters["diabetes"]
        if d_val in (1, 1.0, "1", "diabetic", "yes"):
            mask &= (df["diabetes"] == 1)
            constraints_described.append("Diabetes = Yes (baseline: 10.0%)")
        elif d_val in (0, 0.0, "0", "non-diabetic", "no"):
            mask &= (df["diabetes"] == 0)
            constraints_described.append("Diabetes = No (baseline: 90.0%)")

    # 2. Age
    if "age_gt" in filters:
        gt = int(filters["age_gt"])
        mask &= (df["age"] > gt)
        constraints_described.append(f"Age > {gt}")
    elif "ageOver60" in filters:
        pct = float(filters["ageOver60"])
        if pct > 75.0:
            mask &= (df["age"] >= 60)
            constraints_described.append("Age >= 60 (high prevalence target)")

    if "age_lt" in filters:
        lt = int(filters["age_lt"])
        mask &= (df["age"] < lt)
        constraints_described.append(f"Age < {lt}")

    # 3. Blood pressure
    if "systolic_bp_gt" in filters:
        sbp = int(filters["systolic_bp_gt"])
        mask &= (df["systolic_bp"] > sbp)
        constraints_described.append(f"Systolic BP > {sbp} mmHg")

    # 4. Physical Activity
    if "activity_mims_lt" in filters:
        mims = float(filters["activity_mims_lt"])
        mask &= (df["activity_mims"] < mims)
        constraints_described.append(f"Physical Activity < {int(mims)} MIMS")
    elif "lowActivity" in filters:
        pct = float(filters["lowActivity"])
        if pct > 65.0:
            mask &= (df["activity_mims"] < 8500)
            constraints_described.append("Low Physical Activity (high prevalence target)")

    # 5. Pain score
    if "pain_score_gt" in filters:
        p = float(filters["pain_score_gt"])
        mask &= (df["pain_score"] > p)
        constraints_described.append(f"Pain Score > {p}")

    # 6. Medications
    if "n_medications_ge" in filters:
        meds = int(filters["n_medications_ge"])
        mask &= (df["n_medications"] >= meds)
        constraints_described.append(f"Medications >= {meds}")

    matching_count = int(mask.sum())
    prevalence_pct = round((matching_count / float(n_total)) * 100.0, 2)

    # Threshold for extreme constraint warning is < 2.0% joint representation
    if matching_count == 0:
        warning = True
        explanation = (
            f"Zero exact matches found in the 4,826 NHANES reference records for this joint combination "
            f"({', '.join(constraints_described)}). Hurdle Copula conditional resampling will extrapolate from marginals, "
            f"which may lead to lower empirical target adherence."
        )
    elif prevalence_pct < 2.5:
        warning = True
        explanation = (
            f"This combination ({', '.join(constraints_described)}) represents only {prevalence_pct}% "
            f"({matching_count}/{n_total} records) of the reference population. Synthetic sampling is feasible, "
            f"but tail extrapolation may affect precision."
        )
    else:
        warning = False
        explanation = (
            f"Well-supported clinical profile ({prevalence_pct}% reference prevalence, "
            f"{matching_count} observed records in NHANES training corpus)."
        )

    return warning, prevalence_pct, explanation
