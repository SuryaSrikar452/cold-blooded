import os
import uuid
import pandas as pd
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_representativeness_audit(synthetic_df, real_df=None, reference_name="NHANES 2011-2012 Benchmark", tolerance_pct=2.5):
    """
    Performs a neutral representativeness & subgroup bias audit:
    Compares synthetic subgroup percentages against real/reference population benchmark percentages.
    """
    if real_df is None:
        real_csv = BASE_DIR / "final_training_data" / "nhanes_generative_train.csv"
        if not real_csv.exists():
            real_csv = BASE_DIR.parent / "final_training_data" / "nhanes_generative_train.csv"
        real_df = pd.read_csv(real_csv)
        
    audit_id = f"audit_{uuid.uuid4().hex[:8]}"
    subgroup_results = []
    
    # Define subgroup definitions (lambda evaluation)
    subgroups = [
        ("Age Band: 8-17 yrs", lambda df: (df['age'] >= 8) & (df['age'] <= 17)),
        ("Age Band: 18-39 yrs", lambda df: (df['age'] >= 18) & (df['age'] <= 39)),
        ("Age Band: 40-64 yrs", lambda df: (df['age'] >= 40) & (df['age'] <= 64)),
        ("Age Band: 65-80 yrs", lambda df: (df['age'] >= 65) & (df['age'] <= 80)),
        ("Sex: Male (1)", lambda df: df['sex'] == 1),
        ("Sex: Female (2)", lambda df: df['sex'] == 2),
        ("Diabetes: Non-Diabetic (0)", lambda df: df['diabetes'] == 0),
        ("Diabetes: Diagnosed Diabetic (1)", lambda df: df['diabetes'] == 1),
        ("Intersection: Age >= 60 & Diabetic", lambda df: (df['age'] >= 60) & (df['diabetes'] == 1)),
        ("Intersection: Age < 60 & Diabetic", lambda df: (df['age'] < 60) & (df['diabetes'] == 1)),
        ("Intersection: Female Seniors (Age >= 65)", lambda df: (df['age'] >= 65) & (df['sex'] == 2)),
        ("Intersection: Male Seniors (Age >= 65)", lambda df: (df['age'] >= 65) & (df['sex'] == 1)),
    ]
    
    for label, mask_func in subgroups:
        real_pct = float(mask_func(real_df).mean() * 100)
        syn_pct = float(mask_func(synthetic_df).mean() * 100)
        gap = float(syn_pct - real_pct)
        abs_gap = float(abs(gap))
        
        if abs_gap <= tolerance_pct:
            status = "within_tolerance"
        elif gap > tolerance_pct:
            status = "over_represented"
        else:
            status = "under_represented"
            
        subgroup_results.append({
            "audit_id": audit_id,
            "reference": reference_name,
            "subgroup": label,
            "real_pct": round(real_pct, 2),
            "synthetic_pct": round(syn_pct, 2),
            "gap_pct_points": round(gap, 2),
            "abs_gap": round(abs_gap, 2),
            "status": status,
            "tolerance_pct": tolerance_pct
        })
        
    return {
        "audit_id": audit_id,
        "reference_dataset": reference_name,
        "total_subgroups_audited": len(subgroup_results),
        "within_tolerance_count": sum(1 for r in subgroup_results if r["status"] == "within_tolerance"),
        "flagged_subgroups_count": sum(1 for r in subgroup_results if r["status"] != "within_tolerance"),
        "subgroup_audits": subgroup_results
    }
