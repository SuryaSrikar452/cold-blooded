import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = BASE_DIR / "final_training_data" / "nhanes_generative_train.csv"

def compute_copula_counterfactual(patient_dict, modified_var, new_value, model_path=None):
    """
    Computes a copula-based conditional counterfactual patient profile.
    Uses conditional multivariate normal distribution in Gaussian Copula space:
    Z_other | Z_mod ~ N( mu_other + Sigma_{other,mod} Sigma_{mod,mod}^{-1} (Z_mod - mu_mod), ... )
    """
    # Baseline patient features
    cols = ['age', 'sex', 'diabetes', 'systolic_bp', 'diastolic_bp', 'activity_mims', 'n_medications', 'adherence_pct', 'pain_score']
    
    if modified_var not in cols:
        raise ValueError(f"Variable '{modified_var}' not recognized in schema.")
        
    orig_patient = {k: float(patient_dict.get(k, 0)) for k in cols}
    cf_patient = orig_patient.copy()
    cf_patient[modified_var] = float(new_value)
    
    # Load training dataset to estimate empiric marginals and correlation matrix
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Standardize to normal Z-scores (empirical CDF -> Gaussian quantile)
    # Z_i = Phi^{-1}( (Rank(X_i) - 0.5) / N )
    Z_train = pd.DataFrame(index=train_df.index, columns=cols)
    for col in cols:
        ranks = train_df[col].rank(method='average')
        u = (ranks - 0.5) / float(len(train_df))
        # clip u to avoid +-inf
        u = np.clip(u, 1e-5, 1.0 - 1e-5)
        Z_train[col] = norm.ppf(u)
        
    # Covariance matrix in Z space
    Sigma = Z_train.corr().values
    
    # Index of modified var
    mod_idx = cols.index(modified_var)
    other_indices = [i for i in range(len(cols)) if i != mod_idx]
    
    # Convert modified var original -> u -> Z
    ranks_mod = train_df[modified_var].rank(method='average')
    # Empirical percentile of new_value
    pct_new = (train_df[modified_var] <= new_value).mean()
    pct_new = np.clip(pct_new, 1e-5, 1.0 - 1e-5)
    z_mod_val = norm.ppf(pct_new)
    
    # Original Z values for patient
    z_orig = np.zeros(len(cols))
    for i, c in enumerate(cols):
        pct = (train_df[c] <= orig_patient[c]).mean()
        pct = np.clip(pct, 1e-5, 1.0 - 1e-5)
        z_orig[i] = norm.ppf(pct)
        
    # Conditional mean update for other variables:
    # mu_other|mod = mu_other + Sigma_{other, mod} / Sigma_{mod, mod} * (z_mod - z_mod_orig)
    sigma_mod_mod = Sigma[mod_idx, mod_idx]
    sigma_other_mod = Sigma[other_indices, mod_idx]
    
    z_diff = z_mod_val - z_orig[mod_idx]
    z_other_update = z_orig[other_indices] + (sigma_other_mod / sigma_mod_mod) * z_diff
    
    # Map back Z_other -> u -> original feature value quantile
    changed_vars = []
    for idx_in_other, col_idx in enumerate(other_indices):
        c_name = cols[col_idx]
        z_val = z_other_update[idx_in_other]
        u_val = norm.cdf(z_val)
        u_val = np.clip(u_val, 0.0, 1.0)
        
        # Empirical quantile mapping
        val_mapped = float(np.quantile(train_df[c_name], u_val))
        
        # If integer column, round
        if c_name in ['sex', 'diabetes', 'n_medications', 'age', 'systolic_bp', 'diastolic_bp', 'activity_mims']:
            val_mapped = float(round(val_mapped))
            
        cf_patient[c_name] = val_mapped
        if abs(val_mapped - orig_patient[c_name]) > 1e-4:
            changed_vars.append(c_name)
            
    # Enforce clinical constraints
    if cf_patient['diastolic_bp'] >= cf_patient['systolic_bp']:
        cf_patient['diastolic_bp'] = max(40.0, cf_patient['systolic_bp'] - 10.0)
        
    cf_patient['adherence_pct'] = float(np.clip(cf_patient['adherence_pct'], 0.0, 100.0))
    cf_patient['pain_score'] = float(np.clip(cf_patient['pain_score'], 0.0, 10.0))
    cf_patient['activity_mims'] = float(max(0.0, cf_patient['activity_mims']))
    
    # Euclidean distance between normalized original and counterfactual
    scale_vector = train_df[cols].std().values
    scale_vector = np.where(scale_vector == 0, 1.0, scale_vector)
    orig_vec = np.array([orig_patient[c] for c in cols])
    cf_vec = np.array([cf_patient[c] for c in cols])
    dist = float(np.linalg.norm((orig_vec - cf_vec) / scale_vector))
    
    return {
        "patient_id": patient_dict.get("patient_id", "SYN-000001"),
        "modified_variable": modified_var,
        "new_value": new_value,
        "original": orig_patient,
        "counterfactual": cf_patient,
        "changed_variables": changed_vars,
        "normalized_distance": round(dist, 4),
        "disclaimer": "These are model-generated counterfactuals based on Gaussian Copula conditional distributions, NOT clinical predictions."
    }
