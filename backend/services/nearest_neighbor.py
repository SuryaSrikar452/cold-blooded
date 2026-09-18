import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = BASE_DIR / "final_training_data" / "nhanes_generative_train.csv"

def find_nearest_real_neighbor(patient_id, synthetic_df=None, real_df=None):
    """
    Finds the closest seed record for a given synthetic patient using standardized distance.
    Reuses standardized distance metric consistent with DCR and MIA privacy engines.
    """
    if real_df is None:
        real_df = pd.read_csv(TRAIN_CSV)
    if synthetic_df is None:
        try:
            from backend.services.generator_service import get_cohort_dataframe
            synthetic_df = get_cohort_dataframe()
        except Exception:
            synthetic_df = real_df.copy()
            
    continuous_cols = ['age', 'systolic_bp', 'diastolic_bp', 'activity_mims', 'n_medications', 'adherence_pct', 'pain_score']
    
    # Locate target synthetic patient index
    if "patient_id" in synthetic_df.columns:
        idx_matches = synthetic_df.index[synthetic_df["patient_id"] == patient_id].tolist()
        if not idx_matches:
            target_idx = 0
            patient_id = synthetic_df.iloc[0].get("patient_id", "SYN-000001")
        else:
            target_idx = idx_matches[0]
    else:
        try:
            target_idx = int(patient_id.replace("SYN-", "").replace("SYN-EDGE-", "")) - 1
            target_idx = max(0, min(target_idx, len(synthetic_df) - 1))
        except ValueError:
            target_idx = 0

    target_syn_row = synthetic_df.iloc[target_idx]
    
    # Feature scaling based on Real Dataset
    scaler = StandardScaler()
    scaler.fit(real_df[continuous_cols])
    
    def transform_features(df):
        cont_sc = scaler.transform(df[continuous_cols])
        # Explicit categorical one-hot vectors
        sex_1 = (df['sex'].values == 1).astype(float).reshape(-1, 1)
        sex_2 = (df['sex'].values == 2).astype(float).reshape(-1, 1)
        diab_0 = (df['diabetes'].values == 0).astype(float).reshape(-1, 1)
        diab_1 = (df['diabetes'].values == 1).astype(float).reshape(-1, 1)
        return np.hstack([cont_sc, sex_1, sex_2, diab_0, diab_1])

    X_real = transform_features(real_df)
    
    target_syn_df = pd.DataFrame([target_syn_row[continuous_cols + ['sex', 'diabetes']]])
    X_target = transform_features(target_syn_df)
    
    # Fit Nearest Neighbors on Real Dataset
    nn = NearestNeighbors(n_neighbors=len(X_real), metric='euclidean')
    nn.fit(X_real)
    
    distances, indices = nn.kneighbors(X_target)
    
    nearest_dist = float(distances[0][0])
    nearest_real_idx = int(indices[0][0])
    nearest_real_row = real_df.iloc[nearest_real_idx]
    
    # Compute percentile of this distance across all synthetic-to-real distances
    X_all_syn = transform_features(synthetic_df[continuous_cols + ['sex', 'diabetes']])
    all_dists, _ = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_real).kneighbors(X_all_syn)
    pct = float((all_dists.flatten() <= nearest_dist).mean() * 100)
    
    real_id = f"REAL-SEED-{nearest_real_idx+1:04d}"
    
    return {
        "synthetic_patient_id": str(patient_id),
        "nearest_real_patient_id": real_id,
        "distance": round(nearest_dist, 4),
        "distance_percentile": round(pct, 2),
        "synthetic_patient": {k: (int(v) if isinstance(v, (np.int64, int)) else float(v) if isinstance(v, (np.float64, float)) else str(v)) for k, v in target_syn_row.to_dict().items()},
        "nearest_real_patient": {k: (int(v) if isinstance(v, (np.int64, int)) else float(v) if isinstance(v, (np.float64, float)) else str(v)) for k, v in nearest_real_row.to_dict().items()},
        "synthetic_record": {k: (int(v) if isinstance(v, (np.int64, int)) else float(v) if isinstance(v, (np.float64, float)) else str(v)) for k, v in target_syn_row.to_dict().items()},
        "nearest_real_record": {k: (int(v) if isinstance(v, (np.int64, int)) else float(v) if isinstance(v, (np.float64, float)) else str(v)) for k, v in nearest_real_row.to_dict().items()},
        "disclaimer": "This is a statistical nearest-neighbor comparison. It does not mean the synthetic patient represents or belongs to this real individual."
    }
