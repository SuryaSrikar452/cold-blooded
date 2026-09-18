import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix

RANDOM_SEED = 42

def extract_attack_features(records_df, synthetic_df, continuous_cols, categorical_cols):
    """
    Extracts MIA attack features for target records against a synthetic dataset:
    1. Nearest synthetic record distance (1-NN)
    2. Mean distance to k=5 nearest synthetic records (5-NN)
    3. Local synthetic density (count within radius r)
    4. Normalized 1-NN distance
    """
    scaler = StandardScaler()
    scaler.fit(synthetic_df[continuous_cols])
    
    def proc_features(df):
        cont_sc = scaler.transform(df[continuous_cols])
        sex_1 = (df['sex'].values == 1).astype(float).reshape(-1, 1)
        sex_2 = (df['sex'].values == 2).astype(float).reshape(-1, 1)
        diab_0 = (df['diabetes'].values == 0).astype(float).reshape(-1, 1)
        diab_1 = (df['diabetes'].values == 1).astype(float).reshape(-1, 1)
        return np.hstack([cont_sc, sex_1, sex_2, diab_0, diab_1])

    syn_proc = proc_features(synthetic_df)
    target_proc = proc_features(records_df)
    
    # 1-NN and 5-NN distances
    nn_5 = NearestNeighbors(n_neighbors=min(5, len(syn_proc)), metric='euclidean')
    nn_5.fit(syn_proc)
    distances, _ = nn_5.kneighbors(target_proc)
    
    dist_1nn = distances[:, 0]
    dist_5nn_mean = np.mean(distances, axis=1)
    
    # Local synthetic density
    nn_radius = NearestNeighbors(radius=1.0, metric='euclidean')
    nn_radius.fit(syn_proc)
    in_radius_counts = [len(idx) for idx in nn_radius.radius_neighbors(target_proc, return_distance=False)]
    density = np.array(in_radius_counts) / float(len(syn_proc))
    
    # Normalized 1-NN
    norm_dist_1nn = dist_1nn / (dist_5nn_mean + 1e-6)
    
    features_matrix = np.column_stack([
        dist_1nn,
        dist_5nn_mean,
        density,
        norm_dist_1nn
    ])
    return features_matrix

def run_membership_inference_attack(train_seed_df, holdout_seed_df, synthetic_df, continuous_cols, categorical_cols, attacker_type="both"):
    """
    Executes a black-box Membership Inference Attack (MIA):
    - Train seed records (label 1)
    - Holdout seed records (label 0)
    """
    X_train_seed = extract_attack_features(train_seed_df, synthetic_df, continuous_cols, categorical_cols)
    X_holdout_seed = extract_attack_features(holdout_seed_df, synthetic_df, continuous_cols, categorical_cols)
    
    y_train_seed = np.ones(len(train_seed_df))
    y_holdout_seed = np.zeros(len(holdout_seed_df))
    
    X_attack = np.vstack([X_train_seed, X_holdout_seed])
    y_attack = np.hstack([y_train_seed, y_holdout_seed])
    
    np.random.seed(RANDOM_SEED)
    perm = np.random.permutation(len(y_attack))
    X_attack = X_attack[perm]
    y_attack = y_attack[perm]
    
    split_idx = int(0.7 * len(y_attack))
    X_tr_att, X_te_att = X_attack[:split_idx], X_attack[split_idx:]
    y_tr_att, y_te_att = y_attack[:split_idx], y_attack[split_idx:]
    
    models = {}
    if attacker_type in ["logistic_regression", "both"]:
        clf_lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        clf_lr.fit(X_tr_att, y_tr_att)
        models["logistic_regression"] = clf_lr
        
    if attacker_type in ["gradient_boosting", "both"]:
        clf_gb = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
        clf_gb.fit(X_tr_att, y_tr_att)
        models["gradient_boosting"] = clf_gb
        
    results = {}
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_te_att)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        auc = float(roc_auc_score(y_te_att, y_proba))
        prec = float(precision_score(y_te_att, y_pred, zero_division=0))
        rec = float(recall_score(y_te_att, y_pred, zero_division=0))
        cm = confusion_matrix(y_te_att, y_pred).tolist()
        
        fpr, tpr, _ = roc_curve(y_te_att, y_proba)
        idx = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
        roc_pts = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]
        
        top50_idx = np.argsort(y_proba)[::-1][:50]
        prec_at_50 = float(np.mean(y_te_att[top50_idx] == 1)) if len(top50_idx) > 0 else 0.0
        rec_at_50 = float(np.sum(y_te_att[top50_idx] == 1) / max(1, np.sum(y_te_att == 1)))
        acc = float(np.mean(y_pred == y_te_att))
        
        if auc <= 0.55:
            verdict = "Low empirical membership distinguishability (near random discrimination)"
        elif auc <= 0.70:
            verdict = "Moderate distinguishability under gradient shadow model"
        else:
            verdict = "Elevated distinguishability detected under attack configuration"
            
        results[model_name] = {
            "attacker_model": model_name,
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "precision_at_50": round(prec_at_50, 4),
            "recall_at_50": round(rec_at_50, 4),
            "confusion_matrix": cm,
            "roc_curve": roc_pts,
            "verdict": verdict,
            "train_size": len(train_seed_df),
            "holdout_size": len(holdout_seed_df),
            "disclaimer": "Attack performance near random discrimination indicates low empirical membership distinguishability under this specific attack configuration. No formal differential privacy guarantee claimed."
        }
        
    return results
