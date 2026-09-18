"""
gaussian_copula_final/evaluate_final.py
---------------------------------------
Comprehensive multi-model scientific benchmark for GaussianCopulaFinal.

Strict Holdout Separation:
The real holdout (N=1,207) is evaluated strictly as an untouched external test set
after the final model has been frozen.

Evaluates:
- Real Training Reference (N=4,826)
- Real Holdout Reference (N=1,207)
- Gaussian Copula V1 (audited frozen baseline)
- Gaussian Copula V2 (experimental version)
- CTGAN V1 (audited neural baseline)
- Gaussian Copula Final (hardened model)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats, spatial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaussian_copula_final.copula_final import APPROVED_FEATURES

CONTINUOUS_FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "activity_mims",
    "n_medications",
    "adherence_pct",
    "pain_score",
]

CATEGORICAL_FEATURES = ["sex", "diabetes"]


def compute_tvd(p: pd.Series, q: pd.Series) -> float:
    """Total Variation Distance between two discrete distributions."""
    cats = sorted(list(set(p.index).union(set(q.index))))
    p_norm = p.reindex(cats, fill_value=0) / p.sum()
    q_norm = q.reindex(cats, fill_value=0) / q.sum()
    return float(0.5 * np.sum(np.abs(p_norm - q_norm)))


def evaluate_clinical_validity(df: pd.DataFrame) -> Tuple[float, int, int]:
    """Verify 100% compliance with physiological domain rules."""
    valid_mask = (
        (df["systolic_bp"] > 0)
        & (df["diastolic_bp"] > 0)
        & (df["systolic_bp"] > df["diastolic_bp"])
        & (df["systolic_bp"] - df["diastolic_bp"] >= 5)
        & (df["activity_mims"] >= 0)
        & (df["n_medications"] >= 0)
        & (df["adherence_pct"] >= 0.0)
        & (df["adherence_pct"] <= 100.0)
        & (df["pain_score"] >= 0.0)
        & (df["pain_score"] <= 10.0)
        & (df["age"] >= 8)
        & (df["age"] <= 80)
    )
    total = len(df)
    valid_count = int(valid_mask.sum())
    pct = (valid_count / total) * 100.0 if total > 0 else 0.0
    return pct, valid_count, total


def evaluate_discriminator(real_df: pd.DataFrame, synth_df: pd.DataFrame, seed: int = 42) -> float:
    """
    Logistic regression discriminator predicting Real vs Synthetic with 5-fold CV.
    Reports cross-validated ROC-AUC under neutral framing.
    """
    n = min(len(real_df), len(synth_df))
    X_real = real_df[APPROVED_FEATURES].sample(n=n, random_state=seed).values
    X_synth = synth_df[APPROVED_FEATURES].sample(n=n, random_state=seed).values

    X = np.vstack([X_real, X_synth])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in cv.split(X_scaled, y):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_scaled[train_idx], y[train_idx])
        probs = clf.predict_proba(X_scaled[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))

    return float(np.mean(aucs))


def evaluate_tstr(synth_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42) -> Dict[str, float]:
    """
    Train on Synthetic, Test on Real (TSTR) for downstream diabetes prediction.
    Reports ROC-AUC, PR-AUC, F1, Precision, and Recall.
    """
    feature_cols = [c for c in APPROVED_FEATURES if c != "diabetes"]
    X_train = synth_df[feature_cols].values
    y_train = synth_df["diabetes"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["diabetes"].values.astype(int)

    # If synthetic cohort has only 1 class, utility is undefined
    if len(np.unique(y_train)) < 2:
        return {"roc_auc": 0.5, "pr_auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X_train_s, y_train)

    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
    }


def evaluate_privacy(train_df: pd.DataFrame, holdout_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Empirical privacy and memorization risk diagnostics:
    - Exact full-vector matches
    - Distance to Closest Record (DCR) 5th percentile
    - Size-matched reference set comparison
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[APPROVED_FEATURES].values)
    X_holdout = scaler.transform(holdout_df[APPROVED_FEATURES].values)
    X_synth = scaler.transform(synth_df[APPROVED_FEATURES].values)

    # Exact matches
    train_exact = 0
    train_set = set(map(tuple, train_df[APPROVED_FEATURES].values))
    for row in synth_df[APPROVED_FEATURES].values:
        if tuple(row) in train_set:
            train_exact += 1

    holdout_exact = 0
    holdout_set = set(map(tuple, holdout_df[APPROVED_FEATURES].values))
    for row in synth_df[APPROVED_FEATURES].values:
        if tuple(row) in holdout_set:
            holdout_exact += 1

    # DCR to full training reference
    nn_train = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_train)
    dists_train, _ = nn_train.kneighbors(X_synth)
    dcr_train_p5 = float(np.percentile(dists_train, 5))
    dcr_train_mean = float(np.mean(dists_train))

    # DCR to holdout reference (N=1,207)
    nn_holdout = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_holdout)
    dists_holdout, _ = nn_holdout.kneighbors(X_synth)
    dcr_holdout_p5 = float(np.percentile(dists_holdout, 5))
    dcr_holdout_mean = float(np.mean(dists_holdout))

    # Size-matched training reference (subsample N=1,207 from training)
    rng = np.random.RandomState(42)
    idx_sub = rng.choice(len(train_df), size=len(holdout_df), replace=False)
    X_train_sub = X_train[idx_sub]
    nn_sub = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_train_sub)
    dists_sub, _ = nn_sub.kneighbors(X_synth)
    dcr_sub_p5 = float(np.percentile(dists_sub, 5))
    dcr_sub_mean = float(np.mean(dists_sub))

    return {
        "exact_matches_train": train_exact,
        "exact_matches_holdout": holdout_exact,
        "dcr_train_p5": dcr_train_p5,
        "dcr_train_mean": dcr_train_mean,
        "dcr_holdout_p5": dcr_holdout_p5,
        "dcr_holdout_mean": dcr_holdout_mean,
        "dcr_size_matched_p5": dcr_sub_p5,
        "dcr_size_matched_mean": dcr_sub_mean,
    }


def main():
    print("=" * 80)
    print("SH-405: SCIENTIFIC MULTI-MODEL BENCHMARK (GC FINAL vs V1, V2, CTGAN)")
    print("=" * 80)

    # 1. Load Real Datasets
    train_path = os.path.join(PROJECT_ROOT, "final_training_data", "nhanes_generative_train.csv")
    holdout_path = os.path.join(PROJECT_ROOT, "final_training_data", "nhanes_real_holdout.csv")
    df_train = pd.read_csv(train_path)
    df_holdout = pd.read_csv(holdout_path)
    print(f"Loaded Real Train (N={len(df_train)}) and Real Holdout (N={len(df_holdout)})")

    # 2. Load Synthetic Datasets
    datasets = {
        "Real_Train": df_train,
        "Real_Holdout": df_holdout,
        "Gaussian_Copula_V1": pd.read_csv(os.path.join(PROJECT_ROOT, "training_pipeline", "synthetic", "gaussian_copula_synthetic.csv")),
        "Gaussian_Copula_V2": pd.read_csv(os.path.join(PROJECT_ROOT, "gaussian_copula_v2", "synthetic", "gaussian_copula_v2_synthetic.csv")),
        "CTGAN": pd.read_csv(os.path.join(PROJECT_ROOT, "training_pipeline", "synthetic", "ctgan_synthetic.csv")),
        "Gaussian_Copula_Final": pd.read_csv(os.path.join(SCRIPT_DIR, "synthetic", "gaussian_copula_final_synthetic.csv")),
    }

    comparison_records = []
    detailed_ks = {}

    holdout_corr = df_holdout[APPROVED_FEATURES].corr().values
    holdout_spearman = df_holdout[APPROVED_FEATURES].corr(method="spearman").values

    print("\nEvaluating cohorts against Real Holdout benchmark...")

    for model_name, df_mod in datasets.items():
        print(f" -> Benchmarking {model_name} (N={len(df_mod)})...")

        # A. Marginal KS and Wasserstein vs Real Holdout
        ks_stats = {}
        wass_stats = {}
        for col in CONTINUOUS_FEATURES:
            stat_ks = stats.ks_2samp(df_holdout[col].values, df_mod[col].values).statistic
            stat_wass = stats.wasserstein_distance(df_holdout[col].values, df_mod[col].values)
            ks_stats[col] = float(stat_ks)
            wass_stats[col] = float(stat_wass)

        mean_ks = float(np.mean(list(ks_stats.values())))
        mean_wass = float(np.mean(list(wass_stats.values())))
        detailed_ks[model_name] = ks_stats

        # Zero masses
        pain_zero_frac = float((df_mod["pain_score"] == 0).mean())
        med_zero_frac = float((df_mod["n_medications"] == 0).mean())

        # Categorical
        diab_prev = float((df_mod["diabetes"] == 1).mean())
        sex_tvd = compute_tvd(df_holdout["sex"].value_counts(), df_mod["sex"].value_counts())
        diab_tvd = compute_tvd(df_holdout["diabetes"].value_counts(), df_mod["diabetes"].value_counts())

        # B. Correlation Drifts
        mod_corr = df_mod[APPROVED_FEATURES].corr().values
        mod_spearman = df_mod[APPROVED_FEATURES].corr(method="spearman").values
        pearson_drift = float(np.mean(np.abs(mod_corr - holdout_corr)))
        spearman_drift = float(np.mean(np.abs(mod_spearman - holdout_spearman)))

        # Specific conditional correlations
        corr_diab_age = float(df_mod["diabetes"].corr(df_mod["age"]))
        corr_diab_sbp = float(df_mod["diabetes"].corr(df_mod["systolic_bp"]))
        corr_diab_meds = float(df_mod["diabetes"].corr(df_mod["n_medications"]))
        corr_pain_age = float(df_mod["pain_score"].corr(df_mod["age"]))
        corr_pain_diab = float(df_mod["pain_score"].corr(df_mod["diabetes"]))

        # C. Clinical Validity
        val_pct, val_count, total = evaluate_clinical_validity(df_mod)

        # D. Discriminator
        if model_name in ["Real_Train", "Real_Holdout"]:
            disc_auc = 0.5000 if model_name == "Real_Holdout" else evaluate_discriminator(df_holdout, df_train)
        else:
            disc_auc = evaluate_discriminator(df_holdout, df_mod)

        # E. Downstream Utility (TSTR)
        if model_name == "Real_Holdout":
            # Real test set evaluated against itself is cross-validated reference
            tstr = evaluate_tstr(df_train, df_holdout)
        else:
            tstr = evaluate_tstr(df_mod, df_holdout)

        # F. Privacy
        priv = evaluate_privacy(df_train, df_holdout, df_mod)

        record = {
            "Model": model_name,
            "N_Rows": len(df_mod),
            "Mean_Continuous_KS": round(mean_ks, 4),
            "Pain_Score_KS": round(ks_stats["pain_score"], 4),
            "Meds_Count_KS": round(ks_stats["n_medications"], 4),
            "Age_KS": round(ks_stats["age"], 4),
            "SBP_KS": round(ks_stats["systolic_bp"], 4),
            "Mean_Wasserstein": round(mean_wass, 4),
            "Pain_Zero_Pct": round(pain_zero_frac * 100.0, 2),
            "Meds_Zero_Pct": round(med_zero_frac * 100.0, 2),
            "Diabetes_Prevalence_Pct": round(diab_prev * 100.0, 2),
            "Diabetes_TVD": round(diab_tvd, 4),
            "Mean_Pearson_Drift": round(pearson_drift, 4),
            "Mean_Spearman_Drift": round(spearman_drift, 4),
            "Corr_Diabetes_Age": round(corr_diab_age, 3),
            "Corr_Diabetes_SBP": round(corr_diab_sbp, 3),
            "Corr_Diabetes_Meds": round(corr_diab_meds, 3),
            "Corr_Pain_Age": round(corr_pain_age, 3),
            "Corr_Pain_Diabetes": round(corr_pain_diab, 3),
            "Clinical_Validity_Pct": round(val_pct, 2),
            "Discriminator_ROC_AUC": round(disc_auc, 4),
            "TSTR_ROC_AUC": round(tstr["roc_auc"], 4),
            "TSTR_PR_AUC": round(tstr["pr_auc"], 4),
            "TSTR_F1": round(tstr["f1"], 4),
            "Exact_Matches_Train": priv["exact_matches_train"],
            "Exact_Matches_Holdout": priv["exact_matches_holdout"],
            "DCR_Holdout_P5": round(priv["dcr_holdout_p5"], 4),
            "DCR_Size_Matched_P5": round(priv["dcr_size_matched_p5"], 4),
        }
        comparison_records.append(record)

    # 3. Save outputs
    outputs_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    df_comp = pd.DataFrame(comparison_records)
    csv_path = os.path.join(outputs_dir, "FINAL_MODEL_COMPARISON.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved comparison summary table to: {csv_path}")

    # Also save detailed per-feature KS table
    df_ks = pd.DataFrame(detailed_ks).round(4)
    ks_path = os.path.join(outputs_dir, "detailed_per_feature_ks.csv")
    df_ks.to_csv(ks_path)
    print(f"Saved detailed per-feature KS table to: {ks_path}")

    print("\n--- MULTI-MODEL COMPARISON PREVIEW ---")
    print(df_comp[["Model", "Mean_Continuous_KS", "Mean_Pearson_Drift", "Clinical_Validity_Pct", "Discriminator_ROC_AUC", "TSTR_PR_AUC", "Exact_Matches_Train"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
