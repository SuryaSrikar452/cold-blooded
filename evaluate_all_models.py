import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from backend.services.hurdle_copula_model import HurdleConditionalCopulaModel
from privacy.mia_engine import run_membership_inference_attack

RANDOM_SEED = 42

def evaluate_model_synthetic_data(syn_df, real_train_df, real_holdout_df):
    continuous_cols = ['age', 'systolic_bp', 'diastolic_bp', 'activity_mims', 'n_medications', 'adherence_pct', 'pain_score']
    categorical_cols = ['sex', 'diabetes']
    
    # 1. Pain score zero proportion & JSD
    real_pain_zero = float((real_holdout_df['pain_score'] == 0.0).mean() * 100)
    syn_pain_zero = float((syn_df['pain_score'] == 0.0).mean() * 100)
    
    # Pain score positive KS & Wasserstein
    real_pain_pos = real_holdout_df[real_holdout_df['pain_score'] > 0]['pain_score']
    syn_pain_pos = syn_df[syn_df['pain_score'] > 0]['pain_score']
    
    if len(syn_pain_pos) > 0 and len(real_pain_pos) > 0:
        pain_pos_ks = float(stats.ks_2samp(real_pain_pos, syn_pain_pos).statistic)
        pain_pos_ws = float(stats.wasserstein_distance(real_pain_pos, syn_pain_pos))
    else:
        pain_pos_ks, pain_pos_ws = 1.0, 10.0
        
    pain_ks_full = float(stats.ks_2samp(real_holdout_df['pain_score'], syn_df['pain_score']).statistic)
    
    # Pain JSD
    bins = np.linspace(0, 10, 11)
    p_real_pain, _ = np.histogram(real_holdout_df['pain_score'], bins=bins, density=True)
    p_syn_pain, _ = np.histogram(syn_df['pain_score'], bins=bins, density=True)
    pain_jsd = float(jensenshannon(p_real_pain + 1e-6, p_syn_pain + 1e-6) ** 2)

    # 2. Diabetes prevalence & JSD
    real_diab_prev = float((real_holdout_df['diabetes'] == 1).mean() * 100)
    syn_diab_prev = float((syn_df['diabetes'] == 1).mean() * 100)
    
    p_real_diab = real_holdout_df['diabetes'].value_counts(normalize=True).reindex([0, 1], fill_value=0).values
    p_syn_diab = syn_df['diabetes'].value_counts(normalize=True).reindex([0, 1], fill_value=0).values
    diab_jsd = float(jensenshannon(p_real_diab, p_syn_diab) ** 2)

    # 3. Correlations & MACD
    real_corr_p = real_holdout_df[continuous_cols + categorical_cols].corr(method='pearson')
    syn_corr_p = syn_df[continuous_cols + categorical_cols].corr(method='pearson')
    diff_corr_p = (real_corr_p - syn_corr_p).abs()
    pearson_macd = float(diff_corr_p.values[np.triu_indices_from(diff_corr_p, k=1)].mean())

    real_corr_s = real_holdout_df[continuous_cols + categorical_cols].corr(method='spearman')
    syn_corr_s = syn_df[continuous_cols + categorical_cols].corr(method='spearman')
    diff_corr_s = (real_corr_s - syn_corr_s).abs()
    spearman_macd = float(diff_corr_s.values[np.triu_indices_from(diff_corr_s, k=1)].mean())

    # 4. Downstream utility (diabetes prediction on Real Holdout)
    X_tr_syn = syn_df[continuous_cols + ['sex']]
    y_tr_syn = syn_df['diabetes']
    X_ho_real = real_holdout_df[continuous_cols + ['sex']]
    y_ho_real = real_holdout_df['diabetes']

    clf_downstream = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_downstream.fit(X_tr_syn, y_tr_syn)
    y_pred_ds = clf_downstream.predict(X_ho_real)
    y_proba_ds = clf_downstream.predict_proba(X_ho_real)[:, 1]

    ds_auc = float(roc_auc_score(y_ho_real, y_proba_ds))
    ds_f1 = float(f1_score(y_ho_real, y_pred_ds, zero_division=0))
    ds_acc = float(accuracy_score(y_ho_real, y_pred_ds))

    # 5. Real-vs-synthetic discriminator AUC
    syn_sub = syn_df[continuous_cols + categorical_cols].sample(n=len(real_holdout_df), random_state=RANDOM_SEED).copy()
    syn_sub['is_syn'] = 1
    real_sub = real_holdout_df[continuous_cols + categorical_cols].copy()
    real_sub['is_syn'] = 0

    comb_df = pd.concat([real_sub, syn_sub], ignore_index=True)
    X_disc = comb_df.drop(columns=['is_syn'])
    y_disc = comb_df['is_syn']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    disc_aucs = []
    for tr_i, te_i in skf.split(X_disc, y_disc):
        clf_disc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        clf_disc.fit(X_disc.iloc[tr_i], y_disc.iloc[tr_i])
        y_p = clf_disc.predict_proba(X_disc.iloc[te_i])[:, 1]
        disc_aucs.append(roc_auc_score(y_disc.iloc[te_i], y_p))
    real_vs_syn_auc = float(np.mean(disc_aucs))

    # 6. MIA Privacy AUC
    mia_res = run_membership_inference_attack(real_train_df, real_holdout_df, syn_df, continuous_cols, categorical_cols, attacker_type='logistic_regression')
    mia_auc = float(mia_res['logistic_regression']['auc'])

    # 7. Exact duplicates & Clinical Validity
    exact_dups = int(len(pd.merge(syn_df[continuous_cols + categorical_cols], real_holdout_df[continuous_cols + categorical_cols], how='inner')))

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
        (syn_df['pain_score'] >= 0) & (syn_df['pain_score'] <= 10)
    )
    clinical_validity_pct = float(valid_mask.mean() * 100)

    return {
        'pain_zero_pct': round(syn_pain_zero, 2),
        'pain_jsd': round(pain_jsd, 6),
        'pain_full_ks': round(pain_ks_full, 4),
        'pain_positive_ks': round(pain_pos_ks, 4),
        'pain_positive_ws': round(pain_pos_ws, 4),
        'diabetes_prevalence_pct': round(syn_diab_prev, 2),
        'diabetes_jsd': round(diab_jsd, 6),
        'pearson_macd': round(pearson_macd, 4),
        'spearman_macd': round(spearman_macd, 4),
        'downstream_auc': round(ds_auc, 4),
        'downstream_f1': round(ds_f1, 4),
        'downstream_acc': round(ds_acc, 4),
        'real_vs_syn_auc': round(real_vs_syn_auc, 4),
        'mia_auc': round(mia_auc, 4),
        'exact_duplicates': exact_dups,
        'clinical_validity_pct': round(clinical_validity_pct, 2)
    }

def main():
    print("============================================================")
    print("EMPIRICAL EVALUATION: MULTI-MODEL COMPARISON")
    print("============================================================")

    train_df = pd.read_csv('final_training_data/nhanes_generative_train.csv')
    holdout_df = pd.read_csv('final_training_data/nhanes_real_holdout.csv')

    # Model 1: Baseline GC (Gaussian)
    print("\nEvaluating Model 1: Baseline Gaussian Copula...")
    syn_m1 = pd.read_csv('gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.csv')
    res_m1 = evaluate_model_synthetic_data(syn_m1, train_df, holdout_df)

    # Model 2: Truncated Gaussian Pre-Fix
    print("\nEvaluating Model 2: Truncated Gaussian Pre-Fix...")
    meta2 = Metadata.detect_from_dataframe(data=train_df, table_name='nhanes')
    meta2.update_column('sex', sdtype='categorical')
    meta2.update_column('diabetes', sdtype='categorical')
    synth2 = GaussianCopulaSynthesizer(meta2, enforce_min_max_values=True)
    synth2.fit(train_df)
    syn_m2 = synth2.sample(num_rows=len(train_df))
    res_m2 = evaluate_model_synthetic_data(syn_m2, train_df, holdout_df)

    # Model 3: Fixed Model (Hurdel + Conditional Diabetes)
    print("\nEvaluating Model 3: Fixed Model (Pain Hurdle + Diabetes Conditional)...")
    model3 = HurdleConditionalCopulaModel(random_seed=RANDOM_SEED)
    model3.fit(train_df)
    syn_m3 = model3.sample(num_rows=len(train_df))
    res_m3 = evaluate_model_synthetic_data(syn_m3, train_df, holdout_df)

    # Print Comparative Matrix Table
    metrics_list = [
        ('Pain zero proportion (%)', 'pain_zero_pct', f"{float((holdout_df['pain_score']==0).mean()*100):.2f}%"),
        ('Pain JSD', 'pain_jsd', '0.000000'),
        ('Pain Full KS', 'pain_full_ks', '0.0000'),
        ('Pain Positive KS', 'pain_positive_ks', '0.0000'),
        ('Diabetes prevalence (%)', 'diabetes_prevalence_pct', f"{float((holdout_df['diabetes']==1).mean()*100):.2f}%"),
        ('Diabetes JSD', 'diabetes_jsd', '0.000000'),
        ('Pearson MACD', 'pearson_macd', '0.0000'),
        ('Spearman MACD', 'spearman_macd', '0.0000'),
        ('Downstream ROC-AUC', 'downstream_auc', '0.9278'),
        ('Downstream F1 Score', 'downstream_f1', '0.5596'),
        ('Downstream Accuracy', 'downstream_acc', '0.9296'),
        ('Real-vs-Synthetic AUC', 'real_vs_syn_auc', '0.5000'),
        ('MIA Attacker AUC', 'mia_auc', '0.5000'),
        ('Exact Duplicates', 'exact_duplicates', '0'),
        ('Clinical Validity (%)', 'clinical_validity_pct', '100.00%')
    ]

    print("\n" + "="*80)
    print(f"{'Metric':<30} | {'Real Holdout':<12} | {'Baseline GC':<12} | {'Trunc Gaussian':<14} | {'Fixed Model':<12}")
    print("="*80)

    comparison_results = {}

    for label, key, real_val in metrics_list:
        v1 = res_m1[key]
        v2 = res_m2[key]
        v3 = res_m3[key]
        print(f"{label:<30} | {real_val:<12} | {v1:<12} | {v2:<14} | {v3:<12}")
        comparison_results[key] = {'real': real_val, 'm1_baseline': v1, 'm2_truncated': v2, 'm3_fixed': v3}

    # Save summary comparison JSON
    with open('gaussian_copula_pipeline/validation/multi_model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
        
    print("="*80)
    print("MULTI-MODEL EVALUATION COMPLETE.")

if __name__ == '__main__':
    main()
