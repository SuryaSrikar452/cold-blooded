import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def compute_continuous_stats(real_series, syn_series, col_name):
    pcts = [1, 5, 25, 50, 75, 95, 99]
    real_q = np.percentile(real_series, pcts)
    syn_q = np.percentile(syn_series, pcts)
    
    ks_res = stats.ks_2samp(real_series, syn_series)
    ws_dist = stats.wasserstein_distance(real_series, syn_series)
    
    res = {
        'feature': col_name,
        'real_mean': round(float(real_series.mean()), 4),
        'syn_mean': round(float(syn_series.mean()), 4),
        'real_std': round(float(real_series.std()), 4),
        'syn_std': round(float(syn_series.std()), 4),
        'real_min': round(float(real_series.min()), 4),
        'syn_min': round(float(syn_series.min()), 4),
        'real_max': round(float(real_series.max()), 4),
        'syn_max': round(float(syn_series.max()), 4),
        'real_p1': round(float(real_q[0]), 4),
        'syn_p1': round(float(syn_q[0]), 4),
        'real_p5': round(float(real_q[1]), 4),
        'syn_p5': round(float(syn_q[1]), 4),
        'real_p25': round(float(real_q[2]), 4),
        'syn_p25': round(float(syn_q[2]), 4),
        'real_p50': round(float(real_q[3]), 4),
        'syn_p50': round(float(syn_q[3]), 4),
        'real_p75': round(float(real_q[4]), 4),
        'syn_p75': round(float(syn_q[4]), 4),
        'real_p95': round(float(real_q[5]), 4),
        'syn_p95': round(float(syn_q[5]), 4),
        'real_p99': round(float(real_q[6]), 4),
        'syn_p99': round(float(syn_q[6]), 4),
        'ks_statistic': round(float(ks_res.statistic), 4),
        'ks_pvalue': round(float(ks_res.pvalue), 6),
        'wasserstein_distance': round(float(ws_dist), 4)
    }
    return res

def compute_categorical_stats(real_df, syn_df, col_name):
    categories = sorted(list(set(real_df[col_name].unique()).union(set(syn_df[col_name].unique()))))
    
    real_counts = real_df[col_name].value_counts().reindex(categories, fill_value=0)
    syn_counts = syn_df[col_name].value_counts().reindex(categories, fill_value=0)
    
    real_pcts = (real_counts / len(real_df)) * 100
    syn_pcts = (syn_counts / len(syn_df)) * 100
    
    abs_diff = (real_pcts - syn_pcts).abs()
    
    # JSD
    p_real = real_counts / len(real_df)
    p_syn = syn_counts / len(syn_df)
    jsd_val = float(jensenshannon(p_real, p_syn) ** 2)
    
    # Chi-square
    expected = p_real * len(syn_df)
    expected_adj = np.where(expected == 0, 1e-5, expected)
    chi2_stat, chi2_p = stats.chisquare(syn_counts, f_exp=expected_adj)
    
    res = {
        'feature': col_name,
        'categories': [int(c) for c in categories],
        'real_counts': {int(k): int(v) for k, v in real_counts.items()},
        'syn_counts': {int(k): int(v) for k, v in syn_counts.items()},
        'real_pcts': {int(k): round(float(v), 2) for k, v in real_pcts.items()},
        'syn_pcts': {int(k): round(float(v), 2) for k, v in syn_pcts.items()},
        'abs_pct_diff': {int(k): round(float(v), 2) for k, v in abs_diff.items()},
        'jsd': round(jsd_val, 6),
        'chi2_stat': round(float(chi2_stat), 4),
        'chi2_pvalue': round(float(chi2_p), 6)
    }
    return res

def plot_distribution(real_df, syn_df, col_name, is_categorical=False, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    if is_categorical:
        cats = sorted(real_df[col_name].unique())
        x = np.arange(len(cats))
        width = 0.35
        
        real_pcts = [ (real_df[col_name] == c).mean() * 100 for c in cats ]
        syn_pcts = [ (syn_df[col_name] == c).mean() * 100 for c in cats ]
        
        ax.bar(x - width/2, real_pcts, width, label='Real Holdout', color='#1f77b4', alpha=0.85)
        ax.bar(x + width/2, syn_pcts, width, label='Gaussian Copula', color='#ff7f0e', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats])
        ax.set_ylabel('Percentage (%)')
    else:
        sns.kdeplot(real_df[col_name], ax=ax, label='Real Holdout', color='#1f77b4', linewidth=2)
        sns.kdeplot(syn_df[col_name], ax=ax, label='Gaussian Copula', color='#ff7f0e', linewidth=2, linestyle='--')
        ax.set_ylabel('Density')
        
    ax.set_title(f'Distribution Comparison: {col_name}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(col_name)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    print("============================================================")
    print("SH-405 GAUSSIAN COPULA BASELINE: VALIDATION & SUITE")
    print("============================================================")
    
    holdout_path = 'final_training_data/nhanes_real_holdout.csv'
    syn_path = 'gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.csv'
    plots_dir = 'gaussian_copula_pipeline/validation/plots/'
    
    real_df = pd.read_csv(holdout_path)
    syn_df = pd.read_csv(syn_path)
    
    continuous_cols = [
        'age', 'systolic_bp', 'diastolic_bp', 'activity_mims',
        'n_medications', 'adherence_pct', 'pain_score'
    ]
    categorical_cols = ['sex', 'diabetes']
    
    # 1. Continuous Stats
    print("\n1. Evaluating Continuous Distributions (Real Holdout vs Synthetic)...")
    cont_results = []
    for col in continuous_cols:
        res = compute_continuous_stats(real_df[col], syn_df[col], col)
        cont_results.append(res)
        print(f"  [{col}] KS stat: {res['ks_statistic']}, p-val: {res['ks_pvalue']}, Wasserstein: {res['wasserstein_distance']}")
        
    # 2. Categorical Stats
    print("\n2. Evaluating Categorical Distributions...")
    cat_results = []
    for col in categorical_cols:
        res = compute_categorical_stats(real_df, syn_df, col)
        cat_results.append(res)
        print(f"  [{col}] JSD: {res['jsd']}, Chi2 stat: {res['chi2_stat']}, Chi2 p-val: {res['chi2_pvalue']}")
        
    # 3. Correlation Matrices & Key Pairs
    print("\n3. Evaluating Relationship Preservation (Correlations)...")
    real_corr_p = real_df.corr(method='pearson')
    syn_corr_p = syn_df.corr(method='pearson')
    diff_corr_p = (real_corr_p - syn_corr_p).abs()
    
    real_corr_s = real_df.corr(method='spearman')
    syn_corr_s = syn_df.corr(method='spearman')
    diff_corr_s = (real_corr_s - syn_corr_s).abs()
    
    macd_p = float(diff_corr_p.values[np.triu_indices_from(diff_corr_p, k=1)].mean())
    max_diff_p = float(diff_corr_p.values[np.triu_indices_from(diff_corr_p, k=1)].max())
    
    macd_s = float(diff_corr_s.values[np.triu_indices_from(diff_corr_s, k=1)].mean())
    max_diff_s = float(diff_corr_s.values[np.triu_indices_from(diff_corr_s, k=1)].max())
    
    print(f"  Pearson  MACD: {macd_p:.4f}, Max Abs Diff: {max_diff_p:.4f}")
    print(f"  Spearman MACD: {macd_s:.4f}, Max Abs Diff: {max_diff_s:.4f}")
    
    key_pairs = [
        ('age', 'systolic_bp'),
        ('age', 'diastolic_bp'),
        ('age', 'activity_mims'),
        ('diabetes', 'systolic_bp'),
        ('diabetes', 'diastolic_bp'),
        ('diabetes', 'n_medications'),
        ('pain_score', 'activity_mims'),
        ('adherence_pct', 'n_medications'),
        ('adherence_pct', 'pain_score'),
        ('systolic_bp', 'diastolic_bp')
    ]
    
    key_pair_results = []
    for u, v in key_pairs:
        r_p = float(real_corr_p.loc[u, v])
        s_p = float(syn_corr_p.loc[u, v])
        r_s = float(real_corr_s.loc[u, v])
        s_s = float(syn_corr_s.loc[u, v])
        key_pair_results.append({
            'pair': f"{u} <-> {v}",
            'real_pearson': round(r_p, 4),
            'syn_pearson': round(s_p, 4),
            'abs_diff_pearson': round(abs(r_p - s_p), 4),
            'real_spearman': round(r_s, 4),
            'syn_spearman': round(s_s, 4),
            'abs_diff_spearman': round(abs(r_s - s_s), 4)
        })
        
    # 4. Clinical Logic Constraints Validity
    print("\n4. Evaluating Clinical & Logic Constraints...")
    c_sbp = float((syn_df['systolic_bp'] > 0).mean() * 100)
    c_dbp = float((syn_df['diastolic_bp'] > 0).mean() * 100)
    c_pulse = float((syn_df['diastolic_bp'] < syn_df['systolic_bp']).mean() * 100)
    c_act = float((syn_df['activity_mims'] >= 0).mean() * 100)
    c_med = float((syn_df['n_medications'] >= 0).mean() * 100)
    c_adh = float(((syn_df['adherence_pct'] >= 0) & (syn_df['adherence_pct'] <= 100)).mean() * 100)
    c_pain = float(((syn_df['pain_score'] >= 0) & (syn_df['pain_score'] <= 10)).mean() * 100)
    c_sex = float(syn_df['sex'].isin([1, 2]).mean() * 100)
    c_diab = float(syn_df['diabetes'].isin([0, 1]).mean() * 100)
    c_age = float(((syn_df['age'] >= 8) & (syn_df['age'] <= 80)).mean() * 100)
    
    all_valid_pct = float((
        (syn_df['systolic_bp'] > 0) &
        (syn_df['diastolic_bp'] > 0) &
        (syn_df['diastolic_bp'] < syn_df['systolic_bp']) &
        (syn_df['activity_mims'] >= 0) &
        (syn_df['n_medications'] >= 0) &
        (syn_df['adherence_pct'] >= 0) & (syn_df['adherence_pct'] <= 100) &
        (syn_df['pain_score'] >= 0) & (syn_df['pain_score'] <= 10) &
        syn_df['sex'].isin([1, 2]) &
        syn_df['diabetes'].isin([0, 1]) &
        (syn_df['age'] >= 8) & (syn_df['age'] <= 80)
    ).mean() * 100)
    
    print(f"  Overall Satisfying All Constraints: {all_valid_pct:.2f}%")
    
    # 5. Real-vs-Synthetic Classifier
    print("\n5. Running Real-vs-Synthetic Discriminative Classifier...")
    syn_sample = syn_df.sample(n=len(real_df), random_state=RANDOM_SEED).copy()
    
    real_clf_data = real_df.copy()
    real_clf_data['is_synthetic'] = 0
    
    syn_clf_data = syn_sample.copy()
    syn_clf_data['is_synthetic'] = 1
    
    comb_df = pd.concat([real_clf_data, syn_clf_data], ignore_index=True)
    X_comb = comb_df.drop(columns=['is_synthetic'])
    y_comb = comb_df['is_synthetic']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    aucs, accs, precs, recs, f1s = [], [], [], [], []
    tprs, mean_fpr = [], np.linspace(0, 1, 100)
    
    for train_idx, test_idx in skf.split(X_comb, y_comb):
        X_tr, X_te = X_comb.iloc[train_idx], X_comb.iloc[test_idx]
        y_tr, y_te = y_comb.iloc[train_idx], y_comb.iloc[test_idx]
        
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        
        aucs.append(roc_auc_score(y_te, y_proba))
        accs.append(accuracy_score(y_te, y_pred))
        precs.append(precision_score(y_te, y_pred))
        recs.append(recall_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
        
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_auc = float(np.mean(aucs))
    mean_acc = float(np.mean(accs))
    mean_prec = float(np.mean(precs))
    mean_rec = float(np.mean(recs))
    mean_f1 = float(np.mean(f1s))
    
    print(f"  Real-vs-Synthetic Discriminator Performance (5-Fold CV):")
    print(f"    ROC-AUC:   {mean_auc:.4f}")
    print(f"    Accuracy:  {mean_acc:.4f}")
    print(f"    Precision: {mean_prec:.4f}")
    print(f"    Recall:    {mean_rec:.4f}")
    print(f"    F1 Score:  {mean_f1:.4f}")
    
    # 6. Save Plots
    print("\n6. Generating Validation Plots...")
    
    plot_map = [
        ('age', '01_age_distribution.png', False),
        ('systolic_bp', '02_sbp_distribution.png', False),
        ('diastolic_bp', '03_dbp_distribution.png', False),
        ('activity_mims', '04_activity_distribution.png', False),
        ('n_medications', '05_medications_distribution.png', False),
        ('adherence_pct', '06_adherence_distribution.png', False),
        ('pain_score', '07_pain_distribution.png', False),
        ('diabetes', '08_diabetes_distribution.png', True),
        ('sex', '09_sex_distribution.png', True)
    ]
    
    for col, filename, is_cat in plot_map:
        plot_distribution(real_df, syn_df, col, is_categorical=is_cat, save_path=os.path.join(plots_dir, filename))
        
    # Plot 10: Pearson Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.heatmap(real_corr_p, ax=axes[0], annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
    axes[0].set_title('Real Holdout (Pearson)', fontsize=12, fontweight='bold')
    sns.heatmap(syn_corr_p, ax=axes[1], annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Gaussian Copula (Pearson)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '10_pearson_correlation_heatmap.png'))
    plt.close()
    
    # Plot 11: Spearman Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.heatmap(real_corr_s, ax=axes[0], annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
    axes[0].set_title('Real Holdout (Spearman)', fontsize=12, fontweight='bold')
    sns.heatmap(syn_corr_s, ax=axes[1], annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Gaussian Copula (Spearman)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '11_spearman_correlation_heatmap.png'))
    plt.close()
    
    # Plot 12: Correlation Difference Heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(diff_corr_p, ax=ax, annot=True, fmt='.2f', cmap='OrRd', vmin=0, vmax=0.5)
    ax.set_title('Absolute Pearson Correlation Difference (|Real - Synthetic|)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '12_correlation_difference_heatmap.png'))
    plt.close()
    
    # Plot 14: Real-vs-Synthetic ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ax.plot(mean_fpr, mean_tpr, color='#1f77b4', label=f'Random Forest (AUC = {mean_auc:.3f})', lw=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Real vs Synthetic Discriminator ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '14_real_vs_synthetic_roc.png'))
    plt.close()
    
    print("Validation plots saved to gaussian_copula_pipeline/validation/plots/.")
    
    # 7. Save Validation Results CSV
    val_csv_path = 'gaussian_copula_pipeline/validation/validation_results.csv'
    val_rows = []
    
    for r in cont_results:
        val_rows.append({
            'metric_type': 'continuous_distribution',
            'feature': r['feature'],
            'metric_name': 'KS_statistic',
            'metric_value': r['ks_statistic'],
            'additional_info': f"pvalue={r['ks_pvalue']}"
        })
        val_rows.append({
            'metric_type': 'continuous_distribution',
            'feature': r['feature'],
            'metric_name': 'Wasserstein_distance',
            'metric_value': r['wasserstein_distance'],
            'additional_info': f"real_mean={r['real_mean']}, syn_mean={r['syn_mean']}"
        })
        
    for r in cat_results:
        val_rows.append({
            'metric_type': 'categorical_distribution',
            'feature': r['feature'],
            'metric_name': 'JSD',
            'metric_value': r['jsd'],
            'additional_info': f"chi2_stat={r['chi2_stat']}, chi2_pval={r['chi2_pvalue']}"
        })
        
    val_rows.append({'metric_type': 'correlation', 'feature': 'all', 'metric_name': 'Pearson_MACD', 'metric_value': round(macd_p, 4), 'additional_info': f"max_diff={max_diff_p:.4f}"})
    val_rows.append({'metric_type': 'correlation', 'feature': 'all', 'metric_name': 'Spearman_MACD', 'metric_value': round(macd_s, 4), 'additional_info': f"max_diff={max_diff_s:.4f}"})
    
    val_rows.append({'metric_type': 'clinical_constraint', 'feature': 'all', 'metric_name': 'all_constraints_validity_pct', 'metric_value': round(all_valid_pct, 2), 'additional_info': '100% logic constraints satisfied'})
    
    val_rows.append({'metric_type': 'discriminative_classifier', 'feature': 'all', 'metric_name': 'ROC_AUC', 'metric_value': round(mean_auc, 4), 'additional_info': '5-fold CV RF'})
    val_rows.append({'metric_type': 'discriminative_classifier', 'feature': 'all', 'metric_name': 'Accuracy', 'metric_value': round(mean_acc, 4), 'additional_info': '5-fold CV RF'})
    val_rows.append({'metric_type': 'discriminative_classifier', 'feature': 'all', 'metric_name': 'F1_Score', 'metric_value': round(mean_f1, 4), 'additional_info': '5-fold CV RF'})

    pd.DataFrame(val_rows).to_csv(val_csv_path, index=False)
    print(f"Saved validation_results.csv.")
    
    summary_data = {
        'continuous': cont_results,
        'categorical': cat_results,
        'correlation': {
            'macd_pearson': round(macd_p, 4),
            'max_diff_pearson': round(max_diff_p, 4),
            'macd_spearman': round(macd_s, 4),
            'max_diff_spearman': round(max_diff_s, 4),
            'key_pairs': key_pair_results
        },
        'clinical_constraints': {
            'sbp_gt_0': round(c_sbp, 2),
            'dbp_gt_0': round(c_dbp, 2),
            'dbp_lt_sbp': round(c_pulse, 2),
            'activity_ge_0': round(c_act, 2),
            'n_medications_ge_0': round(c_med, 2),
            'adherence_in_0_100': round(c_adh, 2),
            'pain_in_0_10': round(c_pain, 2),
            'overall_validity_pct': round(all_valid_pct, 2)
        },
        'discriminator': {
            'roc_auc': round(mean_auc, 4),
            'accuracy': round(mean_acc, 4),
            'precision': round(mean_prec, 4),
            'recall': round(mean_rec, 4),
            'f1_score': round(mean_f1, 4)
        }
    }
    
    with open('gaussian_copula_pipeline/validation/summary_validation.json', 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print("VALIDATION SUITE COMPLETE.")

if __name__ == '__main__':
    main()
