import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt

RANDOM_SEED = 42

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    return {
        'roc_auc': round(auc, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'fpr': fpr,
        'tpr': tpr
    }

def main():
    print("============================================================")
    print("SH-405 GAUSSIAN COPULA BASELINE: DOWNSTREAM UTILITY")
    print("============================================================")
    
    train_path = 'final_training_data/nhanes_generative_train.csv'
    holdout_path = 'final_training_data/nhanes_real_holdout.csv'
    syn_path = 'gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.csv'
    plots_dir = 'gaussian_copula_pipeline/validation/plots/'
    
    real_train_df = pd.read_csv(train_path)
    real_holdout_df = pd.read_csv(holdout_path)
    syn_df = pd.read_csv(syn_path)
    
    target_col = 'diabetes'
    feature_cols = [c for c in real_train_df.columns if c != target_col]
    
    X_train_real = real_train_df[feature_cols]
    y_train_real = real_train_df[target_col]
    
    X_train_syn = syn_df[feature_cols]
    y_train_syn = syn_df[target_col]
    
    X_holdout = real_holdout_df[feature_cols]
    y_holdout = real_holdout_df[target_col]
    
    # 1. Train Model A (Real Training Data)
    print("Training Model A on REAL TRAINING DATA...")
    clf_real = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_real.fit(X_train_real, y_train_real)
    metrics_real = evaluate_model(clf_real, X_holdout, y_holdout)
    
    # 2. Train Model B (Gaussian Copula Synthetic Data)
    print("Training Model B on GAUSSIAN COPULA SYNTHETIC DATA...")
    clf_syn = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_syn.fit(X_train_syn, y_train_syn)
    metrics_syn = evaluate_model(clf_syn, X_holdout, y_holdout)
    
    print("\nDownstream Utility Results (Evaluated on REAL HOLDOUT):")
    print(f"  Target Variable: '{target_col}'")
    print(f"  Model A (Trained on Real):      ROC-AUC = {metrics_real['roc_auc']}, Acc = {metrics_real['accuracy']}, Prec = {metrics_real['precision']}, Rec = {metrics_real['recall']}, F1 = {metrics_real['f1']}")
    print(f"  Model B (Trained on Synthetic): ROC-AUC = {metrics_syn['roc_auc']}, Acc = {metrics_syn['accuracy']}, Prec = {metrics_syn['precision']}, Rec = {metrics_syn['recall']}, F1 = {metrics_syn['f1']}")
    
    # 3. Save downstream_results.csv
    results_df = pd.DataFrame([
        {
            'training_data': 'REAL_TRAIN',
            'evaluation_data': 'REAL_HOLDOUT',
            'target': target_col,
            'roc_auc': metrics_real['roc_auc'],
            'accuracy': metrics_real['accuracy'],
            'precision': metrics_real['precision'],
            'recall': metrics_real['recall'],
            'f1_score': metrics_real['f1']
        },
        {
            'training_data': 'GAUSSIAN_COPULA_SYNTHETIC',
            'evaluation_data': 'REAL_HOLDOUT',
            'target': target_col,
            'roc_auc': metrics_syn['roc_auc'],
            'accuracy': metrics_syn['accuracy'],
            'precision': metrics_syn['precision'],
            'recall': metrics_syn['recall'],
            'f1_score': metrics_syn['f1']
        }
    ])
    
    csv_path = 'gaussian_copula_pipeline/validation/downstream_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved downstream results to {csv_path}.")
    
    # 4. Generate Plot 15: Downstream Utility ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(metrics_real['fpr'], metrics_real['tpr'], color='#1f77b4', label=f"Model A: Real Train (AUC = {metrics_real['roc_auc']:.3f})", lw=2)
    ax.plot(metrics_syn['fpr'], metrics_syn['tpr'], color='#ff7f0e', label=f"Model B: Synthetic (AUC = {metrics_syn['roc_auc']:.3f})", lw=2, linestyle='--')
    ax.plot([0, 1], [0, 1], linestyle=':', color='gray', label='Random Chance (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title("Downstream Utility ROC Curve (Target: 'diabetes')", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, '15_downstream_utility_roc.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved downstream utility ROC plot to {plot_path}.")

if __name__ == '__main__':
    main()
