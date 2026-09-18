"""
gaussian_copula_final/generate_plots_final.py
---------------------------------------------
Generates publication-quality comparison figures for Gaussian Copula Final:
1. Marginal distribution overlays (Real Holdout vs V1 vs V2 vs Final)
2. Zero-inflation hurdle fidelity (pain score and medication count)
3. Correlation matrices and drift comparison
4. Discriminator and TSTR utility curves (ROC and PR-AUC)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaussian_copula_final.copula_final import APPROVED_FEATURES


def main():
    plots_dir = os.path.join(SCRIPT_DIR, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # Load datasets
    df_train = pd.read_csv(os.path.join(PROJECT_ROOT, "final_training_data", "nhanes_generative_train.csv"))
    df_holdout = pd.read_csv(os.path.join(PROJECT_ROOT, "final_training_data", "nhanes_real_holdout.csv"))
    df_v1 = pd.read_csv(os.path.join(PROJECT_ROOT, "training_pipeline", "synthetic", "gaussian_copula_synthetic.csv"))
    df_v2 = pd.read_csv(os.path.join(PROJECT_ROOT, "gaussian_copula_v2", "synthetic", "gaussian_copula_v2_synthetic.csv"))
    df_final = pd.read_csv(os.path.join(SCRIPT_DIR, "synthetic", "gaussian_copula_final_synthetic.csv"))

    colors = {
        "Real_Holdout": "#1f77b4",
        "GC_V1": "#d62728",
        "GC_V2": "#ff7f0e",
        "GC_Final": "#2ca02c",
    }

    # -------------------------------------------------------------
    # Plot 1: Marginal Distributions Overlay (6 key features)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plot_feats = ["age", "systolic_bp", "diastolic_bp", "activity_mims", "n_medications", "adherence_pct"]

    for ax, feat in zip(axes.flatten(), plot_feats):
        sns.kdeplot(df_holdout[feat], ax=ax, label="Real Holdout", color=colors["Real_Holdout"], lw=2.5)
        sns.kdeplot(df_v1[feat], ax=ax, label="GC V1 (Baseline)", color=colors["GC_V1"], lw=1.5, ls="--")
        sns.kdeplot(df_v2[feat], ax=ax, label="GC V2", color=colors["GC_V2"], lw=1.5, ls=":")
        sns.kdeplot(df_final[feat], ax=ax, label="GC Final (Hardened)", color=colors["GC_Final"], lw=2.0)
        ax.set_title(f"Marginal: {feat}", fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    p1_path = os.path.join(plots_dir, "01_marginal_distributions_comparison.png")
    plt.savefig(p1_path, dpi=200)
    plt.close()
    print(f"Saved: {p1_path}")

    # -------------------------------------------------------------
    # Plot 2: Zero-Inflation Hurdle Analysis (Pain Score & Medications)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Zero mass comparison barplot
    models = ["Real Train", "Real Holdout", "GC V1", "GC V2", "GC Final"]
    pain_zeros = [
        (df_train["pain_score"] == 0).mean() * 100,
        (df_holdout["pain_score"] == 0).mean() * 100,
        (df_v1["pain_score"] == 0).mean() * 100,
        (df_v2["pain_score"] == 0).mean() * 100,
        (df_final["pain_score"] == 0).mean() * 100,
    ]
    med_zeros = [
        (df_train["n_medications"] == 0).mean() * 100,
        (df_holdout["n_medications"] == 0).mean() * 100,
        (df_v1["n_medications"] == 0).mean() * 100,
        (df_v2["n_medications"] == 0).mean() * 100,
        (df_final["n_medications"] == 0).mean() * 100,
    ]

    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, pain_zeros, width, label="Pain Score (% zeros)", color="#4c72b0")
    ax1.bar(x + width/2, med_zeros, width, label="Meds Count (% zeros)", color="#55a868")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    ax1.set_ylabel("Zero Mass (%)")
    ax1.set_title("Zero Inflation Preservation (% Zeros)", fontweight="bold")
    ax1.axhline(69.5, color="#4c72b0", ls="--", alpha=0.7, label="Real Pain Target (69.5%)")
    ax1.axhline(52.2, color="#55a868", ls="--", alpha=0.7, label="Real Meds Target (52.2%)")
    ax1.legend(fontsize=8)

    # Positive pain magnitude KDE
    sns.kdeplot(df_holdout[df_holdout["pain_score"] > 0]["pain_score"], ax=ax2, label="Real Holdout", color=colors["Real_Holdout"], lw=2.5)
    sns.kdeplot(df_v1[df_v1["pain_score"] > 0]["pain_score"], ax=ax2, label="GC V1", color=colors["GC_V1"], lw=1.5, ls="--")
    sns.kdeplot(df_v2[df_v2["pain_score"] > 0]["pain_score"], ax=ax2, label="GC V2", color=colors["GC_V2"], lw=1.5, ls=":")
    sns.kdeplot(df_final[df_final["pain_score"] > 0]["pain_score"], ax=ax2, label="GC Final", color=colors["GC_Final"], lw=2.0)
    ax2.set_title("Positive Pain Score Magnitude Distribution", fontweight="bold")
    ax2.set_xlabel("Pain Score (Positive Cases Only)")
    ax2.legend(fontsize=8)

    p2_path = os.path.join(plots_dir, "02_pain_score_hurdle_comparison.png")
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f"Saved: {p2_path}")

    # -------------------------------------------------------------
    # Plot 3: Correlation Matrices & Absolute Drift
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    holdout_corr = df_holdout[APPROVED_FEATURES].corr()
    v2_corr = df_v2[APPROVED_FEATURES].corr()
    final_corr = df_final[APPROVED_FEATURES].corr()

    sns.heatmap(holdout_corr, ax=axes[0], cmap="vlag", vmin=-0.6, vmax=0.6, cbar=False, annot=False)
    axes[0].set_title("Real Holdout Correlation Matrix", fontweight="bold")

    sns.heatmap(np.abs(v2_corr - holdout_corr), ax=axes[1], cmap="Reds", vmin=0, vmax=0.3, cbar=True)
    axes[1].set_title("GC V2 Absolute Correlation Error", fontweight="bold")

    sns.heatmap(np.abs(final_corr - holdout_corr), ax=axes[2], cmap="Reds", vmin=0, vmax=0.3, cbar=True)
    axes[2].set_title("GC Final Absolute Correlation Error", fontweight="bold")

    p3_path = os.path.join(plots_dir, "03_correlation_heatmaps_and_drift.png")
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f"Saved: {p3_path}")

    # -------------------------------------------------------------
    # Plot 4: Downstream Utility (ROC-AUC and PR-AUC for Diabetes)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    feature_cols = [c for c in APPROVED_FEATURES if c != "diabetes"]
    X_test = df_holdout[feature_cols].values
    y_test = df_holdout["diabetes"].values.astype(int)

    train_sources = {
        "Real Train": (df_train, colors["Real_Holdout"]),
        "GC V1": (df_v1, colors["GC_V1"]),
        "GC V2": (df_v2, colors["GC_V2"]),
        "GC Final": (df_final, colors["GC_Final"]),
    }

    for name, (src_df, clr) in train_sources.items():
        X_tr = src_df[feature_cols].values
        y_tr = src_df["diabetes"].values.astype(int)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_te_s)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", color=clr, lw=2.0)

        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val = average_precision_score(y_test, probs)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_val:.3f})", color=clr, lw=2.0)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("TSTR: Downstream ROC Curve (Predicting Diabetes)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=8)

    ax2.axhline((y_test == 1).mean(), color="k", ls="--", alpha=0.5, label=f"Random Chance ({((y_test == 1).mean()*100):.1f}%)")
    ax2.set_title("TSTR: Downstream Precision-Recall Curve", fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8)

    p4_path = os.path.join(plots_dir, "04_discriminator_and_utility_curves.png")
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print(f"Saved: {p4_path}")

    print("\nAll publication-grade plots generated successfully.")


if __name__ == "__main__":
    main()
