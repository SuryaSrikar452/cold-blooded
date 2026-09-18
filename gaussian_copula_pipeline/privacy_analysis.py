import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("============================================================")
    print("SH-405 GAUSSIAN COPULA BASELINE: HEURISTIC PRIVACY RISK ANALYSIS")
    print("============================================================")
    
    train_path = 'final_training_data/nhanes_generative_train.csv'
    holdout_path = 'final_training_data/nhanes_real_holdout.csv'
    syn_path = 'gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.csv'
    plots_dir = 'gaussian_copula_pipeline/validation/plots/'
    
    real_train_df = pd.read_csv(train_path)
    real_holdout_df = pd.read_csv(holdout_path)
    syn_df = pd.read_csv(syn_path)
    
    continuous_cols = [
        'age', 'systolic_bp', 'diastolic_bp', 'activity_mims',
        'n_medications', 'adherence_pct', 'pain_score'
    ]
    categorical_cols = ['sex', 'diabetes']
    
    # Preprocessing for distance metric:
    # Scale continuous variables using StandardScaler fitted on real holdout
    scaler = StandardScaler()
    scaler.fit(real_holdout_df[continuous_cols])
    
    def transform_features(df):
        cont_scaled = scaler.transform(df[continuous_cols])
        # One-hot encode categorical features explicitly
        cat_onehot = pd.get_dummies(df[categorical_cols], columns=categorical_cols, drop_first=False).astype(float)
        # Ensure consistent column structure
        return np.hstack([cont_scaled, cat_onehot.values])

    # Build standardized feature matrices
    X_holdout_proc = transform_features(real_holdout_df)
    X_syn_proc = transform_features(syn_df)
    X_train_proc = transform_features(real_train_df)
    
    # 1. Nearest Neighbor Distance to Real Holdout
    print("Calculating Nearest-Neighbor distances from Synthetic to Real Holdout...")
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(X_holdout_proc)
    
    distances, _ = nn.kneighbors(X_syn_proc)
    distances = distances.flatten()
    
    min_dist = float(np.min(distances))
    p1_dist = float(np.percentile(distances, 1))
    p5_dist = float(np.percentile(distances, 5))
    p25_dist = float(np.percentile(distances, 25))
    median_dist = float(np.median(distances))
    mean_dist = float(np.mean(distances))
    p75_dist = float(np.percentile(distances, 75))
    p95_dist = float(np.percentile(distances, 95))
    max_dist = float(np.max(distances))
    
    pct_lt_0_1 = float((distances < 0.1).mean() * 100)
    pct_lt_0_5 = float((distances < 0.5).mean() * 100)
    pct_lt_1_0 = float((distances < 1.0).mean() * 100)
    
    print(f"Nearest-Neighbor Distance Summary (to Real Holdout):")
    print(f"  Min: {min_dist:.4f}")
    print(f"  1st Percentile: {p1_dist:.4f}")
    print(f"  5th Percentile: {p5_dist:.4f}")
    print(f"  Median: {median_dist:.4f}")
    print(f"  Mean:   {mean_dist:.4f}")
    print(f"  Max:    {max_dist:.4f}")
    print(f"  % with Distance < 0.1: {pct_lt_0_1:.2f}%")
    print(f"  % with Distance < 0.5: {pct_lt_0_5:.2f}%")
    print(f"  % with Distance < 1.0: {pct_lt_1_0:.2f}%")
    
    # 2. Exact Feature Vector Match Detection
    print("\nDetecting Exact Feature-Vector Matches...")
    # Exact match against Holdout
    holdout_matches = pd.merge(syn_df, real_holdout_df, how='inner')
    exact_holdout_matches = len(holdout_matches)
    
    # Exact match against Training
    train_matches = pd.merge(syn_df, real_train_df, how='inner')
    exact_train_matches = len(train_matches)
    
    print(f"  Exact Feature Vector Matches (Synthetic vs Real Holdout): {exact_holdout_matches}")
    print(f"  Exact Feature Vector Matches (Synthetic vs Real Train):   {exact_train_matches}")
    
    # 3. Save privacy_results.csv
    results_df = pd.DataFrame([
        {'metric_name': 'nearest_neighbor_min_distance', 'metric_value': round(min_dist, 4)},
        {'metric_name': 'nearest_neighbor_p1_distance', 'metric_value': round(p1_dist, 4)},
        {'metric_name': 'nearest_neighbor_p5_distance', 'metric_value': round(p5_dist, 4)},
        {'metric_name': 'nearest_neighbor_p25_distance', 'metric_value': round(p25_dist, 4)},
        {'metric_name': 'nearest_neighbor_median_distance', 'metric_value': round(median_dist, 4)},
        {'metric_name': 'nearest_neighbor_mean_distance', 'metric_value': round(mean_dist, 4)},
        {'metric_name': 'nearest_neighbor_max_distance', 'metric_value': round(max_dist, 4)},
        {'metric_name': 'pct_distance_lt_0_1', 'metric_value': round(pct_lt_0_1, 2)},
        {'metric_name': 'pct_distance_lt_0_5', 'metric_value': round(pct_lt_0_5, 2)},
        {'metric_name': 'pct_distance_lt_1_0', 'metric_value': round(pct_lt_1_0, 2)},
        {'metric_name': 'exact_feature_matches_holdout', 'metric_value': exact_holdout_matches},
        {'metric_name': 'exact_feature_matches_train', 'metric_value': exact_train_matches}
    ])
    
    csv_path = 'gaussian_copula_pipeline/validation/privacy_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved privacy results to {csv_path}.")
    
    # 4. Plot 13: Nearest-Neighbor Distance Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sns.histplot(distances, kde=True, ax=ax, color='#2ca02c', bins=40, stat='density', alpha=0.6)
    ax.axvline(median_dist, color='red', linestyle='--', label=f'Median Dist = {median_dist:.2f}')
    ax.axvline(p5_dist, color='orange', linestyle=':', label=f'5th Pct Dist = {p5_dist:.2f}')
    
    ax.set_title("Heuristic Privacy Analysis: Nearest-Neighbor Distance Distribution", fontsize=11, fontweight='bold')
    ax.set_xlabel("Standardized Distance to Nearest Real Holdout Row")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, '13_nearest_neighbor_distance.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved nearest neighbor distance plot to {plot_path}.")

if __name__ == '__main__':
    main()
