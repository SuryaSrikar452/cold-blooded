# SH-405 Gaussian Copula Baseline Evaluation Report

**Project:** SH-405 Synthetic Patient Data Generation for Clinical Research  
**Source Dataset:** National Health and Nutrition Examination Survey (NHANES) 2011–2012 (Cycle G)  
**Experiment Role:** BASELINE Experiment  
**Random Seed:** 42  
**Date:** September 18, 2026  

---

## 1. Executive Summary

This report presents the complete empirical benchmark evaluation of the **Gaussian Copula Baseline** for the SH-405 project. The baseline was trained strictly on `final_training_data/nhanes_generative_train.csv` ($N = 4,826$) and evaluated against the held-out real dataset `final_training_data/nhanes_real_holdout.csv` ($N = 1,207$).

The model captures smooth continuous marginal distributions (e.g., `systolic_bp` KS stat = 0.0185, $p = 0.8867$) and binary categorical ratios (`diabetes` JSD = 0.0000), while achieving 99.98% clinical rule compliance. However, due to parametric Gaussian copula assumptions, it underperforms on zero-inflated distributions (`pain_score` KS stat = 0.3354, `n_medications` KS stat = 0.1520) and fails to preserve critical non-linear tail correlations, resulting in a real-vs-synthetic discriminator ROC-AUC of **0.8492**.

---

## 2. Dataset

- **Training File**: `final_training_data/nhanes_generative_train.csv`
  - **SHA-256 Hash**: `690321d829d262dfa0d77a91d7d7159457dcc75541fc345f95098638c89a2866`
  - **Row Count**: 4,826
- **Real Holdout File**: `final_training_data/nhanes_real_holdout.csv`
  - **SHA-256 Hash (Pre & Post Experiment)**: `345a320bb2ab5b57093f54f1791736fadfa2c637646412308560f17015aadf71` (**100% Hash Unchanged**)
  - **Row Count**: 1,207
- **Synthetic Dataset**: `gaussian_copula_pipeline/synthetic/gaussian_copula_synthetic.csv`
  - **Generated Rows**: 4,826
  - **SHA-256 Hash**: `e828fcb0a8ef184aef5cf01bd5ef72bcf92be20bebfddcf5dd626b1d4ef63283`

---

## 3. Feature Schema

The feature schema strictly follows `final_training_data/final_schema.json`:

| Feature Name | Data Type | Modeling Type | Domain Range / Categories | Description |
| :--- | :--- | :--- | :--- | :--- |
| `age` | `int64` | Continuous | $[8, 80]$ | Age in years at examination (top-coded at 80) |
| `sex` | `int64` | Categorical | $\{1, 2\}$ | Biological sex: 1 = Male, 2 = Female |
| `diabetes` | `int64` | Categorical | $\{0, 1\}$ | Doctor-diagnosed diabetes indicator |
| `systolic_bp` | `int64` | Continuous | $[74, 233]$ | Systolic blood pressure in mmHg |
| `diastolic_bp` | `int64` | Continuous | $[14, 116]$ | Diastolic blood pressure in mmHg ($>0, < \text{SBP}$) |
| `activity_mims` | `int64` | Continuous | $[118, 33379]$ | Triaxial physical activity MIMS count |
| `n_medications` | `int64` | Continuous | $[0, 19]$ | Count of reported prescription drugs |
| `adherence_pct` | `float64` | Continuous | $[5.0, 95.0]$ | Medication adherence percentage |
| `pain_score` | `float64` | Continuous | $[0.0, 9.6]$ | Numeric pain score (0–10 NRS) |

---

## 4. Feature Provenance

Every feature is attributed to its validated source level:

- **`DIRECT_NHANES`**: `age`, `sex` (extracted directly from `DEMO_G.xpt`).
- **`DERIVED_FROM_NHANES`**: `diabetes` (from `DIQ_G.xpt`), `systolic_bp` & `diastolic_bp` (from `BPX_G.xpt` with AHA protocol cleaning), `activity_mims` (from `PAXDAY_G.xpt`), `n_medications` (from `RXQ_RX_G.xpt`).
- **`BENCHMARK_AUGMENTED`**: `adherence_pct` and `pain_score` (from `nhanes_complete_seed.csv`).

> [!IMPORTANT]
> **Provenance Distinction**: `adherence_pct` and `pain_score` are **BENCHMARK_AUGMENTED** features created specifically for the SH-405 hackathon benchmark. They are NOT direct NHANES physical examination survey items.

---

## 5. Gaussian Copula Method & Configuration

The Gaussian Copula baseline maps marginal probability distributions to standard normal space using the probability integral transform ($\Phi^{-1}(F(X))$), models dependence via a joint multivariate Gaussian covariance matrix $\mathbf{\Sigma}$, and samples synthetic vectors by transforming multivariate normal samples back through inverse marginal distributions ($F^{-1}(\Phi(Z))$).

- **Implementation**: SDV `GaussianCopulaSynthesizer` (v1.38.3)
- **Metadata Spec**: `V1` Single Table Metadata
- **Random Seed**: `42`
- **Default Distribution**: `gaussian`
- **Min/Max Enforcement**: `enforce_min_max_values = True`
- **Training Time**: $1.8196$ seconds
- **Generation Time**: $0.1152$ seconds ($4,826$ rows)

---

## 6. Continuous Distribution Fidelity

Comparative statistics evaluated against **REAL HOLDOUT** ($N = 1,207$):

| Feature | Real Mean $\pm$ SD | Syn Mean $\pm$ SD | Real Median | Syn Median | KS Stat | KS $p$-value | Wasserstein Dist |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`age`** | $38.86 \pm 22.95$ | $39.46 \pm 22.18$ | $37.0$ | $38.0$ | $0.0375$ | $0.1278$ | $1.4445$ |
| **`systolic_bp`** | $118.52 \pm 17.65$ | $118.25 \pm 17.27$ | $116.0$ | $116.0$ | $0.0185$ | $0.8867$ | $0.6065$ |
| **`diastolic_bp`**| $65.88 \pm 13.91$ | $66.19 \pm 13.56$ | $68.0$ | $67.0$ | $0.0561$ | $0.0044$ | $1.2324$ |
| **`activity_mims`**| $11116.3 \pm 4104.9$| $11054.5 \pm 3971.2$| $11139.0$| $11005.0$| $0.1062$ | $<0.0001$ | $742.5821$ |
| **`n_medications`**| $1.64 \pm 2.45$ | $1.62 \pm 1.86$ | $0.0$ | $1.0$ | $0.1520$ | $<0.0001$ | $0.7878$ |
| **`adherence_pct`**| $43.83 \pm 16.92$ | $44.02 \pm 16.71$ | $44.3$ | $44.0$ | $0.0376$ | $0.1275$ | $1.3453$ |
| **`pain_score`** | $1.04 \pm 1.77$ | $1.02 \pm 0.99$ | $0.0$ | $0.9$ | $0.3354$ | $<0.0001$ | $0.3621$ |

---

## 7. Categorical Fidelity

Frequency distribution comparison against **REAL HOLDOUT**:

| Feature | Category | Real Count (%) | Syn Count (%) | Abs Diff (%) | JSD | Chi-square Stat ($p$-value) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`sex`** | 1 (Male) | 604 (50.04%) | 2,370 (49.11%) | 0.93% | $0.000011$ | $0.4383$ ($p = 0.5080$) |
| | 2 (Female)| 603 (49.96%) | 2,456 (50.89%) | 0.93% | | |
| **`diabetes`**| 0 (No) | 1,086 (89.98%)| 4,342 (89.97%)| 0.01% | $0.000000$ | $0.0001$ ($p = 0.9924$) |
| | 1 (Yes) | 121 (10.02%) | 484 (10.03%) | 0.01% | | |

---

## 8. Relationship Preservation

- **Pearson Correlation MACD**: **$0.0841$** (Max Abs Diff = $0.4185$ for `adherence_pct` $\leftrightarrow$ `n_medications`)
- **Spearman Rank MACD**: **$0.0855$** (Max Abs Diff = $0.3245$ for `n_medications` $\leftrightarrow$ `diabetes`)

---

## 9. Clinical Constraint Validity

- **Overall Satisfying ALL Constraints**: **99.98%**

---

## 10. Real-vs-Synthetic Classification

- **ROC-AUC**: **$0.8492$**
- **Accuracy**: **$0.7664$**
- **F1 Score**: **$0.7484$**

---

## 11. Privacy Risk Analysis

- **Minimum Distance**: $0.1456$
- **Exact Matches with Real Holdout**: **$0$** ($0.00\%$)
- **Exact Matches with Real Training**: **$0$** ($0.00\%$)

---

## 12. Downstream Utility

- **Model A: Real Training Data**: ROC-AUC = **$0.9278$**, Accuracy = **$0.9296$**, F1 = **$0.5596$**
- **Model B: Gaussian Copula Synthetic**: ROC-AUC = **$0.8680$**, Accuracy = **$0.8998$**, F1 = **$0.0000$**

---

## 13. Limitations

1. **Cross-Sectional Sampling**: NHANES 2011–2012 source data are strictly cross-sectional.
2. **Benchmark-Augmented Variables**: `adherence_pct` and `pain_score` are benchmark-augmented features.
3. **Zero-Inflation Smoothing**: Standard copula smooths zero-inflated features (`pain_score`, `n_medications`).
4. **Heuristic Privacy Scope**: Nearest-neighbor analysis is empirical and heuristic.
