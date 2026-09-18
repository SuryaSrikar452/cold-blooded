# FINAL_MODEL_PRE_CHANGE_AUDIT.md: Pre-Change Audit of Gaussian Copula V2

**Author:** Senior ML Engineering & Audit Team  
**Project:** SH-405 — Synthetic Patient Data Generation for Clinical Research  
**Timestamp:** 2026-09-18  
**Audit Objective:** Rigorously verify Gaussian Copula V2 source code, configuration, generated model, synthetic data, and evaluation artifacts against independent technical reviews from Claude and Gemini before implementing any code changes.

---

## 1. Executive Summary & Reviewer Claims Classification

The independent reviews from Claude and Gemini provided substantial critique of both the original Gaussian Copula V1 baseline and the experimental Gaussian Copula V2 implementation. We systematically evaluated each claim against the ground-truth code, serialized models, and empirical datasets.

| # | Reviewer Finding / Claim | Classification | Empirical Verification & Root-Cause Analysis |
|---|---|:---:|---|
| **1** | Blanket normal distribution across all variables in V1 was statistically flawed. | **VERIFIED** | In V1 config (`training_pipeline/configs/gaussian_copula_config.json`), `default_distribution: "norm"` was used for all 9 features, producing negative values for bounded non-negative features like `pain_score`, `activity_mims`, and `n_medications`. |
| **2** | `adherence_pct` is bounded on $[0, 100]$. | **VERIFIED** | Training data empirical range is $[5.0, 95.0]$. However, V1 produced values outside $[0, 100]$ requiring post-hoc hard clipping. In V2, Truncated Normal on $[0, 100]$ was fitted. Notably, training data contains exactly **0** records at $0.0$ and **0** records at $100.0$. |
| **3** | `n_medications` is a discrete, non-negative integer count with 52.2% zeros. | **VERIFIED** | Training data confirms $2,519$ of $4,826$ rows ($52.20\%$) have `n_medications = 0`. Non-zero values are integers from $1$ to $19$. V1 generated fractional negative floats. |
| **4** | `pain_score` has ~69% exact zeros and requires a two-part hurdle model. | **VERIFIED** | Training data contains exactly $3,354$ zeros ($69.50\%$). Positive pain scores range strictly between $1.0$ and $9.6$. V1 generated a continuous Gaussian centered near $1.0$ with $28\%$ negative values. |
| **5** | V1 diabetes prevalence was $7.83\%$ vs $9.95\%$ in real training data. | **VERIFIED** | Real training prevalence is $480 / 4,826 = 9.946\%$. V1 produced $378 / 4,826 = 7.83\%$ due to uncalibrated latent normal thresholding. |
| **6** | V2 positive pain magnitude must retain appropriate joint dependence. | **VERIFIED** | V2's hurdle implementation linked positive pain to the latent normal variable $Z_{\text{pain}}$, but naive uniform jittering severely attenuated its correlation with other variables (e.g., real positive pain correlation with diabetes is $+0.192$, but V2 dropped to $+0.044$). |
| **7** | V2 used deterministic rank assignment and post-hoc adjustment for diabetes prevalence. | **VERIFIED** | In `gaussian_copula_v2/copula_v2.py` (lines 253–255 and 374–391), V2 used `np.partition` to force top $N \times p$ latent values to 1, and post-hoc flipped non-diabetics with highest SBP/meds if rejection sampling altered the count. This artificially distorted the joint distribution. |
| **8** | Models should not rely on arbitrary clipping as the primary modeling mechanism. | **VERIFIED** | V1 relied entirely on post-hoc clipping. V2 improved this by using bounded distributions and rejection sampling, but still applied post-hoc sorting/flipping for diabetes. |
| **9** | Age support in training data is 8–80, not 18–85. | **VERIFIED** | Empirical training minimum age is $8$ (pediatric NHANES inclusion) and maximum is $80$ (top-coded in NHANES). V2 correctly used $[8, 80]$ with a Beta distribution. |
| **10** | PSD correction on correlation matrix must be verified. | **PARTIALLY VERIFIED** | Reviewers raised concerns about PSD violation. When audited, V2's empirical latent matrix $R$ had eigenvalues ranging from $0.330$ to $2.394$; all eigenvalues were already strictly positive. The Higham/eigh clipping at $10^{-4}$ was therefore harmless but redundant. |
| **11** | V2 Pearson correlation drift worsened ($0.0567$ vs $0.0244$ in V1). | **VERIFIED** | Verified. While V2 reduced marginal KS error by $63\%$, correlation drift increased by $132\%$. Root cause: independent uniform jittering injected massive white noise into the $69.5\%$ zero-mass of pain and $52.2\%$ zero-mass of medications, attenuating latent correlations toward zero. |
| **12** | Downstream utility must report PR-AUC alongside ROC-AUC. | **VERIFIED** | Diabetes prevalence is $\sim 10\%$. ROC-AUC is misleadingly resilient to false positives in class-imbalanced settings. PR-AUC is mathematically required for honest assessment. |
| **13** | Discriminator AUC 0.74 should not be called "indistinguishable". | **VERIFIED** | AUC 0.74 indicates a classifier correctly distinguishes synthetic from real records $74\%$ of the time. Describing this as "indistinguishable" is scientifically inaccurate. Neutral phrasing ("reduced distinguishability") is required. |
| **14** | Longitudinal model must be described honestly. | **VERIFIED** | SH-405 longitudinal data is generated via an autoregressive Gaussian persistence process from cross-sectional NHANES baseline, not learned from longitudinal patient follow-up. |

---

## 2. Deep-Dive Audit of Actual V2 Source Code & Architecture

### 2.1 Actual Distributions and Bounds Fitted in V2

Inspection of `gaussian_copula_v2/copula_v2.py` and `gaussian_copula_v2/models/metadata.json` confirms:
1. **`age`**: Beta distribution fitted on $[8.0, 80.0]$: $\alpha = 0.8173, \beta = 1.0963$. Generates integer values in $[8, 80]$.
2. **`sex`**: Bernoulli discrete marginal with $p_{\text{male}} = 0.5002$.
3. **`diabetes`**: Bernoulli with target prevalence $p_{\text{diab}} = 0.0995$.
4. **`systolic_bp`**: Lognormal distribution ($s = 0.1478, \text{scale} = 117.06$). Strictly positive.
5. **`diastolic_bp`**: Truncated Normal on $[10.0, \infty)$ ($\mu = 66.86, \sigma = 13.19$).
6. **`activity_mims`**: Truncated Normal on $[0.0, \infty)$ ($\mu = 11037.7, \sigma = 3987.9$).
7. **`n_medications`**: Empirical discrete inverse CDF over $\{0, 1, \dots, 19\}$, with $52.20\%$ mass at zero.
8. **`adherence_pct`**: Truncated Normal on $[0.0, 100.0]$ ($\mu = 43.51, \sigma = 17.33$).
9. **`pain_score`**: Two-part hurdle model: $P(\text{pain} = 0) = 0.6950$, positive magnitude modeled as Truncated Normal on $[1.0, 10.0]$ ($\mu = 3.286, \sigma = 1.705$).

### 2.2 Discrepancies & Flaws Identified in V2

#### Issue A: Correlation Attenuation via Independent Uniform Jittering
In V2's `fit()` method:
```python
# For diabetes:
u_diab = np.where(diab_vals == 0, rng.uniform(1e-5, p_nodiab, size=n), rng.uniform(p_nodiab, 1.0 - 1e-5, size=n))

# For pain_score:
zero_mask = df_train["pain_score"].values == 0
u_pain[zero_mask] = rng.uniform(1e-5, p0_pain, size=zero_mask.sum())
```
Because $69.5\%$ of pain scores and $90.05\%$ of diabetes values are assigned independent uniform pseudo-random noise, the computed Pearson correlation between $Z_{\text{diab}}$ or $Z_{\text{pain}}$ and the other variables was severely diluted:
- Real `age` $\leftrightarrow$ `diabetes` Pearson: **+0.343** $\rightarrow$ V2 Latent $R$: **+0.180**
- Real `n_medications` $\leftrightarrow$ `diabetes` Pearson: **+0.491** $\rightarrow$ V2 Latent $R$: **+0.239**
- Real `diabetes` $\leftrightarrow$ `pain_score` Pearson: **+0.263** $\rightarrow$ V2 Latent $R$: **+0.117**

#### Issue B: Deterministic Diabetes Stratification and Post-Hoc Record Flipping
In V2's `_sample_batch`:
```python
diab_cut = np.partition(Z_sample[:, 2], n - n_diab)[n - n_diab]
synth["diabetes"] = (Z_sample[:, 2] >= diab_cut).astype(int)
```
And in `sample`:
```python
if current_diab != desired_diab:
    diff = desired_diab - current_diab
    if diff > 0:
        cand_idx = final_df[final_df["diabetes"] == 0].sort_values(
            by=["n_medications", "systolic_bp"], ascending=False
        ).index[:diff]
        final_df.loc[cand_idx, "diabetes"] = 1
```
This post-hoc rule artificially selected records with highest medications and SBP to flip them to diabetic if rejection sampling changed the total. While motivated by preserving prevalence, this is an ad-hoc heuristic that distorts multivariate conditioning.

#### Issue C: Biserial Attenuation under Dichotomization
In probability theory, dichotomizing a standard Gaussian latent variable at threshold $\tau = \Phi^{-1}(1 - p)$ attenuates the observable linear correlation by a factor:
$$\lambda = \frac{\phi(\tau)}{\sqrt{p(1 - p)}}$$
For diabetes ($p = 0.09946$, $\tau \approx 1.285$):
$$\lambda \approx \frac{\phi(1.285)}{\sqrt{0.0995 \times 0.9005}} = \frac{0.1749}{0.2993} \approx 0.5842$$
This means any latent correlation $\rho_Z$ is mechanically attenuated to $0.5842 \times \rho_Z$ in the observable binary variable. In V2, not only was latent $R$ diluted by jittering, but this biserial attenuation further crushed the correlation.

---

## 3. Data Hygiene & Leakage Audit

We audited all files in `gaussian_copula_v2/` and confirmed:
- `train_v2.py` strictly loaded `final_training_data/nhanes_generative_train.csv` (N=4,826) for fitting.
- `nhanes_real_holdout.csv` was **NEVER** imported or referenced in `copula_v2.py` or `train_v2.py`.
- `evaluate_v2.py` imported `nhanes_real_holdout.csv` strictly as an unmodified read-only benchmark for out-of-sample KS and discriminator evaluation.
- **Verdict**: Data hygiene in V2 was strictly zero-leakage. This standard is fully preserved in `gaussian_copula_final/`.

---

## 4. Architectural Decisions for `gaussian_copula_final/`

Based on this pre-change audit, the final hardened model will implement the following mathematically rigorous upgrades:

1. **Eliminate Naive Independent Jittering**:
   - Replace independent noise injection with rank-based / biserial-calibrated latent Gaussian dependence.
   - Use Spearman rank inversion $R_{jk} = 2 \sin(\frac{\pi}{6} \rho_s)$ and Kendall inversion $R_{jk} = \sin(\frac{\pi}{2} \tau)$ with biserial/polyserial scaling factors for binary and count variables.
2. **Latent Bernoulli Thresholding for Diabetes**:
   - Use natural Gaussian thresholding at $\tau = \Phi^{-1}(1 - p_{\text{diab}})$.
   - Calibrate the latent correlation row for diabetes by $1 / \lambda = 1.7119$ to counteract biserial attenuation, allowing the observable correlation to match empirical clinical data without ad-hoc sorting or deterministic record-flipping.
3. **Joint Conditional Hurdle Pain Model**:
   - Maintain the $69.50\%$ zero point mass.
   - When positive, map latent $Z_{\text{pain}}$ through the conditional probability integral transform:
     $$U_{\text{pos}} = \frac{\Phi(Z_{\text{pain}}) - p_0}{1 - p_0} \in (0, 1)$$
     $$X_{\text{pain}} = F_{\text{pos}}^{-1}(U_{\text{pos}})$$
   - Ensure positive pain severity magnitude covaries appropriately with age, medications, and diabetes.
4. **Discrete Non-Negative Medication Modeling**:
   - Map $U_{\text{med}} = \Phi(Z_{\text{med}})$ into discrete counts via empirical inverse CDF $F_{\text{med}}^{-1}(u)$ without continuous extension noise, preserving integer support $\{0, \dots, 19\}$ and $52.20\%$ zero mass.
5. **Physiological Validity & Rejection Sampling**:
   - Reject and resample unphysiological records ($\text{DBP} \ge \text{SBP}$ or $\text{SBP} - \text{DBP} < 5\text{ mmHg}$) during generation, avoiding post-hoc clipping.
6. **Positive Semidefinite (PSD) Projection**:
   - Apply Higham nearest-correlation projection if any eigenvalue $< 10^{-4}$. Document minimum eigenvalue, Frobenius norm, and maximum element modification.
7. **Downstream Utility**:
   - Include both PR-AUC and ROC-AUC for diabetes prediction to provide honest assessment under class imbalance.

---

**Pre-Change Audit Status:** COMPLETE AND APPROVED FOR IMPLEMENTATION.
