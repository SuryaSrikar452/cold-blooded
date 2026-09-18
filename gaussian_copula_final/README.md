# Gaussian Copula Final: Production-Hardened Generative Architecture

**Project:** SH-405 — Synthetic Patient Data Generation for Clinical Research  
**Version:** `final_hardened`  
**Date:** 2026-09-18  

---

## 1. Overview & Architecture

`gaussian_copula_final` is the production-grade, hardened synthetic patient generator for SH-405. It replaces the baseline Gaussian Copula (V1) and experimental version (V2) with a mathematically coherent **Hybrid Gaussian-Copula / Hurdle Generative Architecture** that resolves key statistical limitations identified during independent scientific reviews.

### Core Mathematical Upgrades
1. **Polyserial Rank-Inverted Latent Copula**:
   - Computes latent Gaussian correlation using Spearman rank inversion: $R_{jk} = 2 \sin(\frac{\pi}{6} \rho_s)$.
   - Applies biserial attenuation calibration factors ($\lambda = \phi(\tau) / \sqrt{p(1-p)}$) to discrete/hurdle rows and columns.
   - Eliminates the severe correlation attenuation caused by naive uniform jittering in V2 (Mean Pearson drift dropped from $0.0690$ in V2 down to **$0.0301$**).
2. **Two-Part Hurdle Model with Joint Copula Coupling**:
   - Accurately models the $69.50\%$ zero mass for `pain_score` while coupling positive pain severity to the joint latent Gaussian distribution via conditional probability integral transform.
3. **Natural Latent Bernoulli Thresholding**:
   - Eliminates deterministic rank clipping and post-hoc record flipping for diabetes, achieving natural stochastic prevalence ($9.32\%$) with strong multi-variable conditional dependence.
4. **Discrete Non-Negative Count & Demographic Demarcation**:
   - Inverts medication count and age via discrete empirical quantile functions, guaranteeing non-negative integer support with zero floating-point artifacts.
5. **100% Hemodynamic Plausibility via Rejection Sampling**:
   - Samples records satisfying strict physiological constraints ($\text{SBP} > \text{DBP}$, $\text{SBP} - \text{DBP} \ge 5\text{ mmHg}$, $\text{SBP}, \text{DBP} > 0$) without ad-hoc post-clipping.
6. **Strict Zero Data Leakage**:
   - Fitted exclusively on `final_training_data/nhanes_generative_train.csv` (N=4,826). The real holdout (N=1,207) is evaluated strictly once on frozen artifacts.

---

## 2. Directory Structure

```text
gaussian_copula_final/
├── __init__.py                                 # Package exports
├── config.json                                 # Model configuration and distribution specs
├── copula_final.py                             # Core GaussianCopulaFinal generator class
├── train_final.py                              # Training and cohort generation script
├── evaluate_final.py                           # Multi-model benchmark evaluation pipeline
├── generate_plots_final.py                     # Publication-quality plot generation script
├── test_copula_final.py                        # 12-point unit test suite
├── FINAL_MODEL_PRE_CHANGE_AUDIT.md             # Pre-change review verification audit
├── FINAL_MODEL_REPORT.md                       # Comprehensive multi-model benchmark report
├── FINAL_ADVERSARIAL_MODEL_AUDIT.md            # Skeptical adversarial audit and stress-test
├── README.md                                   # Directory overview and reproduction guide
├── models/
│   ├── model.pkl                               # Serialized fitted model (SHA-256: 80d1d654...)
│   └── metadata.json                           # Model parameters, runtimes, and PSD diagnostics
├── synthetic/
│   ├── gaussian_copula_final_synthetic.csv     # Generated synthetic cohort (N=4,826)
│   └── gaussian_copula_final_synthetic.parquet # Parquet format of synthetic cohort
└── outputs/
    ├── FINAL_MODEL_COMPARISON.csv              # Multi-model comparison summary table
    ├── detailed_per_feature_ks.csv             # Per-feature Kolmogorov-Smirnov statistics
    └── plots/
        ├── 01_marginal_distributions_comparison.png
        ├── 02_pain_score_hurdle_comparison.png
        ├── 03_correlation_heatmaps_and_drift.png
        └── 04_discriminator_and_utility_curves.png
```

---

## 3. Reproduction Commands

All steps are deterministic and independently reproducible:

```bash
# 1. Run unit tests
python -m unittest gaussian_copula_final/test_copula_final.py -v

# 2. Train model and generate synthetic cohort (N=4,826)
python gaussian_copula_final/train_final.py

# 3. Execute comprehensive benchmark evaluation
python gaussian_copula_final/evaluate_final.py

# 4. Generate comparison plots
python gaussian_copula_final/generate_plots_final.py
```

---

## 4. Cryptographic Provenance & Verification Hashes

### Frozen Baseline Artifacts (Verified 100% Byte-Identical)
- `final_training_data/nhanes_generative_train.csv`: `690321d829d262dfa0d77a91d7d7159457dcc75541fc345f95098638c89a2866`
- `final_training_data/nhanes_real_holdout.csv`: `345a320bb2ab5b57093f54f1791736fadfa2c637646412308560f17015aadf71`
- `training_pipeline/synthetic/ctgan_synthetic.csv`: `32f1d2eb2304a7a6e0c795e76ed5f9731d9367ecd1702337b1b4c53f92bbdd40`
- `training_pipeline/synthetic/gaussian_copula_synthetic.csv`: `0089e65d9aa4816ebc74c4eb44284d8b15d6b49d12bc444c87fc4f58e06889ff`
- `training_pipeline/models/ctgan/model.pkl`: `5599c103681bdfcaef1cb1b1d1506f85aa2f7178a8d52d45970c2c158cd447c1`
- `training_pipeline/models/gaussian_copula/model.pkl`: `5e5ac8f8a643f034003e80dc6e4d6ea9a800ca7c2d923e8dfa56aaadffa3ce30`

### Final Production Artifacts
- `gaussian_copula_final/models/model.pkl`: `80d1d6542d3c25c906544ec693bcb362e0130e5b96b3c76db7c8062fbdb76e43`
- `gaussian_copula_final/synthetic/gaussian_copula_final_synthetic.csv`: `e37779c2f6e7324b5feda31fb7c78d8b0f8026ad71653540eb0542d51733739c`
- `gaussian_copula_final/synthetic/gaussian_copula_final_synthetic.parquet`: `c5d6f6e4c17b857073de13858ad8cf1862ce13d313b1f0c748aed89d2552c446`
