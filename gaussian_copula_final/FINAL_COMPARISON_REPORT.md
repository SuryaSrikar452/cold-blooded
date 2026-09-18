# FINAL_COMPARISON_REPORT.md: Multi-Model Tradeoff Analysis

**Project:** SH-405 — Synthetic Patient Data Generation for Clinical Research  
**Date:** 2026-09-18  
**Evaluation Standard:** Objective Scientific Benchmarking Without Arbitrary "Winner" Scoring  

---

## 1. Multi-Model Dimensional Comparison

To provide transparent decision support for researchers, this report compares all generative model architectures evaluated in the SH-405 project across 8 core operational and statistical dimensions:

| Dimension | Real Reference (Train / Holdout) | CTGAN V1 (Audited Neural) | Gaussian Copula V1 (Audited Baseline) | Gaussian Copula V2 (Experimental) | Gaussian Copula Final (Production Hardened) |
|---|---|---|---|---|---|
| **1. Marginal Fidelity** | Natural sampling divergence: KS = 0.0244, Wass = 20.17 | Poor on bounded/skewed variables: KS = 0.2029, Wass = 420.41 | Moderate: KS = 0.1239; failed on pain hurdle (KS = 0.3918) | Strong: KS = 0.0454; Beta age struggled with multi-modal peaks (KS = 0.1055) | **Exceptional:** KS = **0.0369**, Wass = **32.46**; Age KS = **0.0359**, Meds KS = **0.0181** |
| **2. Joint Fidelity & Correlation** | Natural sampling drift: Pearson = 0.0197, Spearman = 0.0191 | High correlation distortion: Pearson drift = 0.1394, Spearman = 0.1408 | Moderate: Pearson drift = 0.0533; correlations with diabetes attenuated | Regressed: Pearson drift = 0.0690 due to naive uniform jittering | **Restored:** Pearson drift = **0.0301** (56% drop over V2); Spearman drift = **0.0258** |
| **3. Clinical Validity** | 100.00% valid records | Poor: **64.94%** valid; produced negative activity and DBP $\ge$ SBP | High: 99.88% valid via post-hoc hard clipping | 100.00% valid via bounded distributions and rejection sampling | **100.00% valid** via bounded distributions and hemodynamic rejection sampling |
| **4. Downstream Utility (TSTR)** | Reference: PR-AUC = 0.6355, ROC-AUC = 0.9373, F1 = 0.5668 | Lower: PR-AUC = 0.5688, ROC-AUC = 0.9080, F1 = 0.5342 | Moderate: PR-AUC = 0.6460, ROC-AUC = 0.9235, F1 = 0.4662 | Moderate: PR-AUC = 0.6257, ROC-AUC = 0.9277, F1 = 0.4511 | **Superior:** PR-AUC = **0.6482**, ROC-AUC = **0.9375**, F1 = **0.6071** |
| **5. Empirical Privacy** | Baseline DCR to size-matched holdout = 0.4057 | Zero exact matches; DCR p5 = 0.5008 | Zero exact matches; DCR p5 = 0.5528 | Zero exact matches; DCR p5 = 0.4837 | Zero exact matches; DCR p5 size-matched = **0.4529 ± 0.0099** (matches holdout 0.4562) |
| **6. Training / Sampling Runtime** | N/A | Slow: ~45 seconds GPU / 180 seconds CPU | Fast: ~0.08 seconds fit / 0.12 seconds sample | Fast: ~0.02 seconds fit / 0.04 seconds sample | **Ultra-Fast:** **0.006s** fit / **0.018s** sample ($N=4,826$) |
| **7. Mathematical Interpretability** | Empirical ground truth | Black-box neural network with complex latent generators | High: standard parametric Gaussian copula | High: hybrid copula with hurdle pain | **High:** rank-calibrated Gaussian copula with explicit analytical biserial multipliers |
| **8. Implementation Complexity** | Standard database | High: PyTorch, WGAN loss, PacGAN discriminator, GMM clusters | Minimal: SDV library out-of-the-box defaults | Moderate: custom hurdle and continuous transforms | **Moderate:** fully standalone implementation, zero heavy external framework dependencies |

---

## 2. Tradeoff Analysis: Why Gaussian Copula Final is the Preferred Architecture

Rather than assigning arbitrary subjective weights to declare an artificial "winner", we analyze the specific engineering and scientific tradeoffs:

### Tradeoff 1: Deep Neural Networks (CTGAN) vs. Calibrated Hybrid Copula (GC Final)
- **CTGAN Advantages:** Capable of modeling non-linear, non-Gaussian manifold shapes without pre-specifying parametric distribution families.
- **CTGAN Drawbacks:** Suffers from severe boundary bleed ($35\%$ invalid clinical records), fails on zero-inflation (generated 0 exact zeros for pain score), requires high computational overhead, and produces significant correlation drift ($0.1394$).
- **GC Final Advantage:** 100% clinical validity, ultra-fast sampling ($0.018$s), exact zero-inflation preservation ($68.5\%$ pain zeros), and lower discriminator distinguishability ($0.5308$ vs $0.7646$).

### Tradeoff 2: Parametric Baseline (GC V1) vs. Hardened Hybrid (GC Final)
- **GC V1 Advantages:** Simplest implementation using out-of-the-box SDV defaults.
- **GC V1 Drawbacks:** Blanket normal distribution produced negative ages, negative pain scores, and impossible medication counts that required ad-hoc post-clipping. Under-represented diabetes prevalence ($7.83\%$ vs $9.95\%$) and suffered high marginal error on zero-inflated variables (Pain KS = $0.3918$).
- **GC Final Resolution:** Replaces blanket normals with variable-specific marginals and a two-part hurdle model, dropping continuous KS error by $70\%$ (from $0.1239$ to $0.0369$).

### Tradeoff 3: Experimental Version (GC V2) vs. Production Hardened (GC Final)
- **GC V2 Regressions:** While V2 successfully introduced the hurdle model, its reliance on naive independent uniform jittering severely attenuated latent correlations (Pearson drift worsened to $0.0690$). Furthermore, V2 used deterministic rank partitioning and post-hoc record flipping to force diabetes prevalence.
- **GC Final Resolution:** Replaced jittering with Spearman rank inversion and polyserial calibration ($1/\lambda$). Pearson drift dropped by $56\%$ (to $0.0301$), key clinical associations were restored (e.g., diabetes $\leftrightarrow$ medications jumped from $+0.127$ to $+0.544$), and deterministic prevalence forcing was eliminated in favor of natural latent thresholding.

---

## 3. Scientific Recommendation for SH-405

Based on the empirical evidence across marginal fidelity, multivariate dependence, clinical safety, computational speed, and downstream utility:

**`GaussianCopulaFinal` is the recommended, production-hardened generator for the SH-405 project.**

It provides:
1. **Mathematical Rigor:** Analytical polyserial calibration over rank-inverted copula dependencies.
2. **Clinical Safety:** 100% compliance with domain physiological constraints.
3. **High Downstream Utility:** Highest minority-class TSTR PR-AUC ($0.6482$) and F1 score ($0.6071$).
4. **Transparent Governance:** Complete disclosure of benchmark-augmented variables, heuristic privacy boundaries, and zero-leakage training isolation.
