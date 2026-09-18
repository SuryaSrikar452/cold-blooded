# FINAL_LIMITATIONS.md: Methodological Assumptions & Scientific Limitations

**Project:** SH-405 — Synthetic Patient Data Generation for Clinical Research  
**Target Model:** `GaussianCopulaFinal`  
**Date:** 2026-09-18  

---

## 1. Intended Purpose & Boundary of Use

`GaussianCopulaFinal` is engineered strictly as a statistical benchmarking tool for developing machine learning algorithms, simulating clinical trial enrollment criteria, and validating healthcare data pipelines without exposing real patient protected health information (PHI).

It is **NOT** designed, validated, or approved for:
- Direct patient medical diagnosis, clinical triage, or treatment selection.
- Pharmacological dose titration or drug efficacy evaluations.
- Real-world epidemiological surveillance or policy decision-making without external verification.

---

## 2. Benchmark-Augmented Variables

The approved 9-feature schema contains two engineered variables:
- **`adherence_pct`**: Represents hypothetical patient compliance with prescribed therapy ($0.0\%$ to $100.0\%$).
- **`pain_score`**: Represents a numeric pain rating scale ($0.0$ to $10.0$).

> [!WARNING]
> Both `adherence_pct` and `pain_score` are **benchmark-augmented variables**. They are **not** native laboratory or physical examination measurements from the CDC NHANES continuous survey cycles. They were mathematically synthesized to benchmark longitudinal adherence and chronic symptom modeling. Any study citing these variables must explicitly disclose their benchmark-augmented provenance.

---

## 3. Cross-Sectional Nature of Generative Baseline

All records produced by `GaussianCopulaFinal` represent single, cross-sectional snapshots of patient state at one observation timepoint, reflecting the cross-sectional survey architecture of NHANES.

- **Longitudinal Extension Engine:** Multi-visit synthetic patient trajectories generated in downstream project modules (`longitudinal_generation/`) use an explicit autoregressive temporal persistence model ($\rho \approx 0.80\text{--}0.85$).
- **Scientific Limitation:** These trajectories are **not** learned from longitudinal follow-up of real patient cohorts. They represent transparent persistence simulations and must not be cited as empirical human disease trajectories.

---

## 4. Conditional Cohort Sampling vs. Causal Inference

The conditional generation API enables researchers to draw synthetic cohorts meeting specific inclusion criteria (e.g., *"Generate 1,000 diabetic patients aged 50–70 with SBP between 130 and 170"*).

> [!CAUTION]
> This represents **observational conditioning**, NOT **causal intervention**:
> - Generating patients conditioned on `diabetes = 1` yields the associated correlation structure observed in survey data.
> - It does **NOT** establish or simulate a causal counterfactual (e.g., *"Inducing diabetes causes blood pressure to increase by X mmHg"*).
> - No structural causal model (SCM) or do-calculus operations are implemented in this version.

---

## 5. Statistical & Parametric Trade-offs

1. **Stochastic Diabetes Prevalence:**
   - Previous experimental versions (V2) forced exactly $9.95\%$ prevalence via deterministic rank partitioning and post-hoc record flipping.
   - To eliminate artificial distortion of multivariate conditioning, `GaussianCopulaFinal` uses natural latent Gaussian thresholding.
   - Consequently, diabetes prevalence in any finite sample fluctuates around the expectation ($9.32\%$ for $N=4,826$ with seed 42).
2. **Polyserial Bivariate Distortion:**
   - Applying analytical polyserial scaling ($1 / \lambda_D$) restores the clinical correlation between diabetes and continuous features (e.g., age, SBP, medications).
   - However, the interaction between two hurdle/discrete variables (`diabetes` and `pain_score`) is slightly amplified in the synthetic cohort ($+0.340$ vs real $+0.263$).
3. **Survey Weight Omission:**
   - The generative model was fitted on raw NHANES feature vectors without incorporating complex survey design strata or sampling weights (WTINT2YR / WTMEC2YR). Synthetic population statistics reflect the sample cohort, not the weighted U.S. civilian population.

---

## 6. Privacy & Legal Disclaimers

1. **Heuristic Privacy Assessment:**
   - Nearest-neighbor analysis and exact feature-vector matching confirmed zero duplicates and healthy minimum distances ($DCR_{p5} = 0.4529$).
   - This constitutes an **empirical memorization risk diagnostic**; it does **NOT** constitute mathematical Differential Privacy ($\epsilon, \delta$).
2. **No HIPAA Safe Harbor Certification:**
   - The synthetic data is composed of realistic feature vectors that do not map to real individuals. However, no formal statistical disclosure limitation (SDL) certification or formal HIPAA Safe Harbor expert determination has been performed.
   - Users must adhere to standard data governance protocols when sharing synthetic clinical files.
