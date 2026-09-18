# Verified References - SH-405 Gaussian Copula Baseline

This document provides verified scholarly and technical references supporting the methodology, implementation, and evaluation of the **Gaussian Copula Baseline** for the SH-405 Synthetic Patient Data Generation project.

---

## 1. Published Theoretical Methodology

1. **Sklar, A. (1959).** *Fonctions de répartition à $n$ dimensions et leurs marges.* Publications de l'Institut de Statistique de l'Université de Paris, 8, 229–231.
   - **Relevance**: Sklar's Theorem proves that any multivariate joint distribution function can be expressed in terms of univariate marginal distribution functions and a copula that binds them together.

2. **Nelsen, R. B. (2006).** *An Introduction to Copulas* (2nd ed.). Springer Series in Statistics. Springer, New York.
   - **Relevance**: Provides formal mathematical foundations for continuous copulas, parametric estimation of marginal distributions, and probability integral transformations.

3. **Clemen, R. T., & Reilly, T. (1999).** *Correlated Monte Carlo simulation and Gaussian copulas.* Management Science, 45(8), 1101–1124.
   - **Relevance**: Formulates practical applications of Gaussian Copulas for joint multivariate sampling and dependence preservation in simulation studies.

---

## 2. Software Implementation & Synthetic Data Frameworks

4. **Patki, N., Wedge, R., & Veeramachaneni, K. (2016).** *The Synthetic Data Vault.* IEEE International Conference on Data Science and Advanced Analytics (DSAA), 399–410.
   - **Relevance**: Introduces the SDV framework, specifying multivariate tabular data synthesis via Gaussian Copulas combined with parametric marginal transformations (e.g., Gaussian, Truncated Gaussian, Gamma).

5. **SDV Core Documentation (v1.38.3).** *Single Table Synthesis: Gaussian Copula Synthesizer.* Synthetic Data Vault Project, Data-to-AI Lab at MIT.
   - **URL**: [https://docs.sdv.dev/sdv/](https://docs.sdv.dev/sdv/)
   - **Relevance**: Official documentation for `sdv.single_table.GaussianCopulaSynthesizer` used in our execution pipeline.

---

## 3. Healthcare Synthetic Data Research & Benchmarking

6. **Rankin, D., Black, M., Bond, R., Wallace, J., Mulvenna, M., & Epelde, G. (2020).** *Reliability of synthetic data sets generated using synthetic data vault for clinical machine learning.* IEEE Journal of Biomedical and Health Informatics, 24(9), 2634–2641.
   - **Relevance**: Evaluates statistical fidelity, clinical logic preservation, and downstream utility of Gaussian Copula baseline synthesis in health and medical research.

7. **Tucker, A., Wang, Z., Rotalinti, Y., & Brett, P. (2020).** *Generating high-dimensional synthetic data using copulas and generative adversarial networks.* Nature Scientific Reports, 10, 8973.
   - **Relevance**: Establishes standard benchmarking protocols for comparing parametric copula baselines against neural generative architectures (e.g., CTGAN) on tabular EHR cohorts.

---

## 4. Disambiguation & Provenance Statement

- **Published Methodology**: Parametric joint distribution modeling via multivariate normal copula and probability integral transformations (Sklar, 1959; Nelsen, 2006).
- **Our Implementation**: Deterministic implementation utilizing SDV `GaussianCopulaSynthesizer` (v1.38.3) with `random_state = 42`, fitting `final_training_data/nhanes_generative_train.csv` ($N = 4,826$).
- **Our Experimental Results**: Empirical quantitative metrics (KS statistic, Wasserstein distance, JSD, downstream RF utility, and nearest-neighbor privacy risk) computed strictly on the held-out evaluation dataset (`nhanes_real_holdout.csv`, $N = 1,207$).
