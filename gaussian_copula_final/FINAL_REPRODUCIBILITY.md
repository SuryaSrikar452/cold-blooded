# FINAL_REPRODUCIBILITY.md: Independent Reproduction & Audit Guide

**Project:** SH-405 — Synthetic Patient Data Generation for Clinical Research  
**Target Model:** `GaussianCopulaFinal`  
**Date:** 2026-09-18  

---

## 1. Runtime Environment & Dependencies

All experiments were executed in a fully controlled Windows 64-bit environment:

| Software / Library | Version | Role in Pipeline |
|---|:---:|---|
| **Python** | `3.14.0` | Core runtime |
| **NumPy** | `2.4.1` | Matrix operations, Cholesky factorization, linear algebra |
| **SciPy** | `1.17.0` | Statistical distributions (Beta, Lognorm, Truncnorm, KS tests) |
| **Pandas** | `2.3.3` | Tabular data manipulation and serialization |
| **Scikit-learn** | `1.8.0` | Scaling, NearestNeighbors, LogisticRegression, StratifiedKFold |
| **PyTorch** | `2.13.0` | Neural baseline benchmarking (CTGAN reference) |
| **SDV** | `1.38.3` | Audited V1 baseline reference |
| **CTGAN** | `0.12.1` | Audited neural comparison baseline |

---

## 2. Cryptographic Checksums (SHA-256)

### Input Datasets
- `final_training_data/nhanes_generative_train.csv`: `690321d829d262dfa0d77a91d7d7159457dcc75541fc345f95098638c89a2866`
- `final_training_data/nhanes_real_holdout.csv`: `345a320bb2ab5b57093f54f1791736fadfa2c637646412308560f17015aadf71`

### Final Model Artifacts
- `gaussian_copula_final/models/model.pkl`: `80d1d6542d3c25c906544ec693bcb362e0130e5b96b3c76db7c8062fbdb76e43`
- `gaussian_copula_final/synthetic/gaussian_copula_final_synthetic.csv`: `e37779c2f6e7324b5feda31fb7c78d8b0f8026ad71653540eb0542d51733739c`
- `gaussian_copula_final/synthetic/gaussian_copula_final_synthetic.parquet`: `c5d6f6e4c17b857073de13858ad8cf1862ce13d313b1f0c748aed89d2552c446`

### Frozen Baseline Reference Artifacts
- `training_pipeline/models/gaussian_copula/model.pkl`: `5e5ac8f8a643f034003e80dc6e4d6ea9a800ca7c2d923e8dfa56aaadffa3ce30`
- `training_pipeline/synthetic/gaussian_copula_synthetic.csv`: `0089e65d9aa4816ebc74c4eb44284d8b15d6b49d12bc444c87fc4f58e06889ff`
- `training_pipeline/models/ctgan/model.pkl`: `5599c103681bdfcaef1cb1b1d1506f85aa2f7178a8d52d45970c2c158cd447c1`
- `training_pipeline/synthetic/ctgan_synthetic.csv`: `32f1d2eb2304a7a6e0c795e76ed5f9731d9367ecd1702337b1b4c53f92bbdd40`

---

## 3. Random Seeds & Determinism Policy

- **Global NumPy Seed:** `42`
- **Python Random Seed:** `42`
- **Rejection Sampling Seed:** `42`
- **Deterministic Sampling Test:** Unit test `test_08_reproducibility` in `test_copula_final.py` asserts bit-for-bit equivalence across independent sampling invocations:
  ```python
  sample_a = model.sample(n=100, random_state=123)
  sample_b = model.sample(n=100, random_state=123)
  pd.testing.assert_frame_equal(sample_a, sample_b)  # PASSED
  ```

---

## 4. Execution Runtimes

- **Fitting Runtime ($N=4,826$ training rows):** `0.006` seconds
- **Synthesis Runtime ($N=4,826$ synthetic rows):** `0.018` seconds
- **Total Pipeline Execution:** Under `0.05` seconds, enabling high-throughput scenario simulation and real-time conditional cohort sampling.

---

## 5. Step-by-Step Reproduction Commands

To independently reproduce the entire training, verification, and benchmarking workflow from a clean command prompt:

```bash
# Step 1: Navigate to repository root
cd s:\data_validation\nhanes_project

# Step 2: Execute automated unit test suite
python -m unittest gaussian_copula_final/test_copula_final.py -v

# Step 3: Train model and generate synthetic cohort
python gaussian_copula_final/train_final.py

# Step 4: Run scientific multi-model benchmark evaluation
python gaussian_copula_final/evaluate_final.py

# Step 5: Regenerate all evaluation figures
python gaussian_copula_final/generate_plots_final.py
```
