# SH-405 Synthetic Patient Data Generation Platform

An end-to-end, production-grade synthetic patient data generation, clinical validation, and privacy assessment platform built for the **SH-405 Clinical Research Project**.

---

## Key Features

1. **Gaussian Copula Baseline Generator**: Preserves original seed distribution relationships ($N = 4,826$, seed = 42) with $99.98\%$ clinical rule compliance.
2. **Membership Inference Attack (MIA) Engine**: Real black-box attack using Logistic Regression and Gradient Boosting ($AUC = 0.4895$, low leakage).
3. **Natural Language Cohort Builder**: Converts queries like *"Generate 5,000 diabetic patients over 60 with low activity"* into structured target schemas with explicit threshold resolution.
4. **Copula "What-If" Counterfactual Engine**: Computes mathematically correlated multivariate updates when patient parameters are modified.
5. **Edge Case & Adversarial Lab**: Generates rare tail-distribution cohorts tagged with `data_provenance = "edge_case"`.
6. **12-Month Longitudinal Year-View**: Mean-reverting AR(1) population trajectories with clinical boundary enforcement.
7. **Representativeness / Bias Audit**: Neutral subgroup gap analysis against reference benchmarks.
8. **Public API & Key Authentication**: Programmatic `X-API-Key` authenticated REST endpoints with rate metrics.
9. **Nearest Real Neighbor Explainability**: Standardized distance lookup matching synthetic patients to closest real seed records.
10. **Glassmorphic Interactive Dashboard**: 12 responsive views with live Chart.js charts.

---

## Quickstart & Launch Instructions

### 1. Requirements & Dependencies
Ensure Python 3.11+ is installed.
```bash
pip install sdv pandas numpy scipy scikit-learn pyarrow matplotlib seaborn fastapi uvicorn sqlalchemy pytest
```

### 2. Run the FastAPI Backend & Dashboard
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
Open your browser at `http://localhost:8000` to access the interactive web dashboard.

### 3. Run Automated Test Suite
```bash
python -m pytest tests/test_full_platform.py
```

---

## Example cURL Command (Public API)

```bash
# 1. Create API Key
curl -X POST "http://localhost:8000/api-keys" \
  -H "Content-Type: application/json" \
  -d '{"label": "Research Client"}'

# 2. Generate Synthetic Patients
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{"n": 500, "targets": {"diabetes": 1, "age_gt": 60}}'
```

---

## Repository Structure

```
final_training_data/
├── backend/                   # FastAPI Server, SQLAlchemy ORM & Services
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   └── services/
├── privacy/                   # MIA Privacy Engine
│   └── mia_engine.py
├── gaussian_copula_pipeline/  # Preserved Baseline Pipeline & Reports
├── frontend/                  # Interactive Dashboard SPA
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/                     # Automated Test Suite
├── IMPLEMENTATION_STATUS.md   # Feature status & verification log
├── DEMO_INSTRUCTIONS.md       # Step-by-step hackathon demo guide
└── README.md                  # Project overview
```
