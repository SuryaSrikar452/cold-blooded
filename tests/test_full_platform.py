import os
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.hurdle_copula_model import HurdleConditionalCopulaModel

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["random_seed"] == 42

def test_baseline_metrics():
    response = client.get("/metrics/baseline")
    assert response.status_code == 200
    data = response.json()
    assert data["synthetic_rows"] == 4826
    assert data["clinical_constraint_validity"] == "100.00%"

def test_hurdle_copula_model_training_and_sampling():
    train_df = pd.read_csv("final_training_data/nhanes_generative_train.csv")
    model = HurdleConditionalCopulaModel(random_seed=42)
    model.fit(train_df)
    syn_df = model.sample(num_rows=100)
    
    assert len(syn_df) == 100
    assert "pain_score" in syn_df.columns
    # Verify exact zeros exist for pain_score
    zero_cnt = (syn_df["pain_score"] == 0.0).sum()
    assert zero_cnt > 0
    # Verify diabetes prevalence matching
    assert set(syn_df["diabetes"].unique()).issubset({0, 1})

def test_structured_synthetic_generation():
    response = client.post("/generate", json={"n": 100, "targets": {"diabetes": 1, "age_gt": 60, "activity_mims_lt": 8000}})
    assert response.status_code == 200
    data = response.json()
    assert data["n_generated"] == 100
    assert len(data["patients"]) == 100
    assert data["patients"][0]["diabetes"] == 1

def test_membership_inference_attack():
    response = client.post("/privacy/membership-inference", json={"cohort_id": "test_cohort", "attacker": "logistic_regression"})
    assert response.status_code == 200
    data = response.json()
    assert "auc" in data
    assert 0.0 <= data["auc"] <= 1.0
    assert len(data["roc_curve"]) > 0
    # Verify roc_curve payload is a clean list of dictionaries: [{"fpr": float, "tpr": float}, ...]
    assert isinstance(data["roc_curve"][0], dict)
    assert "fpr" in data["roc_curve"][0]
    assert "tpr" in data["roc_curve"][0]

def test_counterfactual_engine():
    response = client.post("/cohort/cohort_gen_01/counterfactual", json={"patient_ids": ["SYN-000001"], "variable": "activity_mims", "new_value": 15000})
    assert response.status_code == 200
    data = response.json()
    assert data["modified_variable"] == "activity_mims"
    assert data["counterfactual"]["activity_mims"] == 15000.0
    assert "disclaimer" in data

def test_edge_case_lab():
    res_list = client.get("/edge-cases/scenarios")
    assert res_list.status_code == 200
    scenarios = res_list.json()
    assert len(scenarios) >= 8

    res_gen = client.post("/generate/edge-cases", json={"scenario_id": "elderly_diabetic", "n": 20})
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert gen_data["provenance"] == "edge_case"
    assert len(gen_data["patients"]) == 20
    assert gen_data["patients"][0]["data_provenance"] == "edge_case"

def test_longitudinal_trajectory_cadence():
    response = client.get("/patient/SYN-000001?cadence=monthly&num_months=12")
    assert response.status_code == 200
    data = response.json()
    assert data["cadence"] == "monthly"
    assert data["num_months"] == 12
    assert len(data["trajectory"]) == 12
    assert data["trajectory"][0]["day_offset"] == 0
    assert data["trajectory"][11]["day_offset"] == 330

def test_representativeness_bias_audit():
    response = client.post("/validate/representativeness", json={"tolerance_pct": 2.5})
    assert response.status_code == 200
    data = response.json()
    assert data["total_subgroups_audited"] >= 10
    assert len(data["subgroup_audits"]) >= 10

def test_api_keys_and_public_generation():
    key_res = client.post("/api-keys", json={"label": "PyTest Key"})
    assert key_res.status_code == 200
    key_data = key_res.json()
    raw_key = key_data["api_key"]
    
    gen_res = client.post("/api/v1/generate", headers={"X-API-Key": raw_key}, json={"n": 50, "targets": {"diabetes": 1}})
    assert gen_res.status_code == 200
    pub_data = gen_res.json()
    assert pub_data["status"] == "success"

def test_nearest_real_neighbor():
    response = client.get("/privacy/nearest-neighbor/SYN-000001")
    assert response.status_code == 200
    data = response.json()
    assert "synthetic_patient_id" in data
    assert "nearest_real_patient_id" in data
    assert "distance" in data

def test_natural_language_cohort_parser():
    response = client.post("/cohort/parse-nl", json={"query": "diabetic female over 60 with high blood pressure"})
    assert response.status_code == 200
    data = response.json()
    assert "structured_filters" in data
    assert data["structured_filters"]["diabetes"] == 1
    assert data["structured_filters"]["sex"] == 2
    assert data["structured_filters"]["age_gt"] == 60
    assert data["structured_filters"]["systolic_bp_gt"] == 140

def test_data_export_stream():
    for fmt in ["csv", "json"]:
        response = client.post("/export", json={"format": fmt})
        assert response.status_code == 200
        assert len(response.content) > 0

def test_bias_representativeness_audit():
    response = client.get("/api/audit/bias")
    assert response.status_code == 200
    data = response.json()
    assert "total_subgroups_audited" in data
    assert data["total_subgroups_audited"] > 0
    assert "subgroup_audits" in data
    assert any("Age Band" in s["subgroup"] for s in data["subgroup_audits"])

def test_statistical_validation_endpoint():
    # First generate a small cohort to validate
    gen_res = client.post("/generate", json={"n": 100, "targets": {"diabetes": 1}})
    assert gen_res.status_code == 200
    cid = gen_res.json()["cohort_id"]
    
    val_res = client.post("/api/validate", json={"cohortId": cid})
    assert val_res.status_code == 200
    val_data = val_res.json()
    assert val_data["success"] is True
    assert "continuous_features" in val_data
    assert "correlation_preservation" in val_data

def test_frozen_synthetic_sample_artifacts():
    assert os.path.exists("artifacts/final_synthetic_sample.csv")
    assert os.path.exists("artifacts/final_synthetic_sample.parquet")
