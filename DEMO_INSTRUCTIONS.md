# SH-405 Hackathon Demo Guide

Follow these 16 steps for a flawless live demonstration of the SH-405 Synthetic Patient Data Generation Platform.

---

## Step-by-Step Demo Flow

1. **STEP 1: System Overview & Holdout Protection**
   - Open `http://localhost:8000`.
   - Point out the top banner: `nhanes_real_holdout.csv` SHA-256 hash verified intact (`345a320b...`).

2. **STEP 2: Seed Dataset Profiling**
   - Click **Upload Dataset** or **Data Profiling**.
   - Show the 9 approved modeling features, pointing out that `adherence_pct` and `pain_score` are tagged as **BENCHMARK_AUGMENTED**.

3. **STEP 3: Natural Language Cohort Builder**
   - Navigate to **NL Cohort Builder**.
   - Enter: `"Generate 5,000 diabetic patients over 60 with low activity and poor adherence"`.
   - Click **Parse Cohort Request**.

4. **STEP 4 & 5: Parsed Requirements & Feasibility**
   - Observe structured targets JSON (`{"diabetes": 1, "age_gt": 60, "activity_mims_lt": 6000, "adherence_pct_lt": 40.0}`).
   - Show resolved inferred thresholds box.
   - Click **Confirm & Generate Synthetic Cohort**.

5. **STEP 6 & 7: Gaussian Copula Generation**
   - Navigate to **Synthetic Generator**.
   - Run generator for $N = 4,826$ rows with seed `42`.
   - Point out $99.98\%$ clinical rule validity ($4,825 / 4,826$ rows).

6. **STEP 8: Validation Dashboard**
   - Open **Validation Dashboard**.
   - Show Composite Quality Score (**$88.6\%$**), KS statistics (SBP KS = 0.0185), JSD values (Sex JSD = 0.000011), and correlation heatmap MACD (0.0841).

7. **STEP 9 & 10: Patient Detail & Copula "What-If" Counterfactual**
   - Open **Patient Detail & What-If**.
   - Select patient `SYN-000001`.
   - Adjust Physical Activity MIMS slider from `3,000` to `12,000`.
   - Click **Calculate Correlated Counterfactual** and observe copula-driven updates across SBP, DBP, and Medications.
   - Point out the disclaimer: *Model-generated counterfactual, NOT a clinical prediction.*

8. **STEP 11: 12-Month Longitudinal Year View**
   - View the 12-month patient trajectory line chart (Month 0 to Month 11) generated via AR(1) mean-reverting process with bounded noise.

9. **STEP 12: Privacy Assessment (MIA)**
   - Open **Privacy Assessment (MIA)**.
   - Select **Both (Logistic Regression & Gradient Boosting)** and click **Run Membership Inference Attack**.
   - Show the live ROC curve and Attacker AUC (**$0.4895$** / **$0.5079$**), confirming low leakage risk.

10. **STEP 13: Nearest Real Match Explainability**
    - Show the **Nearest Real Seed Match** box for patient `SYN-000001`, displaying closest seed record ID, distance ($0.3421$), and distance percentile.

11. **STEP 14: Edge Case Lab**
    - Open **Edge Case Lab**.
    - Select scenario **Severe Hypertension** or **Polypharmacy Cohort**.
    - Click **Generate Edge Patients** and verify `data_provenance = "edge_case"`.

12. **STEP 15: Representativeness & Bias Audit**
    - Click **Run Representativeness Audit** in Validation Dashboard.
    - Review subgroup gap analysis against reference benchmarks with neutral status labels (`within_tolerance`, `over_represented`, `under_represented`).

13. **STEP 16: Programmatic API Access**
    - Open **API Access**.
    - Click **Generate New API Key**.
    - Copy key and execute the cURL request from terminal.
