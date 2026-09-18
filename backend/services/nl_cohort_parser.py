import os
import re
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any

from backend.services.feasibility_engine import evaluate_feasibility

def parse_natural_language_cohort_query(query: str) -> Dict[str, Any]:
    """
    Parses natural language cohort descriptions into structured target filters
    and canonical cohort state.
    Uses Gemini LLM if GEMINI_API_KEY is configured in backend environment;
    otherwise uses high-precision deterministic regex rules.
    Runs empirical joint feasibility evaluation against NHANES training corpus.
    """
    query_lower = query.lower()
    targets: Dict[str, Any] = {}
    canonical_state: Dict[str, Any] = {
        "size": 10000,
        "ageOver60": 40,
        "diabetes": 30,
        "lowActivity": 35
    }

    # 1. Cohort Size (e.g. "10,000 patients", "5000 diabetic", "n = 2500")
    size_match = re.search(r'(?:generate|want|cohort of)?\s*(\d{1,3}(?:,\d{3})+|\d+)\s*(?:patients|records|subjects|individuals|people)?', query_lower)
    if size_match:
        try:
            raw_val = size_match.group(1).replace(',', '')
            val = int(raw_val)
            if val >= 50:  # Reasonable cohort size
                canonical_state["size"] = val
                targets["targetSize"] = val
        except ValueError:
            pass

    # 2. Diabetes
    if "non-diabetic" in query_lower or "without diabetes" in query_lower or "no diabetes" in query_lower:
        targets["diabetes"] = 0
        canonical_state["diabetes"] = 0
    elif "diabetic" in query_lower or "diabetes" in query_lower:
        # Check if percentage is specified, e.g. "30% diabetic" or "around 35% diabetes"
        diab_pct_match = re.search(r'(\d+)\s*%\s*(?:diabetic|diabetes)', query_lower)
        if diab_pct_match:
            pct = int(diab_pct_match.group(1))
            canonical_state["diabetes"] = pct
            targets["diabetes"] = pct
        else:
            targets["diabetes"] = 1
            canonical_state["diabetes"] = 40  # Elevated diabetic cohort target

    # 3. Sex
    if re.search(r'\b(female|females|woman|women)\b', query_lower):
        targets["sex"] = 2
    elif re.search(r'\b(male|males|man|men)\b', query_lower):
        targets["sex"] = 1

    # 4. Age
    age_pct_match = re.search(r'(\d+)\s*%\s*(?:over 60|elderly|seniors|aged 60)', query_lower)
    if age_pct_match:
        pct = int(age_pct_match.group(1))
        canonical_state["ageOver60"] = pct
        targets["age_gt"] = 60
        targets["ageOver60"] = pct
    else:
        age_gt_match = re.search(r'(?:over|above|older than|>)\s*(\d+)', query_lower)
        if age_gt_match:
            gt = int(age_gt_match.group(1))
            targets["age_gt"] = gt
            if gt >= 60:
                canonical_state["ageOver60"] = 60
        elif "elderly" in query_lower or "senior" in query_lower or "seniors" in query_lower:
            targets["age_gt"] = 65
            canonical_state["ageOver60"] = 65

    age_lt_match = re.search(r'(?:under|below|younger than|<)\s*(\d+)', query_lower)
    if age_lt_match:
        targets["age_lt"] = int(age_lt_match.group(1))

    # 5. Physical Activity
    act_pct_match = re.search(r'(\d+)\s*%\s*(?:low activity|sedentary)', query_lower)
    if act_pct_match:
        pct = int(act_pct_match.group(1))
        canonical_state["lowActivity"] = pct
        targets["activity_mims_lt"] = 8500
        targets["lowActivity"] = pct
    elif "low activity" in query_lower or "sedentary" in query_lower:
        targets["activity_mims_lt"] = 5000.0
        canonical_state["lowActivity"] = 55
    elif "high activity" in query_lower or "active" in query_lower:
        targets["activity_mims_gt"] = 10000.0
        canonical_state["lowActivity"] = 10

    # 6. Blood Pressure
    bp_gt_match = re.search(r'(?:systolic|bp|blood pressure)\s*(?:>|above|over)?\s*(\d{3})', query_lower)
    if bp_gt_match:
        targets["systolic_bp_gt"] = int(bp_gt_match.group(1))
    elif "hypertension" in query_lower or "high blood pressure" in query_lower:
        targets["systolic_bp_gt"] = 140

    # 7. Pain Score
    pain_match = re.search(r'pain\s*(?:score)?\s*(?:>|above|over|gt)?\s*(\d+)', query_lower)
    if pain_match:
        targets["pain_score_gt"] = float(pain_match.group(1))
    elif "severe pain" in query_lower:
        targets["pain_score_gt"] = 7.0
    elif "moderate pain" in query_lower:
        targets["pain_score_gt"] = 4.0

    # 8. Medications
    med_match = re.search(r'(?:at least|>|>=)\s*(\d+)\s*medication', query_lower)
    if med_match:
        targets["n_medications_ge"] = int(med_match.group(1))

    parser_backend = "deterministic_regex_engine"

    # Backend-only Gemini integration if GEMINI_API_KEY is present
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        # Check backend/.env
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        gemini_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if gemini_key and gemini_key not in ["YOUR_REAL_KEY_HERE", "your_gemini_api_key_here"]:
        candidate_models = ["gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
        for model_name in candidate_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                payload_data = {
                    "contents": [{
                        "parts": [{
                            "text": (
                                f"You are a clinical cohort criteria extractor. Extract parameters from this prompt: '{query}'.\n"
                                "Return JSON ONLY with any identified keys:\n"
                                "- targetSize (int between 1000 and 50000)\n"
                                "- ageOver60 (int 0-100 representing percentage of cohort >=60 years old)\n"
                                "- diabetes (int 0-100 representing percentage prevalence of diabetes)\n"
                                "- lowActivity (int 0-100 representing percentage with low physical activity)\n"
                                "- age_gt (int threshold)\n"
                                "- systolic_bp_gt (int threshold)\n"
                                "- activity_mims_lt (float)\n"
                                "- pain_score_gt (float)\n"
                                "- n_medications_ge (int)\n"
                            )
                        }]
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.1
                    }
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload_data).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    clean_json = re.search(r'\{[\s\S]*\}', text)
                    if clean_json:
                        raw_parsed = json.loads(clean_json.group(0))
                        parsed = {k: v for k, v in raw_parsed.items() if v is not None}
                        targets.update(parsed)
                        for k in ["size", "targetSize"]:
                            if k in parsed and parsed[k] is not None:
                                canonical_state["size"] = int(parsed[k])
                        if "ageOver60" in parsed and parsed["ageOver60"] is not None:
                            canonical_state["ageOver60"] = int(parsed["ageOver60"])
                        if "diabetes" in parsed and parsed["diabetes"] is not None:
                            v = float(parsed["diabetes"])
                            canonical_state["diabetes"] = int(v) if v > 1 else (100 if v == 1.0 else 0)
                            targets["diabetes"] = 1 if v > 0 else 0
                        if "lowActivity" in parsed and parsed["lowActivity"] is not None:
                            canonical_state["lowActivity"] = int(parsed["lowActivity"])
                        parser_backend = f"gemini_api_{model_name}"
                        break
            except Exception as e:
                print(f"[NL_COHORT_PARSER] Gemini model '{model_name}' failed: {e}")
                continue  # Try next candidate Gemini model
        print(f"[NL_COHORT_PARSER] All Gemini candidate models failed. Using deterministic regex fallback.")
    else:
        if not gemini_key:
            print("[NL_COHORT_PARSER] No GEMINI_API_KEY found in env or backend/.env — using deterministic regex.")
        else:
            print(f"[NL_COHORT_PARSER] GEMINI_API_KEY is a placeholder value — using deterministic regex.")

    # Run empirical feasibility check
    warning, prev_pct, explanation = evaluate_feasibility(targets)

    return {
        "query": query,
        "structured_filters": targets,
        "canonical_state": canonical_state,
        "confidence": 0.95 if targets else 0.50,
        "feasibility_warning": warning,
        "prevalence_pct": prev_pct,
        "explanation": explanation,
        "parser_backend": parser_backend,
        "disclaimer": "Cohort criteria extracted via backend parser with empirical joint distribution feasibility audit."
    }
