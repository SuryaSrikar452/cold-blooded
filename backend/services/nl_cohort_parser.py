import os
import re
import json
import urllib.request
import urllib.error
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
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload_data = {
                "contents": [{
                    "parts": [{
                        "text": (
                            f"Extract cohort criteria from: '{query}'. "
                            "Return valid JSON ONLY with possible keys: targetSize (int), diabetes (0 or 1), "
                            "age_gt (int), age_lt (int), ageOver60 (int 0-100), lowActivity (int 0-100), "
                            "systolic_bp_gt (int), activity_mims_lt (float), pain_score_gt (float), n_medications_ge (int)."
                        )
                    }]
                }]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Extract JSON block
                clean_json = re.search(r'\{[\s\S]*\}', text)
                if clean_json:
                    parsed = json.loads(clean_json.group(0))
                    targets.update(parsed)
                    for k in ["size", "targetSize"]:
                        if k in parsed: canonical_state["size"] = int(parsed[k])
                    if "ageOver60" in parsed: canonical_state["ageOver60"] = int(parsed["ageOver60"])
                    if "diabetes" in parsed:
                        canonical_state["diabetes"] = int(parsed["diabetes"]) if parsed["diabetes"] > 1 else (100 if parsed["diabetes"] == 1 else 0)
                    if "lowActivity" in parsed: canonical_state["lowActivity"] = int(parsed["lowActivity"])
                    parser_backend = "gemini_flash_backend"
        except Exception:
            pass  # Seamless fallback to deterministic regex

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
