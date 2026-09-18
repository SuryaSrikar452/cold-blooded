# SYNTHIA — Healthcare Synthetic-Data Platform

Full-stack architecture for SYNTHIA, a high-fidelity healthcare synthetic-data platform featuring mathematical multivariate covariance modelling, natural-language cohort interpretation with Google Gemini, live PCA cluster visualization, and longitudinal patient trajectory scrubbing.

---

## Architecture Overview

```
SYNTHIA/
│
├── frontend/                     # Client application (Vanilla HTML5/ES6/CSS3)
│   ├── src/
│   │   ├── config.js             # Centralized API and environment configuration
│   │   └── services/
│   │       └── cohortApi.js      # Frontend API client communicating with backend
│   ├── public/                   # Static assets
│   ├── index.html                # SYNTHIA Landing page (Preserved design & particle mesh)
│   ├── create.html               # Cohort setup & population canvas (Single cohortState)
│   └── package.json              # Frontend scripts & local dev server
│
├── backend/                      # Secure Express trust boundary
│   ├── controllers/
│   │   └── cohortController.js   # API controllers (request extraction & HTTP responses)
│   ├── middleware/
│   │   └── errorHandler.js       # Centralized error handler & status normalization
│   ├── routes/
│   │   └── cohortRoutes.js       # Express route definitions (/api/cohort/*)
│   ├── services/
│   │   ├── geminiService.js      # Isolated Gemini AI communication (Flash models)
│   │   └── cohortInterpreter.js  # Epidemiology system prompts & strict schema validation
│   ├── server.js                 # Express server entry point (Port 5000, CORS, Health check)
│   ├── package.json              # Backend dependencies (@google/genai, express, cors, dotenv)
│   ├── .env                      # Local secret environment file (GIT IGNORED)
│   └── .env.example              # Environment template with placeholder keys
│
├── .gitignore                    # Git rules strictly ignoring .env and node_modules
└── README.md                     # Architecture documentation and run guide
```

---

## Key Architectural Principles

### 1. Trust & Privacy Boundary
- **Zero Client Secrets**: `GEMINI_API_KEY` is loaded strictly on the backend from `process.env.GEMINI_API_KEY` via `dotenv`. The frontend contains no secrets, tokens, or direct Gemini API calls.
- **Patient Data Privacy**: The natural language cohort interpreter **never sends patient records or raw CSV rows to Gemini**. Only the user's natural language prompt, the current cohort parameter baseline, and the column schema (field names only) are transmitted.

### 2. Single Canonical `cohortState`
The frontend maintains exactly one source of truth for cohort parameters:
```javascript
const cohortState = {
  size: 10000,
  ageOver60: 40,
  diabetes: 30,
  lowActivity: 35,
  hasCsv: false,
  fileName: "clinical_sample.csv",
  schema: []
};
```
The manual precision sliders, the Gemini LLM assistant interpretation, the PCA cluster canvas, the longitudinal trajectory scrubber, and the synthetic generation simulator all consume and update this single canonical state.

### 3. Strict Epidemiology Schema Validation
The backend enforces bounded integers for supported cohort parameters:
- `targetSize`: Positive integer (1,000 – 25,000)
- `ageOver60`: Percentage integer (0 – 100)
- `diabetes`: Percentage integer (0 – 100)
- `lowActivity`: Percentage integer (0 – 100)

---

## Getting Started

### 1. Backend Setup

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Configure your Gemini API key in `backend/.env`:
   ```bash
   cp .env.example .env
   ```
   Edit `backend/.env`:
   ```env
   PORT=5000
   FRONTEND_URL=http://localhost:8080
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
4. Start the backend server:
   ```bash
   npm start
   # or
   node server.js
   ```
   The backend will start at `http://localhost:5000`.

### 2. Health Check
Verify the backend is running and Gemini is configured:
```bash
curl http://localhost:5000/api/health
```
Expected response:
```json
{
  "status": "ok",
  "service": "SYNTHIA Backend",
  "geminiConfigured": true,
  "timestamp": "2026-09-18T12:17:50.139Z"
}
```

### 3. Frontend Setup

In another terminal, start a static web server from the project root or `frontend/`:
```bash
# Option A: From project root using python
python -m http.server 8080

# Option B: Using Node
npx serve -p 8080 .
```

Open `http://localhost:8080` in your browser.

---

## API Reference

### `POST /api/cohort/interpret`
Interprets natural language cohort requests into structured epidemiology parameters.

### `POST /api/generate/conditional`
Generates authentic multivariate synthetic clinical patient cohorts using the trained SDV Gaussian Copula model.

**Request:**
```json
{
  "targetSize": 10000,
  "model": "Gaussian Copula",
  "conditions": {
    "ageOver60": 40,
    "diabetes": 30,
    "lowActivity": 35
  },
  "sourceDataset": "nhanes_generative_train.csv"
}
```

**Response:**
```json
{
  "success": true,
  "cohort_id": "SYN-GC-2E98C081",
  "model": "Gaussian Copula",
  "model_type": "GaussianCopulaSynthesizer",
  "sdv_version": "1.38.3",
  "source_records": 4826,
  "generated_count": 10000,
  "target_size": 10000,
  "num_features": 9,
  "features": [
    "age", "sex", "diabetes", "systolic_bp", "diastolic_bp",
    "activity_mims", "n_medications", "adherence_pct", "pain_score"
  ],
  "constraints_applied": {
    "targetSize": 10000,
    "ageOver60": 40,
    "diabetes": 30,
    "lowActivity": 35
  },
  "summary_metrics": {
    "age_over_60_pct": 28.2,
    "diabetes_pct": 30.0,
    "low_activity_pct": 32.9,
    "mean_systolic_bp": 118.0,
    "mean_diastolic_bp": 67.0,
    "generation_time_sec": 0.192
  }
}
```

---

## Design Preservation Guarantee
All visual elements, typography (`Newsreader`, `IBM Plex Sans`, `JetBrains Mono`), color palettes (`#0B0B0D`, `#F5F3EE`, `#6C4DFF`), animations, particle field physics, timeline scrubbers, and interactive canvas components remain 100% faithful to the approved SYNTHIA design system.
