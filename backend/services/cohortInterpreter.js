/**
 * Cohort Interpreter Service
 * Translates clinical natural language cohort specifications into strict SYNTHIA cohort parameters.
 * Does NOT send raw patient records to Gemini (Privacy Boundary).
 */

const { generateStructuredContent } = require('./geminiService');

const SYSTEM_INSTRUCTION = `You are a clinical epidemiology cohort specification assistant for SYNTHIA (a synthetic healthcare data platform).
Your task is to interpret a researcher's natural language request into four target cohort parameters:
1. targetSize: total synthetic patient population size (positive integer between 1000 and 25000, or null if unspecified)
2. ageOver60: percentage of patients aged 60 and older (integer between 0 and 100, or null if unspecified)
3. diabetes: percentage of patients with diabetes diagnosis (integer between 0 and 100, or null if unspecified)
4. lowActivity: percentage of patients with low physical activity level (integer between 0 and 100, or null if unspecified)

Rules:
- Strictly map relative adjustments (e.g., "more elderly", "increase diabetes prevalence", "double the cohort") using the provided currentCohort as baseline.
- If a parameter is not mentioned or implied by the request, return null for that field.
- You must ONLY return a JSON object conforming to this exact schema:
{
  "requirements": {
    "targetSize": integer or null,
    "ageOver60": integer or null,
    "diabetes": integer or null,
    "lowActivity": integer or null
  },
  "summary": "Short 1-sentence plain clinical explanation of the cohort specifications"
}`;

/**
 * Validate and sanitize cohort requirements.
 * Rejects or clamps out-of-bounds values according to the strict SYNTHIA schema.
 */
function validateAndSanitizeRequirements(rawReq) {
  if (!rawReq || typeof rawReq !== 'object') {
    throw new Error('Invalid response structure: "requirements" object missing.');
  }

  const sanitized = {};

  function normalizePct(rawVal) {
    if (rawVal === null || rawVal === undefined) return null;
    const num = parseFloat(rawVal);
    if (isNaN(num)) return null;
    const pct = num > 0 && num <= 1 ? Math.round(num * 100) : Math.round(num);
    return Math.max(0, Math.min(100, pct));
  }

  // targetSize: integer 1000 - 25000
  if (rawReq.targetSize !== null && rawReq.targetSize !== undefined) {
    const val = parseInt(rawReq.targetSize, 10);
    if (isNaN(val) || val <= 0) {
      sanitized.targetSize = null;
    } else {
      sanitized.targetSize = Math.max(1000, Math.min(25000, val));
    }
  } else {
    sanitized.targetSize = null;
  }

  sanitized.ageOver60 = normalizePct(rawReq.ageOver60);
  sanitized.diabetes = normalizePct(rawReq.diabetes);
  sanitized.lowActivity = normalizePct(rawReq.lowActivity);

  return sanitized;
}

/**
 * Interpret a natural language cohort prompt.
 * @param {string} text User's natural language request
 * @param {object} [currentCohort] Current baseline cohort state
 * @param {string[]} [datasetSchema] Available dataset columns (schema only, NO patient rows)
 */
async function interpretCohortRequest(text, currentCohort = {}, datasetSchema = []) {
  if (!text || typeof text !== 'string' || !text.trim()) {
    throw new Error('Prompt text is required.');
  }

  const userContext = {
    userPrompt: text.trim(),
    supportedFields: ['targetSize', 'ageOver60', 'diabetes', 'lowActivity'],
    currentCohort: {
      targetSize: currentCohort.size || currentCohort.targetSize || 10000,
      ageOver60: currentCohort.ageOver60 ?? 40,
      diabetes: currentCohort.diabetes ?? 30,
      lowActivity: currentCohort.lowActivity ?? 35,
    },
    datasetSchema: Array.isArray(datasetSchema) ? datasetSchema.slice(0, 50) : [],
  };

  const promptContent = `User Request: "${userContext.userPrompt}"
Context: ${JSON.stringify(userContext, null, 2)}

Interpret the user's intent into the specified JSON format.`;

  const rawResult = await generateStructuredContent(SYSTEM_INSTRUCTION, promptContent);
  const validatedReqs = validateAndSanitizeRequirements(rawResult?.requirements);

  return {
    requirements: validatedReqs,
    summary: rawResult?.summary || 'Cohort parameters interpreted successfully.',
  };
}

module.exports = {
  interpretCohortRequest,
  validateAndSanitizeRequirements,
};
