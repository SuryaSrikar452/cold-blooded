/**
 * Cohort API Service
 * Handles API communication between the frontend and the SYNTHIA backend.
 * Zero API keys or secrets are exposed here.
 */

(function(global) {
  'use strict';

  const getBackendUrl = () => {
    if (global.SYNTHIA_CONFIG && global.SYNTHIA_CONFIG.BACKEND_URL) {
      return global.SYNTHIA_CONFIG.BACKEND_URL;
    }
    if (typeof window !== 'undefined' && window.location && window.location.origin && !window.location.origin.startsWith('file:') && window.location.origin !== 'null' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      return window.location.origin;
    }
    return 'http://localhost:5000';
  };

  /**
   * Interpret a natural language cohort prompt via the backend Gemini service.
   * @param {string} text User's natural language request
   * @param {object} currentCohort Current cohort state baseline { size, ageOver60, diabetes, lowActivity }
   * @param {string[]} datasetSchema Optional array of column names from the uploaded CSV
   * @returns {Promise<{ success: boolean, requirements: object, summary?: string, error?: string, source: 'backend' | 'heuristic_fallback' }>}
   */
  async function interpretCohort(text, currentCohort = {}, datasetSchema = []) {
    if (!text || !text.trim()) {
      return {
        success: false,
        error: 'Please enter a cohort description.'
      };
    }

    const backendUrl = getBackendUrl();
    const endpoint = `${backendUrl}/api/cohort/interpret`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text.trim(),
          currentCohort: {
            size: currentCohort.size || currentCohort.targetSize || 10000,
            ageOver60: currentCohort.ageOver60 ?? 40,
            diabetes: currentCohort.diabetes ?? 30,
            lowActivity: currentCohort.lowActivity ?? 35
          },
          datasetSchema: Array.isArray(datasetSchema) ? datasetSchema : []
        })
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const errMsg = data?.error || `Server responded with status ${response.status}`;
        console.warn('[cohortApi] Backend responded with error:', errMsg, '- applying local heuristic fallback.');
        return fallbackLocalInterpretation(text, currentCohort);
      }

      if (data && data.success && data.requirements) {
        return {
          success: true,
          requirements: data.requirements,
          summary: data.summary || 'Cohort requirements interpreted successfully.',
          source: 'backend'
        };
      }

      return fallbackLocalInterpretation(text, currentCohort);
    } catch (networkError) {
      console.warn('[cohortApi] Backend server offline at', endpoint, '- using resilient local heuristic parser.');
      return fallbackLocalInterpretation(text, currentCohort);
    }
  }

  /**
   * Local rule-based fallback parser used when backend is offline or unconfigured.
   * Ensures the UI remains functional under all testing conditions.
   */
  function fallbackLocalInterpretation(text, currentCohort) {
    const lower = text.toLowerCase();

    let size = currentCohort.size || currentCohort.targetSize || 10000;
    let age = currentCohort.ageOver60 ?? 40;
    let diabetes = currentCohort.diabetes ?? 30;
    let activity = currentCohort.lowActivity ?? 35;

    const sizeMatchK = lower.match(/(\d+)\s*k/);
    const sizeMatchNum = lower.match(/(\d[\d,]*)\s*(patients|cohort|records)?/);
    if (sizeMatchK) {
      size = Math.min(25000, Math.max(1000, parseInt(sizeMatchK[1], 10) * 1000));
    } else if (sizeMatchNum) {
      const raw = parseInt(sizeMatchNum[1].replace(/,/g, ''), 10);
      if (raw >= 1000 && raw <= 25000) {
        size = raw;
      }
    }

    const ageMatch = lower.match(/(\d+)\s*%\s*(over|elderly|age|\b60)/i) || lower.match(/(over\s*60|elderly)[^\d]*(\d+)\s*%/i);
    if (ageMatch) {
      const val = parseInt(ageMatch[1] || ageMatch[2], 10);
      if (!isNaN(val)) age = Math.min(100, Math.max(0, val));
    } else if (lower.includes('more patients over 60') || lower.includes('more elderly')) {
      age = Math.min(100, age + 15);
    }

    const diaMatch = lower.match(/(\d+)\s*%\s*(diabetic|diabetes)/i) || lower.match(/(diabetic|diabetes)[^\d]*(\d+)\s*%/i);
    if (diaMatch) {
      const val = parseInt(diaMatch[1] || diaMatch[2], 10);
      if (!isNaN(val)) diabetes = Math.min(100, Math.max(0, val));
    } else if (lower.includes('diabetic') || lower.includes('diabetes')) {
      diabetes = 35;
    }

    const actMatch = lower.match(/(\d+)\s*%\s*(low activity|activity)/i) || lower.match(/(activity)[^\d]*(\d+)\s*%/i);
    if (actMatch) {
      const val = parseInt(actMatch[1] || actMatch[2], 10);
      if (!isNaN(val)) activity = Math.min(100, Math.max(0, val));
    } else if (lower.includes('high activity')) {
      activity = 18;
    } else if (lower.includes('low activity')) {
      activity = 48;
    }

    return {
      success: true,
      requirements: {
        targetSize: size,
        ageOver60: age,
        diabetes: diabetes,
        lowActivity: activity
      },
      summary: `Interpreted: ${size.toLocaleString('en-US')} patients, ${age}% over 60, ${diabetes}% diabetes, ${activity}% low activity.`,
      source: 'heuristic_fallback'
    };
  }

  // Attach to global window
  global.SYNTHIA_COHORT_API = {
    interpretCohort,
    fallbackLocalInterpretation
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { interpretCohort, fallbackLocalInterpretation };
  }
})(typeof window !== 'undefined' ? window : globalThis);
