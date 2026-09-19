/**
 * SYNTHIA Validation API Service
 * Communicates with POST /api/validate to perform real statistical
 * fidelity and empirical privacy verification.
 * Zero client secrets or fabricated metrics.
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
   * Request statistical and empirical privacy validation for a generated synthetic cohort.
   * @param {object} params
   * @param {string} params.cohortId Generated synthetic cohort ID
   * @param {string} [params.sourceDataset] Source benchmark file (defaults to benchmark)
   * @param {object} [params.targetConditions] User-requested cohort targets
   * @returns {Promise<object>} Validation report
   */
  async function validateCohort(params = {}) {
    const backendUrl = getBackendUrl();
    const endpoint = `${backendUrl}/api/validate`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    const cohortId = params.cohortId || params.cohort_id;
    if (!cohortId) {
      return {
        success: false,
        error: 'Missing required cohortId for validation.'
      };
    }

    const payload = {
      cohortId,
      sourceDataset: params.sourceDataset || 'source_benchmark_1000.csv',
      targetConditions: params.targetConditions || params.conditions || {}
    };

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        let errorMsg = 'Validation computation failed.';
        if (data && data.error) {
          errorMsg = data.error;
        } else if (response.status === 404) {
          errorMsg = `Cohort ${cohortId} was not found on the server. Please generate a cohort first.`;
        }
        return {
          success: false,
          status: response.status,
          error: errorMsg
        };
      }

      if (data && data.success) {
        return data;
      }

      return {
        success: false,
        error: data?.error || 'Unknown response from validation server.'
      };
    } catch (networkErr) {
      clearTimeout(timeoutId);
      if (networkErr.name === 'AbortError') {
        return {
          success: false,
          error: 'Validation request timed out after 35 seconds.'
        };
      }
      return {
        success: false,
        error: `Could not connect to validation server at ${backendUrl}. Ensure the backend is active on port 5000.`
      };
    }
  }

  // Global browser exposure
  global.SYNTHIA_VALIDATION_API = {
    validateCohort
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { validateCohort };
  }
})(typeof window !== 'undefined' ? window : globalThis);
