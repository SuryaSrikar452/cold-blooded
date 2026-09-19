/**
 * SYNTHIA Generation API Service
 * Handles client-side communication with the /api/generate/conditional endpoint.
 * Zero secrets or credentials exposed here.
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
   * Request conditional cohort synthesis from the real backend model.
   * @param {object} params Cohort configuration
   * @param {number} params.targetSize Desired cohort size (e.g. 10,000)
   * @param {object} params.conditions Proportions: { ageOver60, diabetes, lowActivity }
   * @param {string} [params.model] Model architecture (e.g. 'Gaussian Copula')
   * @param {string} [params.sourceDataset] Source dataset reference
   * @returns {Promise<object>} Generated cohort metadata or structured error
   */
  async function generateConditionalCohort(params = {}) {
    const backendUrl = getBackendUrl();
    const endpoint = `${backendUrl}/api/generate/conditional`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    const payload = {
      targetSize: params.targetSize || params.size || 10000,
      model: params.model || 'Gaussian Copula',
      conditions: params.conditions || {
        ageOver60: params.ageOver60 ?? 40,
        diabetes: params.diabetes ?? 30,
        lowActivity: params.lowActivity ?? 35
      },
      sourceDataset: params.sourceDataset || params.fileName || 'nhanes_generative_train.csv'
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
        let errorMsg = 'Generation request failed.';
        if (data && data.error) {
          errorMsg = data.error;
        } else if (response.status === 400) {
          errorMsg = 'Bad request: invalid cohort configuration parameters.';
        } else if (response.status === 422) {
          errorMsg = 'Unprocessable entity: constraints could not be satisfied by model.';
        } else if (response.status === 500) {
          errorMsg = 'Server internal error during statistical copula sampling.';
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
        error: data?.error || 'Unknown response from synthesis server.'
      };
    } catch (networkErr) {
      clearTimeout(timeoutId);
      if (networkErr.name === 'AbortError') {
        return {
          success: false,
          error: 'Synthesis request timed out after 45 seconds.'
        };
      }
      return {
        success: false,
        error: `Could not connect to backend server at ${backendUrl}. Ensure the server is running on port 5000.`
      };
    }
  }

  // Global browser exposure
  global.SYNTHIA_GENERATE_API = {
    generateConditionalCohort
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { generateConditionalCohort };
  }
})(typeof window !== 'undefined' ? window : globalThis);
