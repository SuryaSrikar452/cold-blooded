/**
 * SYNTHIA Generation Service
 * Interfaces with the local SDV Gaussian Copula Python synthesizer.
 * Enforces project-relative paths, real model execution, and cached cohort storage.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// In-memory cache of generated cohorts for the session
const cohortCache = new Map();

/**
 * Generate a conditional synthetic patient cohort using the real trained Gaussian Copula model.
 * @param {object} params Cohort configuration
 * @param {number} params.targetSize Desired cohort size (1,000 - 50,000)
 * @param {object} params.conditions Proportions: { ageOver60, diabetes, lowActivity }
 * @param {string} [params.model] Model type (defaults to 'Gaussian Copula')
 * @returns {Promise<object>} Generated cohort metadata and statistics
 */
async function generateConditionalCohort(params = {}) {
  const targetSize = Math.max(1000, Math.min(50000, parseInt(params.targetSize || params.size || 10000, 10)));
  const conditions = params.conditions || {
    ageOver60: params.ageOver60 ?? 40,
    diabetes: params.diabetes ?? 30,
    lowActivity: params.lowActivity ?? 35
  };

  const payload = {
    targetSize,
    conditions,
    model: params.model || 'Gaussian Copula'
  };

  const scriptPath = path.join(__dirname, 'syntheticGenerator.py');

  return new Promise((resolve, reject) => {
    // Spawn python process passing payload as JSON string
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const child = spawn(pythonCmd, [scriptPath, JSON.stringify(payload)]);

    let stdoutData = '';
    let stderrData = '';

    child.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    child.on('error', (err) => {
      reject(new Error(`Failed to execute Python synthetic generator: ${err.message}`));
    });

    child.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Generator exited with code ${code}: ${stderrData || stdoutData}`));
      }

      try {
        // Find JSON in stdout (in case any warnings preceded it)
        const jsonMatch = stdoutData.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('No valid JSON output produced by generator.');
        }

        const result = JSON.parse(jsonMatch[0]);

        if (!result.success) {
          return reject(new Error(result.error || 'Synthetic generation failed.'));
        }

        // Cache generated cohort for later retrieval by Validation page
        if (result.cohort_id) {
          cohortCache.set(result.cohort_id, result);
        }

        resolve(result);
      } catch (parseErr) {
        reject(new Error(`Failed to parse generation output: ${parseErr.message}\nRaw: ${stdoutData.slice(0, 300)}`));
      }
    });
  });
}

/**
 * Retrieve a generated cohort from cache by ID.
 * @param {string} cohortId
 * @returns {object|null}
 */
function getCohortById(cohortId) {
  return cohortCache.get(cohortId) || null;
}

module.exports = {
  generateConditionalCohort,
  getCohortById
};
