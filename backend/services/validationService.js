/**
 * SYNTHIA Validation Service
 * Interfaces with the validationEngine.py script to execute authentic
 * statistical and empirical privacy validation between source benchmark and synthetic cohort.
 */

const { spawn } = require('child_process');
const path = require('path');

// In-memory cache of validation results
const validationCache = new Map();

/**
 * Run statistical and privacy validation.
 * @param {object} params
 * @param {string} params.cohortId Generated synthetic cohort ID
 * @param {string} [params.sourceDataset] Source dataset filename (defaults to benchmark)
 * @param {object} [params.targetConditions] User-requested cohort targets
 * @returns {Promise<object>} Validation report
 */
async function validateCohort(params = {}) {
  const cohortId = params.cohortId || params.cohort_id;
  if (!cohortId) {
    throw new Error('cohortId is required for validation.');
  }

  // Check cache first
  if (validationCache.has(cohortId)) {
    return validationCache.get(cohortId);
  }

  const scriptPath = path.join(__dirname, 'validationEngine.py');
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  return new Promise((resolve, reject) => {
    const child = spawn(pythonCmd, [scriptPath, JSON.stringify(params)]);

    let stdoutData = '';
    let stderrData = '';

    child.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    child.on('error', (err) => {
      reject(new Error(`Failed to spawn validation engine: ${err.message}`));
    });

    child.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Validation engine exited with code ${code}: ${stderrData || stdoutData}`));
      }

      try {
        const jsonMatch = stdoutData.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('No valid JSON output produced by validation engine.');
        }

        const result = JSON.parse(jsonMatch[0]);

        if (!result.success) {
          return reject(new Error(result.error || 'Validation failed.'));
        }

        // Cache result
        validationCache.set(cohortId, result);
        resolve(result);
      } catch (parseErr) {
        reject(new Error(`Failed to parse validation output: ${parseErr.message}\nRaw: ${stdoutData.slice(0, 300)}`));
      }
    });
  });
}

/**
 * Retrieve cached validation report by cohort ID.
 */
function getCachedValidation(cohortId) {
  return validationCache.get(cohortId) || null;
}

module.exports = {
  validateCohort,
  getCachedValidation
};
