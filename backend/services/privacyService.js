/**
 * SYNTHIA Privacy & Clinical Analytics Service
 * Interfaces with privacyRunner.py to run authentic Nearest Neighbor,
 * MIA Attack, Bias Audit, Counterfactual, and Longitudinal analysis.
 */

const { spawn } = require('child_process');
const path = require('path');

// Cache to keep repeat nearest-neighbor and audit queries ultra-fast
const nnCache = new Map();
const auditCache = new Map();

/**
 * Execute a task via privacyRunner.py
 * @param {object} payload Task parameters
 * @returns {Promise<object>} Result
 */
function runPythonTask(payload) {
  const scriptPath = path.join(__dirname, 'privacyRunner.py');
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  return new Promise((resolve, reject) => {
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
      reject(new Error(`Failed to spawn privacy runner: ${err.message}`));
    });

    child.on('close', (code) => {
      if (code !== 0 && !stdoutData) {
        return reject(new Error(`Privacy runner exited with code ${code}: ${stderrData || stdoutData}`));
      }

      try {
        const jsonMatch = stdoutData.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('No valid JSON output produced by privacy runner: ' + (stdoutData || stderrData));
        }

        const result = JSON.parse(jsonMatch[0]);
        if (result.success === false && result.error) {
          return reject(new Error(result.error));
        }

        resolve(result);
      } catch (err) {
        reject(new Error(`Failed to parse output: ${err.message}`));
      }
    });
  });
}

/**
 * Get nearest real neighbor for a synthetic patient
 */
async function getNearestNeighbor(patientId, cohortId) {
  const cacheKey = `${patientId}_${cohortId || 'default'}`;
  if (nnCache.has(cacheKey)) {
    return nnCache.get(cacheKey);
  }

  const result = await runPythonTask({
    action: 'nearest_neighbor',
    patient_id: patientId || 'SYN-000001',
    cohort_id: cohortId
  });

  nnCache.set(cacheKey, result);
  return result;
}

/**
 * Run Membership Inference Attack
 */
async function runAttack(cohortId, attacker) {
  return await runPythonTask({
    action: 'attack',
    cohort_id: cohortId || 'default_cohort',
    attacker: attacker || 'logistic_regression'
  });
}

/**
 * Run Representativeness & Bias Audit
 */
async function runBiasAudit(cohortId, tolerancePct) {
  const cacheKey = `${cohortId || 'default'}_${tolerancePct || 2.5}`;
  if (auditCache.has(cacheKey)) {
    return auditCache.get(cacheKey);
  }

  const result = await runPythonTask({
    action: 'bias',
    cohort_id: cohortId,
    tolerance_pct: tolerancePct || 2.5
  });

  auditCache.set(cacheKey, result);
  return result;
}

/**
 * Compute Copula Counterfactual
 */
async function getCounterfactual(patientId, variable, newValue, cohortId) {
  return await runPythonTask({
    action: 'counterfactual',
    patient_id: patientId || 'SYN-000001',
    variable: variable || 'activity_mims',
    new_value: newValue || 10000,
    cohort_id: cohortId
  });
}

/**
 * Generate Longitudinal Trajectory
 */
async function getLongitudinalTrajectory(patientId, numMonths, cadence, cohortId) {
  return await runPythonTask({
    action: 'longitudinal',
    patient_id: patientId || 'SYN-000001',
    num_months: numMonths || 12,
    cadence: cadence || 'monthly',
    cohort_id: cohortId
  });
}

/**
 * Generate Edge Cases
 */
async function generateEdgeCases(scenarioId, n) {
  return await runPythonTask({
    action: 'edge_cases',
    scenario_id: scenarioId || 'elderly_diabetic',
    n: n || 50
  });
}

/**
 * List Edge Case Scenarios
 */
async function getEdgeCaseScenarios() {
  return await runPythonTask({
    action: 'edge_case_scenarios'
  });
}

module.exports = {
  getNearestNeighbor,
  runAttack,
  runBiasAudit,
  getCounterfactual,
  getLongitudinalTrajectory,
  generateEdgeCases,
  getEdgeCaseScenarios
};
