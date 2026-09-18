/**
 * SYNTHIA Privacy & Clinical Analytics Controller
 * Handles HTTP requests for:
 * - Nearest Real Reference Record (GET /api/privacy/nearest-neighbor/:patientId)
 * - Membership Inference Attack (POST /api/privacy/attack)
 * - Subgroup Bias Audit (GET /api/audit/bias)
 * - Copula Counterfactual (POST /api/copula/counterfactual)
 * - Longitudinal Trajectory (GET /patient/:patient_id/longitudinal)
 * - Edge Cases (POST /api/generate/edge-cases, GET /api/edge-cases/scenarios)
 */

const privacyService = require('../services/privacyService');

/**
 * GET /api/privacy/nearest-neighbor/:patientId
 */
async function getNearestNeighbor(req, res, next) {
  try {
    const { patientId } = req.params;
    const cohortId = req.query.cohort_id || req.query.cohortId;

    const data = await privacyService.getNearestNeighbor(patientId, cohortId);
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController nearest-neighbor Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Nearest neighbor calculation failed: ${err.message}`
    });
  }
}

/**
 * POST /api/privacy/attack
 */
async function runAttack(req, res, next) {
  try {
    const { cohort_id, cohortId, attacker } = req.body || {};
    const finalCohortId = cohort_id || cohortId || 'default_cohort';

    const data = await privacyService.runAttack(finalCohortId, attacker);
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController attack Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Privacy attack execution failed: ${err.message}`
    });
  }
}

/**
 * GET /api/audit/bias
 */
async function runBiasAudit(req, res, next) {
  try {
    const cohortId = req.query.cohort_id || req.query.cohortId;
    const tolerancePct = parseFloat(req.query.tolerance_pct || 2.5);

    const data = await privacyService.runBiasAudit(cohortId, tolerancePct);
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController bias audit Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Bias audit failed: ${err.message}`
    });
  }
}

/**
 * POST /api/copula/counterfactual
 */
async function getCounterfactual(req, res, next) {
  try {
    const { patient_ids, variable, new_value, cohort_id } = req.body || {};
    const patientId = Array.isArray(patient_ids) && patient_ids.length > 0 ? patient_ids[0] : 'SYN-000001';

    const data = await privacyService.getCounterfactual(patientId, variable, new_value, cohort_id);
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController counterfactual Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Counterfactual analysis failed: ${err.message}`
    });
  }
}

/**
 * GET /patient/:patient_id/longitudinal
 */
async function getLongitudinalTrajectory(req, res, next) {
  try {
    const patientId = req.params.patient_id || req.params.patientId || 'SYN-000001';
    const numMonths = parseInt(req.query.num_months || 12, 10);
    const cadence = req.query.cadence || 'monthly';
    const cohortId = req.query.cohort_id || req.query.cohortId;

    const data = await privacyService.getLongitudinalTrajectory(patientId, numMonths, cadence, cohortId);
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController longitudinal Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Longitudinal trajectory generation failed: ${err.message}`
    });
  }
}

/**
 * POST /api/generate/edge-cases
 */
async function generateEdgeCases(req, res, next) {
  try {
    const { scenario_id, n } = req.body || {};
    const data = await privacyService.generateEdgeCases(scenario_id, n);
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController edge-cases Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Edge case generation failed: ${err.message}`
    });
  }
}

/**
 * GET /api/edge-cases/scenarios
 */
async function getEdgeCaseScenarios(req, res, next) {
  try {
    const data = await privacyService.getEdgeCaseScenarios();
    return res.status(200).json(data);
  } catch (err) {
    console.error('[privacyController edge-case scenarios Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Edge case scenarios retrieval failed: ${err.message}`
    });
  }
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
