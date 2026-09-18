/**
 * Validation Controller
 * Handles POST /api/validate and GET /api/validate/:id
 */

const validationService = require('../services/validationService');

/**
 * POST /api/validate
 */
async function validateCohort(req, res, next) {
  try {
    const { cohortId, cohort_id, sourceDataset, targetConditions, conditions } = req.body || {};
    const finalCohortId = cohortId || cohort_id;

    if (!finalCohortId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required field: cohortId is required to run validation.'
      });
    }

    const report = await validationService.validateCohort({
      cohortId: finalCohortId,
      sourceDataset: sourceDataset || 'source_benchmark_1000.csv',
      targetConditions: targetConditions || conditions || {}
    });

    return res.status(200).json(report);
  } catch (err) {
    console.error('[validationController Error]:', err.message);
    return res.status(500).json({
      success: false,
      error: `Validation failed: ${err.message}`
    });
  }
}

/**
 * GET /api/validate/:id
 */
async function getValidationById(req, res, next) {
  try {
    const { id } = req.params;
    const report = validationService.getCachedValidation(id);
    if (!report) {
      return res.status(404).json({
        success: false,
        error: `Validation report for cohort ${id} not found.`
      });
    }
    return res.status(200).json(report);
  } catch (err) {
    next(err);
  }
}

module.exports = {
  validateCohort,
  getValidationById
};
