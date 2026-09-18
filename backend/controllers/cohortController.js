/**
 * Cohort Controller
 * Manages HTTP endpoints for cohort operations.
 */

const { interpretCohortRequest } = require('../services/cohortInterpreter');

/**
 * POST /api/cohort/interpret
 * Body: { text: string, currentCohort?: object, datasetSchema?: string[] }
 */
async function interpretCohort(req, res, next) {
  try {
    const { text, currentCohort, datasetSchema } = req.body;

    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({
        success: false,
        error: 'A non-empty "text" field is required for cohort interpretation.'
      });
    }

    const interpretation = await interpretCohortRequest(text, currentCohort, datasetSchema);

    return res.status(200).json({
      success: true,
      requirements: interpretation.requirements,
      summary: interpretation.summary
    });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  interpretCohort,
};
