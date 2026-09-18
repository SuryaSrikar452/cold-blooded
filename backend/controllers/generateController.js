/**
 * Generate Controller
 * Handles POST /api/generate/conditional and GET /api/generate/:id
 */

const generationService = require('../services/generationService');

/**
 * POST /api/generate/conditional
 */
async function generateConditionalCohort(req, res, next) {
  try {
    const { targetSize, size, conditions, ageOver60, diabetes, lowActivity, model, sourceDataset } = req.body || {};

    const finalTargetSize = parseInt(targetSize || size || 10000, 10);
    if (isNaN(finalTargetSize) || finalTargetSize < 1000 || finalTargetSize > 50000) {
      return res.status(400).json({
        success: false,
        error: 'Invalid targetSize. Must be an integer between 1,000 and 50,000.'
      });
    }

    const finalConditions = conditions || {
      ageOver60: ageOver60 != null ? Number(ageOver60) : 40,
      diabetes: diabetes != null ? Number(diabetes) : 30,
      lowActivity: lowActivity != null ? Number(lowActivity) : 35
    };

    const result = await generationService.generateConditionalCohort({
      targetSize: finalTargetSize,
      conditions: finalConditions,
      model: model || 'Gaussian Copula',
      sourceDataset: sourceDataset || 'nhanes_generative_train.csv'
    });

    return res.status(200).json(result);
  } catch (error) {
    console.error('[generateController Error]:', error.message);
    return res.status(500).json({
      success: false,
      error: `Synthetic generation failed: ${error.message}`
    });
  }
}

/**
 * GET /api/generate/:id
 */
async function getCohortById(req, res, next) {
  try {
    const { id } = req.params;
    const cohort = generationService.getCohortById(id);
    if (!cohort) {
      return res.status(404).json({
        success: false,
        error: `Cohort with ID ${id} not found.`
      });
    }
    return res.status(200).json({
      success: true,
      cohort
    });
  } catch (error) {
    next(error);
  }
}

/**
 * GET /api/generate/download/:id
 */
async function downloadCohortCsv(req, res, next) {
  try {
    const { id } = req.params;
    const path = require('path');
    const fs = require('fs');

    const csvPath = path.join(__dirname, '../generated_cohorts', `${id}.csv`);
    if (!fs.existsSync(csvPath)) {
      return res.status(404).json({
        success: false,
        error: `Cohort CSV file for ${id} not found.`
      });
    }

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="${id}.csv"`);
    fs.createReadStream(csvPath).pipe(res);
  } catch (error) {
    next(error);
  }
}

module.exports = {
  generateConditionalCohort,
  getCohortById,
  downloadCohortCsv
};

