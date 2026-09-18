/**
 * Generation Routes
 * Defines endpoints for /api/generate/*
 */

const express = require('express');
const router = express.Router();
const generateController = require('../controllers/generateController');

const privacyController = require('../controllers/privacyController');

// POST /api/generate/conditional
router.post('/conditional', generateController.generateConditionalCohort);

// GET /api/generate/download/:id
router.get('/download/:id', generateController.downloadCohortCsv);

// POST /api/generate/edge-cases
router.post('/edge-cases', privacyController.generateEdgeCases);

// GET /api/generate/edge-cases/scenarios
router.get('/edge-cases/scenarios', privacyController.getEdgeCaseScenarios);

// GET /api/generate/:id
router.get('/:id', generateController.getCohortById);

module.exports = router;

