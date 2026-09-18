/**
 * Generation Routes
 * Defines endpoints for /api/generate/*
 */

const express = require('express');
const router = express.Router();
const generateController = require('../controllers/generateController');

// POST /api/generate/conditional
router.post('/conditional', generateController.generateConditionalCohort);

// GET /api/generate/:id
router.get('/:id', generateController.getCohortById);

module.exports = router;
