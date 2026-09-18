/**
 * Validation Routes
 * Defines endpoints for /api/validate/*
 */

const express = require('express');
const router = express.Router();
const validationController = require('../controllers/validationController');

// POST /api/validate
router.post('/', validationController.validateCohort);

// GET /api/validate/:id
router.get('/:id', validationController.getValidationById);

module.exports = router;
