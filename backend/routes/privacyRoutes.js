/**
 * Privacy Routes
 * Defines endpoints for /api/privacy/*
 */

const express = require('express');
const router = express.Router();
const privacyController = require('../controllers/privacyController');

// GET /api/privacy/nearest-neighbor/:patientId
router.get('/nearest-neighbor/:patientId', privacyController.getNearestNeighbor);

// POST /api/privacy/attack
router.post('/attack', privacyController.runAttack);

// Alias: POST /api/privacy/membership-inference
router.post('/membership-inference', privacyController.runAttack);

module.exports = router;
