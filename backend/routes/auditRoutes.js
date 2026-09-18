/**
 * Audit Routes
 * Defines endpoints for /api/audit/*
 */

const express = require('express');
const router = express.Router();
const privacyController = require('../controllers/privacyController');

// GET /api/audit/bias
router.get('/bias', privacyController.runBiasAudit);

module.exports = router;
