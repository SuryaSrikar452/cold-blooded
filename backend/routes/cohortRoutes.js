/**
 * Cohort API Routes
 */

const express = require('express');
const router = express.Router();
const { interpretCohort } = require('../controllers/cohortController');

router.post('/interpret', interpretCohort);

module.exports = router;
