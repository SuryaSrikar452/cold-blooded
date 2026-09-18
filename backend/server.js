/**
 * SYNTHIA Platform - Express Backend Server
 * Serves API endpoints, manages Gemini communication, and enforces security trust boundary.
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const cohortRoutes = require('./routes/cohortRoutes');
const errorHandler = require('./middleware/errorHandler');

const app = express();
const PORT = process.env.PORT || 5000;
const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:8080';

// CORS configuration (allow local frontend dev servers)
const allowedOrigins = [
  FRONTEND_URL,
  'http://localhost:8080',
  'http://127.0.0.1:8080',
  'http://localhost:3000',
  'http://127.0.0.1:3000',
  'http://localhost:5173',
  'http://127.0.0.1:5173'
];

app.use(cors({
  origin: function (origin, callback) {
    // Allow requests with no origin (e.g. mobile apps, curl, direct browser navigation)
    if (!origin || allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    return callback(null, true); // Permissive for local paired development
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Body parser
app.use(express.json({ limit: '2mb' }));

// Health check endpoint
app.get('/api/health', (req, res) => {
  const apiKey = process.env.GEMINI_API_KEY;
  const isKeySet = Boolean(apiKey && apiKey !== 'YOUR_REAL_KEY_HERE' && apiKey !== 'your_gemini_api_key_here');

  res.status(200).json({
    status: 'ok',
    service: 'SYNTHIA Backend',
    geminiConfigured: isKeySet,
    timestamp: new Date().toISOString()
  });
});

// Cohort interpretation API
app.use('/api/cohort', cohortRoutes);

// Optional: Serve frontend static files if accessed directly through backend port
const frontendDir = path.join(__dirname, '../frontend');
app.use(express.static(frontendDir));

// Clean URL rewrites for direct navigation
app.get('/', (req, res) => {
  res.sendFile(path.join(frontendDir, 'index.html'));
});
app.get('/create', (req, res) => {
  res.sendFile(path.join(frontendDir, 'create.html'));
});

// Error handling middleware
app.use(errorHandler);

// Start server
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`[SYNTHIA Backend] Server running on http://localhost:${PORT}`);
    console.log(`[SYNTHIA Backend] Health check: http://localhost:${PORT}/api/health`);
    console.log(`[SYNTHIA Backend] Trust boundary active. GEMINI_API_KEY isolated to backend.`);
  });
}

module.exports = app;
