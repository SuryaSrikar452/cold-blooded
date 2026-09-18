/**
 * Error Handling Middleware
 * Ensures sanitized, structured JSON error responses without leaking internal stack traces.
 */

function errorHandler(err, req, res, next) {
  console.error('[API Error]:', err.message || err);

  const statusCode = err.statusCode || (res.statusCode >= 400 ? res.statusCode : 500);
  const userMessage = err.message || 'Unable to interpret cohort request.';

  res.status(statusCode).json({
    success: false,
    error: userMessage
  });
}

module.exports = errorHandler;
