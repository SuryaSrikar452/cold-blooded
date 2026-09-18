/**
 * Frontend Configuration
 * Single point of truth for API URLs and backend environment connectivity.
 */

(function(global) {
  'use strict';
  
  const BACKEND_URL = (global.SYNTHIA_CONFIG && global.SYNTHIA_CONFIG.BACKEND_URL)
    || global.SYNTHIA_BACKEND_URL
    || 'http://localhost:5000';

  global.SYNTHIA_CONFIG = {
    ...(global.SYNTHIA_CONFIG || {}),
    BACKEND_URL: BACKEND_URL
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BACKEND_URL };
  }
})(typeof window !== 'undefined' ? window : globalThis);
