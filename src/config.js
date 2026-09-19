/**
 * Frontend Configuration
 * Single point of truth for API URLs and backend environment connectivity.
 */

(function(global) {
  'use strict';
  
  const getOrigin = () => {
    if (typeof window !== 'undefined' && window.location && window.location.origin) {
      if (window.location.origin !== 'null' && !window.location.origin.startsWith('file:')) {
        return window.location.origin;
      }
    }
    return 'http://localhost:5000';
  };

  const BACKEND_URL = (global.SYNTHIA_CONFIG && global.SYNTHIA_CONFIG.BACKEND_URL)
    || global.SYNTHIA_BACKEND_URL
    || getOrigin();

  global.SYNTHIA_CONFIG = {
    ...(global.SYNTHIA_CONFIG || {}),
    BACKEND_URL: BACKEND_URL,
    API_BASE_URL: BACKEND_URL
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BACKEND_URL };
  }
})(typeof window !== 'undefined' ? window : globalThis);
