/**
 * Frontend Configuration
 * Single point of truth for API URLs and backend environment connectivity.
 * Auto-detects Render cloud origin vs local dev environment with background warm-up.
 */

(function(global) {
  'use strict';
  
  const computeBackendUrl = () => {
    // 1. Explicit override if present
    if (global.SYNTHIA_CONFIG && global.SYNTHIA_CONFIG.BACKEND_URL) {
      return global.SYNTHIA_CONFIG.BACKEND_URL;
    }
    if (global.SYNTHIA_BACKEND_URL) {
      return global.SYNTHIA_BACKEND_URL;
    }

    if (typeof window !== 'undefined' && window.location) {
      const origin = window.location.origin;
      const hostname = window.location.hostname;
      const port = window.location.port;

      // Deployed to cloud (e.g. *.onrender.com or custom domain)
      if (origin && origin !== 'null' && !origin.startsWith('file:') && hostname !== 'localhost' && hostname !== '127.0.0.1') {
        return origin;
      }

      // Local development
      if (hostname === 'localhost' || hostname === '127.0.0.1') {
        if (port === '5000') {
          return origin;
        }
        return 'http://localhost:5000';
      }
    }

    return 'http://localhost:5000';
  };

  const BACKEND_URL = computeBackendUrl();

  global.SYNTHIA_CONFIG = {
    ...(global.SYNTHIA_CONFIG || {}),
    BACKEND_URL: BACKEND_URL,
    API_BASE_URL: BACKEND_URL,
    getBackendUrl: () => BACKEND_URL
  };

  // Pre-warm Render instance in background on page load (eliminates cold-start lag for subsequent clicks)
  if (typeof window !== 'undefined' && window.fetch && BACKEND_URL && !BACKEND_URL.startsWith('file:')) {
    try {
      // Fire-and-forget health check ping
      fetch(`${BACKEND_URL}/api/health`, { method: 'GET', cache: 'no-store' })
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data) {
            console.log('[SYNTHIA] Cloud engine online & ready.');
          }
        })
        .catch(() => {
          // Silent catch for local offline testing
        });
    } catch (e) {}
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BACKEND_URL };
  }
})(typeof window !== 'undefined' ? window : globalThis);
