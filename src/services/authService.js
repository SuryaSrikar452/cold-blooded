/**
 * SYNTHIA Authentication Service
 * Wraps Supabase Auth SDK v2 for email OTP, session management,
 * route guarding, and authentication-aware navigation.
 *
 * Adheres strictly to SYNTHIA trust boundary:
 * - Zero passwords
 * - Zero custom OTP database or custom OTP generation
 * - Supabase Auth manages OTP generation and verification
 * - Brevo SMTP handles email delivery via Supabase backend
 * - Preserves existing cohort state (synthia_cohort_state)
 */

(function (global) {
  'use strict';

  const SUPABASE_URL = 'https://ptjvffbdrjqxpljocsev.supabase.co';
  const SUPABASE_PUBLISHABLE_KEY = 'sb_publishable_MBFsgx7alLUvEX9nzUh5Fg_vZANcX8G';

  let _client = null;

  /**
   * Initializes and returns the Supabase client singleton.
   */
  function getClient() {
    if (_client) return _client;

    const supabaseLib = global.supabase || (typeof require !== 'undefined' ? require('@supabase/supabase-js') : null);
    if (!supabaseLib || typeof supabaseLib.createClient !== 'function') {
      console.warn('[SYNTHIA Auth] @supabase/supabase-js is not loaded yet.');
      return null;
    }

    _client = supabaseLib.createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
        storage: global.localStorage
      }
    });

    return _client;
  }

  /**
   * Formats Supabase Auth errors into editorial, human-readable UI messages.
   */
  function formatAuthError(err) {
    if (!err) return null;
    const msg = (err.message || String(err)).toLowerCase();
    const status = err.status || 0;

    if (status === 429 || msg.includes('rate limit') || msg.includes('too many requests') || msg.includes('over_email_send_rate_limit')) {
      return {
        title: 'TOO MANY REQUESTS',
        message: 'Please wait a moment before requesting another verification code.'
      };
    }

    if (msg.includes('invalid format') || msg.includes('valid email') || msg.includes('email address is invalid')) {
      return {
        title: 'INVALID EMAIL',
        message: 'Please enter a valid email address.'
      };
    }

    if (msg.includes('expired') || msg.includes('token has expired') || msg.includes('otp expired')) {
      return {
        title: 'CODE EXPIRED',
        message: 'This code has expired. Please request a new verification code.'
      };
    }

    if (msg.includes('invalid') || msg.includes('token is invalid') || msg.includes('does not match') || msg.includes('bad token')) {
      return {
        title: 'INVALID CODE',
        message: "That code doesn't match. Please check your email and try again."
      };
    }

    if (msg.includes('network') || msg.includes('fetch') || msg.includes('failed to fetch')) {
      return {
        title: 'NETWORK ERROR',
        message: 'Unable to reach the authentication service. Check your internet connection.'
      };
    }

    return {
      title: 'AUTHENTICATION ERROR',
      message: err.message || "We couldn't complete authentication. Please try again."
    };
  }

  /**
   * Request email OTP for Login or Registration.
   */
  async function sendOTP(email, options = {}) {
    const client = getClient();
    if (!client) {
      return { success: false, error: new Error('Supabase client is not available.') };
    }

    const cleanEmail = (email || '').trim().toLowerCase();
    if (!cleanEmail || !cleanEmail.includes('@')) {
      return { success: false, error: new Error('Please enter a valid email address.') };
    }

    const payload = {
      email: cleanEmail,
      options: {
        shouldCreateUser: options.shouldCreateUser !== false,
        data: options.data || (options.fullName ? { full_name: options.fullName } : undefined)
      }
    };

    try {
      const { data, error } = await client.auth.signInWithOtp(payload);
      if (error) {
        console.error('[SYNTHIA Auth] signInWithOtp error:', error);
        return { success: false, error };
      }
      return { success: true, data };
    } catch (err) {
      console.error('[SYNTHIA Auth] signInWithOtp exception:', err);
      return { success: false, error: err };
    }
  }

  /**
   * Verify email OTP token.
   */
  async function verifyOTP(email, token) {
    const client = getClient();
    if (!client) {
      return { success: false, error: new Error('Supabase client is not available.') };
    }

    const cleanEmail = (email || '').trim().toLowerCase();
    const cleanToken = (token || '').trim();

    if (!cleanToken || cleanToken.length < 6) {
      return { success: false, error: new Error('Please enter the full 6-digit verification code.') };
    }

    try {
      const { data, error } = await client.auth.verifyOtp({
        email: cleanEmail,
        token: cleanToken,
        type: 'email'
      });

      if (error) {
        console.error('[SYNTHIA Auth] verifyOtp error:', error);
        return { success: false, error };
      }

      return {
        success: true,
        data
      };
    } catch (err) {
      console.error('[SYNTHIA Auth] verifyOtp exception:', err);
      return { success: false, error: err };
    }
  }

  /**
   * Get active Supabase session.
   */
  async function getSession() {
    try {
      const client = getClient();
      if (!client) return null;
      const { data, error } = await client.auth.getSession();
      if (error) {
        console.warn('[SYNTHIA Auth] getSession error:', error.message);
        return null;
      }
      return data && data.session ? data.session : null;
    } catch (e) {
      console.warn('[SYNTHIA Auth] getSession failed:', e.message);
      return null;
    }
  }

  /**
   * Listen for auth changes.
   */
  function onAuthStateChange(callback) {
    const client = getClient();
    if (!client) return null;
    return client.auth.onAuthStateChange(callback);
  }

  /**
   * Sign out and redirect to /login while preserving cohort state.
   */
  async function signOut(redirectPath = 'login.html') {
    try {
      const client = getClient();
      if (client) {
        await client.auth.signOut();
      }
    } catch (e) {
      console.warn('[SYNTHIA Auth] Sign out error:', e.message);
    }
    // Retain synthia_cohort_state in sessionStorage as required by architecture
    window.location.href = redirectPath;
  }

  /**
   * Protected Route Guard:
   * Redirects unauthenticated visitors to /login?redirect=...
   */
  async function requireAuth() {
    const session = await getSession();
    if (!session || !session.user) {
      const currentPath = window.location.pathname.split('/').pop() || 'create.html';
      const search = window.location.search;
      const target = encodeURIComponent(currentPath + search);
      window.location.href = `login.html?redirect=${target}`;
      return null;
    }
    return session;
  }

  /**
   * Public Auth Route Guard:
   * Redirects already-authenticated visitors from /login or /register directly to application workspace.
   */
  async function redirectIfAuthenticated() {
    const session = await getSession();
    if (session && session.user) {
      const urlParams = new URLSearchParams(window.location.search);
      let target = urlParams.get('redirect');
      if (target) {
        try {
          target = decodeURIComponent(target);
          if (target.startsWith('/') || target.endsWith('.html')) {
            window.location.href = target;
            return;
          }
        } catch (e) {}
      }
      window.location.href = 'create.html';
    }
  }

  /**
   * Renders minimal, design-system compliant account indicator and Sign Out CTA in navigation.
   */
  async function renderNavAuth() {
    const session = await getSession();
    const nav = document.querySelector('nav');
    if (!nav) return;

    let authNavEl = document.getElementById('synthiaNavAuth');
    if (!authNavEl) {
      authNavEl = document.createElement('div');
      authNavEl.id = 'synthiaNavAuth';
      authNavEl.style.display = 'inline-flex';
      authNavEl.style.alignItems = 'center';
      authNavEl.style.gap = '0.75rem';
      authNavEl.style.marginLeft = '1rem';
    }

    if (session && session.user) {
      const email = session.user.email || 'Workspace User';
      const name = (session.user.user_metadata && session.user.user_metadata.full_name) || email.split('@')[0];

      authNavEl.innerHTML = `
        <div style="display:inline-flex;align-items:center;gap:0.45rem;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:rgba(245,243,238,0.85);" title="${email}">
          <span style="width:6px;height:6px;border-radius:50%;background:#10B981;box-shadow:0 0 6px rgba(16,185,129,0.5);"></span>
          <span style="max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${name}</span>
        </div>
        <button id="synthiaSignOutBtn" class="btn" style="background:transparent;border:1px solid rgba(255,255,255,0.18);color:rgba(245,243,238,0.85);font-size:11px;padding:0.35rem 0.65rem;cursor:pointer;border-radius:3px;font-family:'IBM Plex Sans',sans-serif;transition:all 160ms var(--ease-out, ease);">
          Sign out
        </button>
      `;

      const signOutBtn = authNavEl.querySelector('#synthiaSignOutBtn');
      if (signOutBtn) {
        signOutBtn.addEventListener('click', (e) => {
          e.preventDefault();
          signOut('login.html');
        });
      }
    } else {
      authNavEl.innerHTML = `
        <a href="login.html" class="btn" style="background:transparent;border:1px solid rgba(255,255,255,0.22);color:rgba(245,243,238,0.9);font-size:12px;padding:0.4rem 0.85rem;border-radius:3px;text-decoration:none;font-family:'IBM Plex Sans',sans-serif;transition:all 160ms var(--ease-out, ease);">
          Sign in &rarr;
        </a>
      `;
    }

    const navInner = nav.querySelector('.nav-inner');
    if (navInner) {
      const children = navInner.children;
      if (children.length >= 2) {
        const rightContainer = children[children.length - 1];
        if (!rightContainer.contains(authNavEl)) {
          rightContainer.style.display = 'flex';
          rightContainer.style.alignItems = 'center';
          rightContainer.appendChild(authNavEl);
        }
      } else {
        navInner.appendChild(authNavEl);
      }
    }
  }

  // Pre-initialize client when Supabase is available
  const clientInstance = getClient();

  // Export to global window.synthiaAuth
  global.synthiaAuth = {
    get supabase() {
      return getClient();
    },
    sendOTP,
    verifyOTP,
    getSession,
    signOut,
    requireAuth,
    redirectIfAuthenticated,
    renderNavAuth,
    formatAuthError,
    onAuthStateChange
  };

  // Backward compatibility: export supabaseClient directly if expected
  if (clientInstance) {
    global.supabaseClient = clientInstance;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = global.synthiaAuth;
  }
})(typeof window !== 'undefined' ? window : globalThis);