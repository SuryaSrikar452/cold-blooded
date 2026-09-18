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

    const optionsPayload = {
      shouldCreateUser: options.shouldCreateUser !== false
    };

    if (options.data) {
      optionsPayload.data = options.data;
    } else if (options.fullName) {
      optionsPayload.data = { full_name: options.fullName };
    }

    // Set emailRedirectTo so that if a confirmation link is clicked in email, it redirects back smoothly
    if (typeof window !== 'undefined' && window.location) {
      optionsPayload.emailRedirectTo = options.emailRedirectTo || window.location.href;
    }

    const payload = {
      email: cleanEmail,
      options: optionsPayload
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
      return { success: false, error: new Error('Please enter the complete verification code.') };
    }

    try {
      let { data, error } = await client.auth.verifyOtp({
        email: cleanEmail,
        token: cleanToken,
        type: 'email'
      });

      // If 'email' type fails, try 'signup' type (which Supabase uses for new accounts in some configurations)
      if (error) {
        const res2 = await client.auth.verifyOtp({
          email: cleanEmail,
          token: cleanToken,
          type: 'signup'
        });
        if (!res2.error) {
          data = res2.data;
          error = null;
        }
      }

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
  async function signOut(redirectPath = 'http://localhost:5000/login.html') {
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
          if (target.startsWith('/') || target.endsWith('.html') || target.startsWith('http')) {
            window.location.href = target;
            return;
          }
        } catch (e) {}
      }
      window.location.href = 'http://localhost:5000/index.html';
    }
  }

  /**
   * Renders minimal, design-system compliant account indicator and Sign Out CTA in navigation.
   * Placed harmoniously BEFORE the primary CTA button to preserve editorial balance.
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
      authNavEl.style.marginRight = '0.5rem';
    }

    if (session && session.user) {
      const email = session.user.email || 'Workspace User';
      const rawName = (session.user.user_metadata && session.user.user_metadata.full_name) || email.split('@')[0];
      const initial = (rawName || 'U').charAt(0).toUpperCase();
      const displayName = rawName.length > 14 ? rawName.slice(0, 12) + '…' : rawName;

      authNavEl.innerHTML = `
        <div class="synthia-user-pill" style="display:inline-flex;align-items:center;height:34px;background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.09);border-radius:999px;padding:0 5px 0 6px;gap:8px;backdrop-filter:blur(12px);box-shadow:0 2px 8px rgba(0,0,0,0.25);flex-shrink:0;transition:all 200ms ease;">
          <div style="display:inline-flex;align-items:center;gap:7px;" title="${email}">
            <div style="position:relative;width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,rgba(108,77,255,0.4),rgba(139,92,246,0.18));border:1px solid rgba(167,139,250,0.35);display:flex;align-items:center;justify-content:center;color:#EDECE6;font-family:'Newsreader',serif;font-size:11px;font-weight:600;flex-shrink:0;line-height:1;">
              ${initial}
              <span style="position:absolute;bottom:-1px;right:-1px;width:6px;height:6px;border-radius:50%;background:#10B981;border:1.5px solid #0B0B0D;box-shadow:0 0 6px #10B981;"></span>
            </div>
            <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:500;color:rgba(245,243,238,0.85);letter-spacing:0.01em;white-space:nowrap;">${displayName}</span>
          </div>
          <span style="width:1px;height:12px;background:rgba(255,255,255,0.12);display:inline-block;flex-shrink:0;"></span>
          <button id="synthiaSignOutBtn" type="button" title="Log out from SYNTHIA" style="width:24px;height:24px;border-radius:50%;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);color:rgba(245,243,238,0.5);display:inline-flex;align-items:center;justify-content:center;cursor:pointer;transition:all 180ms ease;flex-shrink:0;padding:0;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
          </button>
        </div>
      `;

      const signOutBtn = authNavEl.querySelector('#synthiaSignOutBtn');
      if (signOutBtn) {
        signOutBtn.addEventListener('mouseenter', () => {
          signOutBtn.style.color = '#F87171';
          signOutBtn.style.background = 'rgba(239, 68, 68, 0.16)';
          signOutBtn.style.borderColor = 'rgba(239, 68, 68, 0.35)';
          signOutBtn.style.boxShadow = '0 0 10px rgba(239, 68, 68, 0.3)';
          signOutBtn.style.transform = 'scale(1.08)';
        });
        signOutBtn.addEventListener('mouseleave', () => {
          signOutBtn.style.color = 'rgba(245,243,238,0.5)';
          signOutBtn.style.background = 'rgba(255,255,255,0.04)';
          signOutBtn.style.borderColor = 'rgba(255,255,255,0.08)';
          signOutBtn.style.boxShadow = 'none';
          signOutBtn.style.transform = 'scale(1)';
        });
        signOutBtn.addEventListener('click', async (e) => {
          e.preventDefault();
          signOutBtn.innerHTML = `<span style="font-size:9px;color:#FCA5A5;">…</span>`;
          signOutBtn.disabled = true;
          await signOut('http://localhost:5000/login.html');
        });
      }
    } else {
      authNavEl.innerHTML = `
        <a href="http://localhost:5000/login.html" class="btn" style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.14);color:rgba(245,243,238,0.85);font-size:12px;padding:0.3rem 0.8rem;border-radius:999px;text-decoration:none;font-family:'IBM Plex Sans',sans-serif;transition:all 160ms ease;">
          Sign In &rarr;
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
          // Insert BEFORE the CTA button so Generation -> remains on the far right
          rightContainer.insertBefore(authNavEl, rightContainer.firstChild);
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