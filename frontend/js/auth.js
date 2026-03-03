/**
 * HerdSync Auth Module
 *
 * Drop-in authentication for all frontend pages. Handles:
 *   - Login flow (username/password → JWT)
 *   - Token storage (sessionStorage for access, localStorage for refresh)
 *   - Automatic token refresh before expiry
 *   - Authenticated fetch() wrapper
 *   - Login gate (redirects to login if not authenticated)
 *
 * Usage in any page:
 *
 *   <script src="../js/auth.js"></script>
 *   <script>
 *     // At page load — checks auth, shows login if needed
 *     HerdAuth.requireAuth().then(user => {
 *       console.log('Logged in as', user.username, user.role);
 *       // user.role is 'admin' or 'operator'
 *     });
 *
 *     // For API calls — automatically adds Bearer token
 *     const resp = await HerdAuth.fetch('/api/prod/grade', {
 *       method: 'POST',
 *       body: formData
 *     });
 *
 *     // Logout
 *     HerdAuth.logout();
 *   </script>
 *
 * The auth module uses the CloudFront domain for auth API calls:
 *   POST /auth/login
 *   POST /auth/refresh
 *   POST /auth/logout
 *   GET  /auth/me
 *
 * Pi API calls still go through CONFIG.API_BASE (Tailscale hostname).
 * Auth API calls go through CloudFront → EC2.
 */

const HerdAuth = (() => {
    // ==========================================================================
    // CONFIG
    // ==========================================================================

    // Auth API is on CloudFront (routed to EC2:8001)
    const AUTH_BASE = '';

    // How many seconds before expiry to trigger a refresh
    const REFRESH_BUFFER_SEC = 120; // 2 minutes before expiry

    // Storage keys
    const KEYS = {
        accessToken: 'herdsync_access_token',
        refreshToken: 'herdsync_refresh_token',
        expiresAt: 'herdsync_token_expires_at',
        user: 'herdsync_user',
    };

    let _refreshTimer = null;
    let _currentUser = null;

    // ==========================================================================
    // TOKEN STORAGE
    // ==========================================================================

    function saveTokens(accessToken, refreshToken, expiresIn, user) {
        const expiresAt = Date.now() + (expiresIn * 1000);

        // Access token in sessionStorage (cleared on tab close)
        sessionStorage.setItem(KEYS.accessToken, accessToken);
        sessionStorage.setItem(KEYS.expiresAt, expiresAt.toString());

        // Refresh token in localStorage (survives tab close, 30-day expiry)
        if (refreshToken) {
            localStorage.setItem(KEYS.refreshToken, refreshToken);
        }

        // User info in localStorage
        localStorage.setItem(KEYS.user, JSON.stringify(user));
        _currentUser = user;

        // Schedule auto-refresh
        _scheduleRefresh(expiresIn);
    }

    function getAccessToken() {
        return sessionStorage.getItem(KEYS.accessToken);
    }

    function getRefreshToken() {
        return localStorage.getItem(KEYS.refreshToken);
    }

    function getUser() {
        if (_currentUser) return _currentUser;
        try {
            const stored = localStorage.getItem(KEYS.user);
            _currentUser = stored ? JSON.parse(stored) : null;
            return _currentUser;
        } catch {
            return null;
        }
    }

    function clearTokens() {
        sessionStorage.removeItem(KEYS.accessToken);
        sessionStorage.removeItem(KEYS.expiresAt);
        localStorage.removeItem(KEYS.refreshToken);
        localStorage.removeItem(KEYS.user);
        _currentUser = null;
        if (_refreshTimer) {
            clearTimeout(_refreshTimer);
            _refreshTimer = null;
        }
    }

    function isAccessTokenValid() {
        const token = getAccessToken();
        const expiresAt = parseInt(sessionStorage.getItem(KEYS.expiresAt) || '0');
        return token && Date.now() < expiresAt;
    }

    // ==========================================================================
    // AUTO-REFRESH
    // ==========================================================================

    function _scheduleRefresh(expiresInSec) {
        if (_refreshTimer) {
            clearTimeout(_refreshTimer);
        }

        // Refresh 2 minutes before expiry
        const refreshIn = Math.max((expiresInSec - REFRESH_BUFFER_SEC) * 1000, 10000);

        _refreshTimer = setTimeout(async () => {
            try {
                await refreshAccessToken();
            } catch (err) {
                console.warn('[HerdAuth] Auto-refresh failed:', err.message);
                // Don't clear tokens — let the next API call trigger a manual refresh
            }
        }, refreshIn);
    }

    // ==========================================================================
    // AUTH API CALLS
    // ==========================================================================

    async function login(username, password) {
        const resp = await fetch(`${AUTH_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Login failed' }));
            throw new Error(err.detail || 'Login failed');
        }

        const data = await resp.json();
        saveTokens(data.access_token, data.refresh_token, data.expires_in, data.user);
        return data.user;
    }

    async function refreshAccessToken() {
        const refreshToken = getRefreshToken();
        if (!refreshToken) {
            throw new Error('No refresh token');
        }

        const resp = await fetch(`${AUTH_BASE}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!resp.ok) {
            // Refresh token expired or revoked — full re-login needed
            clearTokens();
            throw new Error('Session expired');
        }

        const data = await resp.json();
        // Keep existing refresh token, update access token
        saveTokens(data.access_token, null, data.expires_in, getUser());
        return data.access_token;
    }

    async function logout() {
        const refreshToken = getRefreshToken();
        if (refreshToken) {
            try {
                await fetch(`${AUTH_BASE}/auth/logout`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ refresh_token: refreshToken }),
                });
            } catch {
                // Best-effort logout — token will expire anyway
            }
        }
        clearTokens();
        // Show login UI
        _showLoginModal();
    }

    // ==========================================================================
    // AUTHENTICATED FETCH
    // ==========================================================================

    /**
     * Wrapper around fetch() that adds the Authorization header.
     * Automatically refreshes the access token if expired.
     *
     * Use this for ALL Pi API calls:
     *   const resp = await HerdAuth.fetch(CONFIG.API_BASE + '/api/prod/grade', opts);
     */
    async function authFetch(url, opts = {}) {
        let token = getAccessToken();

        // If access token expired, try refreshing
        if (!isAccessTokenValid()) {
            try {
                token = await refreshAccessToken();
            } catch {
                _showLoginModal();
                throw new Error('Authentication required');
            }
        }

        // Add auth header
        opts.headers = opts.headers || {};
        if (opts.headers instanceof Headers) {
            opts.headers.set('Authorization', `Bearer ${token}`);
        } else {
            opts.headers['Authorization'] = `Bearer ${token}`;
        }

        const resp = await fetch(url, opts);

        // If 401, try one refresh and retry
        if (resp.status === 401) {
            try {
                token = await refreshAccessToken();
                opts.headers['Authorization'] = `Bearer ${token}`;
                return await fetch(url, opts);
            } catch {
                _showLoginModal();
                throw new Error('Authentication required');
            }
        }

        return resp;
    }

    // ==========================================================================
    // LOGIN UI
    // ==========================================================================

    function _createLoginModal() {
        // Check if already exists
        if (document.getElementById('herdsync-login-overlay')) return;

        const overlay = document.createElement('div');
        overlay.id = 'herdsync-login-overlay';
        overlay.innerHTML = `
            <style>
                #herdsync-login-overlay {
                    position: fixed;
                    top: 0; left: 0; right: 0; bottom: 0;
                    background: rgba(0, 0, 0, 0.85);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 99999;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                }
                #herdsync-login-box {
                    background: #1a1a2e;
                    border: 1px solid #333;
                    border-radius: 12px;
                    padding: 40px;
                    width: 360px;
                    max-width: 90vw;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                }
                #herdsync-login-box h2 {
                    color: #e0e0e0;
                    margin: 0 0 8px 0;
                    font-size: 24px;
                    font-weight: 600;
                }
                #herdsync-login-box .subtitle {
                    color: #888;
                    margin: 0 0 28px 0;
                    font-size: 14px;
                }
                #herdsync-login-box input {
                    width: 100%;
                    padding: 12px 16px;
                    margin-bottom: 16px;
                    border: 1px solid #333;
                    border-radius: 8px;
                    background: #0d0d1a;
                    color: #e0e0e0;
                    font-size: 15px;
                    box-sizing: border-box;
                    outline: none;
                    transition: border-color 0.2s;
                }
                #herdsync-login-box input:focus {
                    border-color: #4a6cf7;
                }
                #herdsync-login-box button {
                    width: 100%;
                    padding: 12px;
                    border: none;
                    border-radius: 8px;
                    background: #4a6cf7;
                    color: white;
                    font-size: 15px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                #herdsync-login-box button:hover {
                    background: #3b5de7;
                }
                #herdsync-login-box button:disabled {
                    background: #333;
                    cursor: not-allowed;
                }
                #herdsync-login-error {
                    color: #ef4444;
                    font-size: 13px;
                    margin: 0 0 16px 0;
                    min-height: 20px;
                }
            </style>
            <div id="herdsync-login-box">
                <h2>HerdSync</h2>
                <p class="subtitle">Sign in to continue</p>
                <p id="herdsync-login-error"></p>
                <input type="text" id="herdsync-username" placeholder="Username" autocomplete="username" />
                <input type="password" id="herdsync-password" placeholder="Password" autocomplete="current-password" />
                <button id="herdsync-login-btn">Sign In</button>
            </div>
        `;

        document.body.appendChild(overlay);

        // Wire up events
        const btn = document.getElementById('herdsync-login-btn');
        const usernameInput = document.getElementById('herdsync-username');
        const passwordInput = document.getElementById('herdsync-password');
        const errorEl = document.getElementById('herdsync-login-error');

        async function handleLogin() {
            const username = usernameInput.value.trim();
            const password = passwordInput.value;

            if (!username || !password) {
                errorEl.textContent = 'Enter username and password';
                return;
            }

            btn.disabled = true;
            btn.textContent = 'Signing in...';
            errorEl.textContent = '';

            try {
                const user = await login(username, password);
                overlay.remove();
                // Dispatch event so the page knows auth succeeded
                window.dispatchEvent(new CustomEvent('herdsync-auth', { detail: user }));
            } catch (err) {
                errorEl.textContent = err.message;
                btn.disabled = false;
                btn.textContent = 'Sign In';
                passwordInput.value = '';
                passwordInput.focus();
            }
        }

        btn.addEventListener('click', handleLogin);
        passwordInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleLogin();
        });
        usernameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') passwordInput.focus();
        });

        // Focus username field
        setTimeout(() => usernameInput.focus(), 100);
    }

    function _showLoginModal() {
        _createLoginModal();
    }

    function _removeLoginModal() {
        const overlay = document.getElementById('herdsync-login-overlay');
        if (overlay) overlay.remove();
    }

    // ==========================================================================
    // PUBLIC API
    // ==========================================================================

    /**
     * Call this on page load. Returns a Promise that resolves with the
     * user object once authenticated.
     *
     * If the user has a valid session, resolves immediately.
     * If not, shows the login modal and resolves when login succeeds.
     */
    function requireAuth() {
        return new Promise(async (resolve) => {
            // Try existing access token
            if (isAccessTokenValid()) {
                resolve(getUser());
                return;
            }

            // Try refreshing with stored refresh token
            const refreshToken = getRefreshToken();
            if (refreshToken) {
                try {
                    await refreshAccessToken();
                    resolve(getUser());
                    return;
                } catch {
                    // Refresh failed — need full login
                }
            }

            // Show login modal
            _showLoginModal();

            // Wait for successful login
            window.addEventListener('herdsync-auth', (e) => {
                resolve(e.detail);
            }, { once: true });
        });
    }

    return {
        requireAuth,
        login,
        logout,
        fetch: authFetch,
        getUser,
        getAccessToken,
        isAuthenticated: isAccessTokenValid,
    };
})(); //