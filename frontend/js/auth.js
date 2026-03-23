/**
 * HerdSync Auth Module
 *
 * Drop-in authentication for protected pages. Handles:
 *   - Token storage (sessionStorage for access, localStorage for refresh)
 *   - Automatic token refresh before expiry
 *   - Authenticated fetch() wrapper
 *   - Auth gate (redirects to /signin/ if not authenticated)
 *   - Logout (clears tokens, redirects to /signin/)
 *
 * Usage:
 *   <script src="/js/config.js"></script>
 *   <script src="/js/auth.js"></script>
 *   <script>
 *     HerdAuth.requireAuth().then(user => {
 *       console.log('Logged in as', user.username, user.role);
 *     });
 *
 *     const resp = await HerdAuth.fetch(CONFIG.DB_BASE + '/goats');
 *     HerdAuth.logout();
 *   </script>
 *
 * Login is handled by /signin/index.html — this module does not
 * render any UI. All auth failures redirect to /signin/.
 */

const HerdAuth = (() => {
    const AUTH_BASE =
        (typeof CONFIG !== "undefined" && CONFIG.AUTH_BASE)
            ? CONFIG.AUTH_BASE
            : "";

    const REFRESH_BUFFER_SEC = 120; // refresh 2 min before expiry

    const KEYS = {
        accessToken: 'herdsync_access_token',
        refreshToken: 'herdsync_refresh_token',
        expiresAt: 'herdsync_token_expires_at',
        user: 'herdsync_user',
    };

    let _refreshTimer = null;
    let _currentUser = null;

    // ======================================================================
    // TOKEN STORAGE
    // ======================================================================

    function saveTokens(accessToken, refreshToken, expiresIn, user) {
        const expiresAt = Date.now() + (expiresIn * 1000);
        sessionStorage.setItem(KEYS.accessToken, accessToken);
        sessionStorage.setItem(KEYS.expiresAt, expiresAt.toString());
        if (refreshToken) {
            localStorage.setItem(KEYS.refreshToken, refreshToken);
        }
        if (user) {
            localStorage.setItem(KEYS.user, JSON.stringify(user));
            _currentUser = user;
        }
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

    // ======================================================================
    // AUTO-REFRESH
    // ======================================================================

    function _scheduleRefresh(expiresInSec) {
        if (_refreshTimer) clearTimeout(_refreshTimer);
        const refreshIn = Math.max((expiresInSec - REFRESH_BUFFER_SEC) * 1000, 10000);
        _refreshTimer = setTimeout(async () => {
            try {
                await refreshAccessToken();
            } catch (err) {
                console.warn('[HerdAuth] Auto-refresh failed:', err.message);
            }
        }, refreshIn);
    }

    // ======================================================================
    // AUTH API
    // ======================================================================

    async function refreshAccessToken() {
        const refreshToken = getRefreshToken();
        if (!refreshToken) throw new Error('No refresh token');

        const resp = await fetch(`${AUTH_BASE}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!resp.ok) {
            clearTokens();
            throw new Error('Session expired');
        }

        const data = await resp.json();
        saveTokens(data.access_token, data.refresh_token, data.expires_in, getUser());

        // Refresh MJPEG streams with new token
        document.querySelectorAll('img[src*="token="]').forEach(img => {
            try {
                const url = new URL(img.src);
                url.searchParams.set('token', data.access_token);
                img.src = url.toString();
            } catch {}
        });

        return data.access_token;
    }

    function _redirectToSignin() {
        window.location.href = '/signin/';
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
                // Best-effort server-side revocation
            }
        }
        clearTokens();
        sessionStorage.clear();
        _redirectToSignin();
    }

    // ======================================================================
    // AUTHENTICATED FETCH
    // ======================================================================

    async function authFetch(url, opts = {}) {
        let token = getAccessToken();

        // If access token expired, try refreshing
        if (!isAccessTokenValid()) {
            try {
                token = await refreshAccessToken();
            } catch {
                _redirectToSignin();
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
                _redirectToSignin();
                throw new Error('Authentication required');
            }
        }

        return resp;
    }

    // ======================================================================
    // PUBLIC API
    // ======================================================================

    function requireAuth() {
        return new Promise(async (resolve) => {
            // Try existing access token
            if (isAccessTokenValid()) {
                resolve(getUser());
                return;
            }

            // Try refresh token
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

            // No valid session — redirect to signin page
            _redirectToSignin();
        });
    }

    return {
        requireAuth,
        logout,
        fetch: authFetch,
        getUser,
        getAccessToken,
        isAuthenticated: isAccessTokenValid,
    };
})();