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
 * Login is handled by signin.html — this module does not
 * render any UI. All auth failures redirect to signin.html.
 */

const HerdAuth = (() => {

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
        localStorage.removeItem('herdsync_active_new_animal_capture_session');
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

        const resp = await fetch(`/auth/refresh`, {
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
        return data.access_token;
    }

    function _redirectToSignin() {
        window.location.href = '/signin';
    }

    async function logout() {
        const refreshToken = getRefreshToken();
        if (refreshToken) {
            try {
                await fetch(`/auth/logout`, {
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
                if (opts.headers instanceof Headers) {
                    opts.headers.set('Authorization', `Bearer ${token}`);
                } else {
                    opts.headers['Authorization'] = `Bearer ${token}`;
                }
                return await fetch(url, opts);
            } catch {
                _redirectToSignin();
                throw new Error('Authentication required');
            }
        }

        return resp;
    }

    async function createPiTicket(kind, view, serialId) {
        const body = { kind, view };
        if (serialId != null) body.serial_id = String(serialId);

        const resp = await authFetch('/api/auth/tickets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            const error = await resp.json().catch(() => ({}));
            throw new Error(
                error.error ||
                error.detail ||
                `Ticket request failed (${resp.status})`
            );
        }

        return resp.json();
    }

    function buildPiResourceKey(config) {
        const view = String(config.view || '').trim().toLowerCase();
        if (!view) return null;
        if (config.kind === 'stream') {
            return `stream:${view}`;
        }
        if (config.kind === 'debug' && config.serialId != null) {
            return `debug:${String(config.serialId)}:${view}`;
        }
        return null;
    }

    async function getPiImageUrl(config) {
        const ticket = await createPiTicket(config.kind, config.view, config.serialId);
        const url = new URL(ticket.resource, window.location.origin);
        url.searchParams.set('ticket', ticket.ticket);
        const resourceKey = ticket.resource_key || buildPiResourceKey(config);
        if (resourceKey) {
            url.searchParams.set('rk', resourceKey);
        }
        url.searchParams.set('kind', config.kind);
        url.searchParams.set('view', config.view);
        if (config.serialId != null) {
            url.searchParams.set('serial_id', String(config.serialId));
        }
        if (config.cacheBust !== false) {
            url.searchParams.set('t', Date.now().toString());
        }
        return {
            url: url.pathname + url.search,
            expiresIn: ticket.expires_in,
            resource: ticket.resource,
        };
    }

    async function setPiImageSource(img, config) {
        if (!img) return null;
        img._piTicketConfig = { ...config };
        const next = await getPiImageUrl(img._piTicketConfig);
        img._piTicketExpiresAt = Date.now() + ((next.expiresIn || 120) * 1000);
        img.src = next.url;
        return next.url;
    }

    async function refreshPiImageSource(img) {
        if (!img || !img._piTicketConfig) return null;
        return setPiImageSource(img, img._piTicketConfig);
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
        createPiTicket,
        getUser,
        getAccessToken,
        isAuthenticated: isAccessTokenValid,
        setPiImageSource,
        refreshPiImageSource,
    };
})();
