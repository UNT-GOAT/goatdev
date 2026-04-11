const KEYS = {
  accessToken: "herdsync_access_token",
  refreshToken: "herdsync_refresh_token",
  expiresAt: "herdsync_token_expires_at",
  user: "herdsync_user",
};

(function checkExisting() {
  const token = sessionStorage.getItem(KEYS.accessToken);
  const expiresAt = parseInt(sessionStorage.getItem(KEYS.expiresAt) || "0");
  if (token && Date.now() < expiresAt) {
    window.location.href = "/dashboard";
    return;
  }

  const refresh = localStorage.getItem(KEYS.refreshToken);
  if (refresh) {
    fetch("/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    })
      .then((r) => {
        if (r.ok) return r.json();
        throw new Error("expired");
      })
      .then((data) => {
        sessionStorage.setItem(KEYS.accessToken, data.access_token);
        sessionStorage.setItem(
          KEYS.expiresAt,
          (Date.now() + data.expires_in * 1000).toString(),
        );
        if (data.refresh_token) {
          localStorage.setItem(KEYS.refreshToken, data.refresh_token);
        }
        if (data.user) {
          localStorage.setItem(KEYS.user, JSON.stringify(data.user));
        }
        window.location.href = "/dashboard";
      })
      .catch(() => {
        localStorage.removeItem(KEYS.refreshToken);
      });
  }
})();

const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const btn = document.getElementById("signinBtn");
const errorEl = document.getElementById("errorMsg");

async function handleLogin() {
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  if (!username || !password) {
    errorEl.textContent = "Enter username and password";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Signing in...";
  errorEl.textContent = "";

  try {
    const resp = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "Login failed" }));
      throw new Error(err.detail || "Login failed");
    }

    const data = await resp.json();
    sessionStorage.setItem(KEYS.accessToken, data.access_token);
    sessionStorage.setItem(
      KEYS.expiresAt,
      (Date.now() + data.expires_in * 1000).toString(),
    );
    if (data.refresh_token) {
      localStorage.setItem(KEYS.refreshToken, data.refresh_token);
    }
    localStorage.setItem(KEYS.user, JSON.stringify(data.user));
    window.location.href = "/dashboard";
  } catch (err) {
    errorEl.textContent = err.message;
    btn.disabled = false;
    btn.textContent = "Sign In";
    passwordInput.value = "";
    passwordInput.focus();
  }
}

btn.addEventListener("click", handleLogin);
passwordInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleLogin();
});
usernameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") passwordInput.focus();
});
setTimeout(() => usernameInput.focus(), 50);
