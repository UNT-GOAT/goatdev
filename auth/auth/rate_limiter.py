"""
Login Rate Limiter

In-memory sliding window rate limiter for login attempts.
Tracks by both IP and username independently — an attacker can't
spray passwords across usernames from one IP, and can't hit one
username from a botnet, without triggering limits.

Not persistent across restarts (intentional — a restart clears
lockouts, which is fine for this scale and avoids DB complexity).

Stale entries are cleaned up lazily on check + periodically via
a background sweep to prevent slow memory growth from distributed
scanners that each only hit once.
"""

import time
import threading
from collections import defaultdict


class LoginRateLimiter:
    def __init__(
        self,
        max_attempts: int = 5,
        window_sec: int = 300,       # 5-minute sliding window
        lockout_sec: int = 900,      # 15-minute lockout after max_attempts
        cleanup_interval: int = 600, # Sweep stale entries every 10 min
    ):
        self.max_attempts = max_attempts
        self.window_sec = window_sec
        self.lockout_sec = lockout_sec

        self._attempts: dict[str, list[float]] = defaultdict(list)
        self._lockouts: dict[str, float] = {}  # key -> lockout_until timestamp
        self._lock = threading.Lock()

        # Background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            args=(cleanup_interval,),
            daemon=True,
        )
        self._cleanup_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, *keys: str) -> tuple[bool, int]:
        """
        Check if ANY of the given keys are rate-limited.

        Call with both IP and username:
            allowed, retry = limiter.check(ip, username)

        Returns:
            (True, 0)             — request is allowed
            (False, retry_after)  — blocked, retry_after in seconds
        """
        with self._lock:
            now = time.time()
            worst_retry = 0

            for key in keys:
                # Check active lockout
                if key in self._lockouts:
                    if now < self._lockouts[key]:
                        remaining = int(self._lockouts[key] - now) + 1
                        worst_retry = max(worst_retry, remaining)
                        continue
                    else:
                        del self._lockouts[key]

                # Check attempt count in window
                self._prune_old(key, now)
                if len(self._attempts[key]) >= self.max_attempts:
                    self._lockouts[key] = now + self.lockout_sec
                    worst_retry = max(worst_retry, self.lockout_sec)

            if worst_retry > 0:
                return False, worst_retry
            return True, 0

    def record_failure(self, *keys: str):
        """Record a failed login attempt for all given keys."""
        with self._lock:
            now = time.time()
            for key in keys:
                self._attempts[key].append(now)

    def record_success(self, *keys: str):
        """Clear attempt history on successful login."""
        with self._lock:
            for key in keys:
                self._attempts.pop(key, None)
                self._lockouts.pop(key, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_old(self, key: str, now: float):
        """Remove attempts outside the sliding window."""
        cutoff = now - self.window_sec
        attempts = self._attempts[key]
        self._attempts[key] = [t for t in attempts if t > cutoff]

    def _periodic_cleanup(self, interval: int):
        """Background sweep to remove stale entries from long-gone IPs."""
        while True:
            time.sleep(interval)
            with self._lock:
                now = time.time()
                cutoff = now - self.window_sec

                stale_keys = [
                    k for k, timestamps in self._attempts.items()
                    if not timestamps or timestamps[-1] < cutoff
                ]
                for k in stale_keys:
                    del self._attempts[k]

                expired_lockouts = [
                    k for k, until in self._lockouts.items()
                    if now >= until
                ]
                for k in expired_lockouts:
                    del self._lockouts[k]


# Singleton — shared across all login route invocations
login_limiter = LoginRateLimiter()