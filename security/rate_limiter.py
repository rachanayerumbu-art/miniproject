import threading
import time
from collections import deque


REQUEST_LIMIT = 10
WINDOW_SECONDS = 60.0

_request_windows = {}
_state_lock = threading.Lock()


def _normalize_source_id(source_id):
    if isinstance(source_id, str):
        normalized = source_id.strip()
        if not normalized:
            return None
        return normalized

    if isinstance(source_id, int) and not isinstance(source_id, bool):
        if source_id <= 0:
            return None
        return str(source_id)

    return None


def check_rate_limit(source_id, time_source=None):
    """
    Check and record one request for a source in the sliding 60-second window.

    Returns a dictionary containing:
    - allowed: True or False
    - remaining: requests remaining in the current window
    - retry_after: seconds until a valid source can retry (0.0 when allowed)
    - reason: a clear explanation
    """
    normalized_source_id = _normalize_source_id(source_id)
    if normalized_source_id is None:
        return {
            "allowed": False,
            "remaining": 0,
            "retry_after": 0.0,
            "reason": "Invalid source_id: expected a non-empty string or positive integer",
        }

    now = time.time() if time_source is None else time_source()
    if not isinstance(now, (int, float)) or isinstance(now, bool):
        return {
            "allowed": False,
            "remaining": 0,
            "retry_after": 0.0,
            "reason": "Invalid time source: expected a numeric timestamp",
        }

    now = float(now)
    if now < 0:
        return {
            "allowed": False,
            "remaining": 0,
            "retry_after": 0.0,
            "reason": "Invalid time source: timestamp cannot be negative",
        }

    cutoff = now - WINDOW_SECONDS

    with _state_lock:
        request_times = _request_windows.setdefault(
            normalized_source_id,
            deque(),
        )

        while request_times and request_times[0] <= cutoff:
            request_times.popleft()

        if len(request_times) >= REQUEST_LIMIT:
            retry_after = max(0.0, request_times[0] + WINDOW_SECONDS - now)
            return {
                "allowed": False,
                "remaining": 0,
                "retry_after": retry_after,
                "reason": (
                    f"Rate limit exceeded: maximum {REQUEST_LIMIT} requests "
                    f"per {WINDOW_SECONDS:g} seconds"
                ),
            }

        request_times.append(now)
        remaining = REQUEST_LIMIT - len(request_times)
        return {
            "allowed": True,
            "remaining": remaining,
            "retry_after": 0.0,
            "reason": "Request allowed within the current rate limit",
        }


def reset_rate_limiter():
    """Clear all tracked source windows. Intended for tests and process reset."""
    with _state_lock:
        _request_windows.clear()
