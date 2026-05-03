"""Offline-mode enforcement.

When ``DeepDiveConfig.offline`` is True, every outbound destination must be
loopback (localhost / 127.0.0.1 / ::1). Cloud LLM providers raise
``OfflineViolation`` at pipeline construction; non-loopback URLs are dropped
by the scraper at fetch time.

The wedge: combined with ``--allow-domains localhost``, this gives a hard,
testable guarantee — `unshare -n` would have nothing to attack — that the
research run made zero outbound calls.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

_LOOPBACK_NAMES = frozenset({"localhost"})


class OfflineViolation(RuntimeError):
    """Raised when offline mode would be broken by an attempted external call."""


def is_loopback(url: str) -> bool:
    """True iff the URL's host is loopback.

    Accepts: ``localhost``, anything in 127.0.0.0/8, and ``::1`` (in any IPv6
    form). Rejects suffix-spoofers like ``127.evil.com`` because the hostname
    is parsed as an IP first; a string like that fails ip parsing AND isn't
    the literal name ``localhost``.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def assert_loopback(url: str, *, what: str = "URL") -> None:
    """Raise OfflineViolation if ``url`` isn't loopback."""
    if not is_loopback(url):
        raise OfflineViolation(
            f"{what} {url!r} is not a loopback address; offline mode forbids it. "
            "Disable offline mode or change the destination."
        )
