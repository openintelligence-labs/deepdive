from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
import structlog
from selectolax.parser import HTMLParser

from deepdive import __version__
from deepdive.models import ScrapedPage

log = structlog.get_logger(__name__)

_USER_AGENT = f"DeepDive/{__version__} (+https://github.com/openintelligence-labs/deepdive)"
_MAX_BYTES = 2_000_000


def _resolves_to_private(host: str) -> bool:
    """True if ``host`` resolves to a non-routable / private address.

    Defense against SSRF via DNS or redirect: a public-looking hostname can
    resolve to 127.0.0.1, 169.254.169.254 (cloud metadata), 10.0.0.0/8, etc.
    Returns True for any address that is loopback, link-local, multicast,
    private, reserved, or unspecified.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # can't resolve → treat as unsafe
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_private
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def _is_safe_url(url: str) -> bool:
    """Reject non-http(s) schemes and hosts that resolve to private addresses."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    return not _resolves_to_private(host)


class Scraper:
    def __init__(
        self,
        timeout: float = 15.0,
        max_concurrency: int = 5,
        client: httpx.AsyncClient | None = None,
        *,
        allow_private_hosts: bool = False,
    ) -> None:
        """Web scraper.

        ``allow_private_hosts`` defaults False — redirects to private/loopback
        IPs are rejected to block SSRF. Tests that hit a local fixture server
        pass an external httpx client, which bypasses this scraper entirely.
        """
        self.timeout = timeout
        self.allow_private_hosts = allow_private_hosts
        self._sem = asyncio.Semaphore(max_concurrency)
        self._external = client is not None
        # follow_redirects=False so each hop is validated against the SSRF guard.
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        )

    async def aclose(self) -> None:
        if not self._external:
            await self._client.aclose()

    async def fetch(self, url: str) -> ScrapedPage | None:
        async with self._sem:
            if not self.allow_private_hosts and not _is_safe_url(url):
                log.debug("scrape_blocked_private_host", url=url)
                return None
            try:
                r = await self._follow(url)
                if r is None:
                    return None
                r.raise_for_status()
            except Exception as exc:
                log.debug("scrape_failed", url=url, error=str(exc))
                return None
            content = r.content[:_MAX_BYTES]
            try:
                title, text = self._extract(content)
            except Exception as exc:
                log.debug("parse_failed", url=url, error=str(exc))
                return None
            return ScrapedPage(url=str(r.url), title=title, text=text)

    async def _follow(self, url: str, *, max_hops: int = 5) -> httpx.Response | None:
        """Follow redirects manually, validating each hop's host."""
        current = url
        for _ in range(max_hops):
            r = await self._client.get(current)
            if r.is_redirect:
                next_url = r.headers.get("location")
                if not next_url:
                    return r
                next_url = str(httpx.URL(current).join(next_url))
                if not self.allow_private_hosts and not _is_safe_url(next_url):
                    log.debug("redirect_blocked_private_host", url=next_url)
                    return None
                current = next_url
                continue
            return r
        log.debug("redirect_loop", url=url)
        return None

    async def fetch_many(self, urls: list[str]) -> list[ScrapedPage]:
        results = await asyncio.gather(*(self.fetch(u) for u in urls), return_exceptions=True)
        return [r for r in results if isinstance(r, ScrapedPage)]

    @staticmethod
    def _extract(html: bytes) -> tuple[str, str]:
        tree = HTMLParser(html)
        title_node = tree.css_first("title")
        title = title_node.text(strip=True) if title_node else ""
        for selector in ("script", "style", "nav", "footer", "header", "aside"):
            for node in tree.css(selector):
                node.decompose()
        body = tree.body
        text = body.text(separator=" ", strip=True) if body else ""
        text = " ".join(text.split())
        return title, text
