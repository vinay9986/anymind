from __future__ import annotations

import ipaddress
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import structlog

_DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; Strands-CoreTools/0.1)"
logger = structlog.get_logger(__name__)


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _strip_wrapping_quotes(value: str) -> str:
    cleaned = (value or "").strip()
    for _ in range(2):
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
            continue
        break
    return cleaned


def _validate_public_https_url(url: str) -> str:
    url = _strip_wrapping_quotes(url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError("Only https URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("URL must not include credentials")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")
    lowered = hostname.lower().strip(".")
    if lowered in {"localhost"} or lowered.endswith(".local"):
        raise ValueError("URL hostname is not allowed")

    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        ip = None

    if ip is not None:
        if _is_forbidden_ip(ip):
            raise ValueError("URL hostname resolves to a forbidden IP")
        return parsed.geturl()

    try:
        infos = socket.getaddrinfo(
            hostname,
            parsed.port or 443,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except Exception as exc:
        raise ValueError(f"Failed to resolve hostname: {exc}") from exc

    resolved_ips: set[str] = set()
    for _, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        resolved_ips.add(ip_str)

    if not resolved_ips:
        raise ValueError("Hostname did not resolve to any IP addresses")

    for ip_str in resolved_ips:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_forbidden_ip(ip_obj):
            raise ValueError("URL hostname resolves to a forbidden IP")

    return parsed.geturl()


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        _validate_public_https_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


def _read_limited_bytes(response: Any, *, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            length = int(content_length)
        except (TypeError, ValueError):
            length = None
        if length is not None and length > max_bytes:
            raise ValueError("PDF exceeds max_bytes limit")

    buf = bytearray()
    chunk_size = 1024 * 1024
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise ValueError("PDF exceeds max_bytes limit")
    return bytes(buf)


def _download_html_from_url(
    *, url: str, timeout_seconds: float, max_bytes: int
) -> tuple[bytes, str, str]:
    """Download HTML content from a URL."""
    validated = _validate_public_https_url(url)
    start = time.perf_counter()
    logger.info(
        "core_http_download_start",
        url=validated,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    req = urllib.request.Request(
        validated,
        method="GET",
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None) or resp.getcode() or 0
            if int(status) >= 400:
                raise ValueError(f"HTTP {int(status)}")
            final_url = resp.geturl() or validated
            content_type = str(resp.headers.get("Content-Type", "") or "")
            html_bytes = _read_limited_bytes(resp, max_bytes=max_bytes)
    except urllib.error.HTTPError as exc:
        logger.info(
            "core_http_download_error",
            url=validated,
            error=f"HTTP {exc.code}",
        )
        raise ValueError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        logger.info(
            "core_http_download_error",
            url=validated,
            error=f"Failed to download: {exc.reason}",
        )
        raise ValueError(f"Failed to download: {exc.reason}") from exc
    except Exception as exc:
        logger.info(
            "core_http_download_error",
            url=validated,
            error=f"Download error: {exc}",
        )
        raise ValueError(f"Download error: {exc}") from exc

    if not html_bytes:
        raise ValueError("Downloaded content was empty")

    duration_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "core_http_download_complete",
        url=final_url,
        status=int(status),
        content_type=content_type,
        bytes=len(html_bytes),
        duration_ms=duration_ms,
    )

    return html_bytes, content_type, final_url
