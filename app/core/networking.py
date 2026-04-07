import ipaddress
import socket
from collections import OrderedDict
from typing import Any

from fastapi import Request


def build_health_network_info(request: Request) -> dict[str, Any]:
    scheme = request.url.scheme or "http"
    port = request.url.port or (443 if scheme == "https" else 80)
    observed_host = request.url.hostname or "localhost"
    hostname = _safe_hostname(socket.gethostname(), fallback=observed_host)
    fqdn = _safe_hostname(socket.getfqdn(), fallback=hostname)

    candidate_hosts = [hostname]
    if hostname and not hostname.endswith(".local"):
        candidate_hosts.append(f"{hostname}.local")
    if fqdn and fqdn not in candidate_hosts:
        candidate_hosts.append(fqdn)

    advertised_urls: list[str] = []
    for host in [observed_host, *candidate_hosts, *_detect_lan_ipv4_addresses()]:
        if not host:
            continue
        advertised_urls.append(_build_base_url(scheme, host, port))

    advertised_urls = list(OrderedDict.fromkeys(advertised_urls))
    preferred_base_url = next(
        (url for url in advertised_urls if hostname and f"//{hostname}" in url),
        advertised_urls[0] if advertised_urls else _build_base_url(scheme, observed_host, port),
    )

    return {
        "hostname": hostname,
        "fqdn": fqdn,
        "advertised_urls": advertised_urls,
        "preferred_base_url": preferred_base_url,
        "observed_base_url": _build_base_url(scheme, observed_host, port),
    }


def _detect_lan_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        if not _is_private_or_routable_lan_ipv4(value):
            return
        if value in seen:
            return
        seen.add(value)
        addresses.append(value)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            add(probe.getsockname()[0])
    except OSError:
        pass

    for host in OrderedDict.fromkeys([socket.gethostname(), socket.getfqdn()]):
        if not host:
            continue
        try:
            for family, _, _, _, sockaddr in socket.getaddrinfo(host, None, socket.AF_INET):
                if family != socket.AF_INET:
                    continue
                add(sockaddr[0])
        except OSError:
            continue

    return addresses


def _is_private_or_routable_lan_ipv4(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.version != 4:
        return False
    return not (address.is_loopback or address.is_multicast or address.is_unspecified or address.is_link_local)


def _build_base_url(scheme: str, host: str, port: int) -> str:
    default_port = 443 if scheme == "https" else 80
    if port == default_port:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _safe_hostname(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip()
    return normalized or fallback
