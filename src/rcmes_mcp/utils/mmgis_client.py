"""
MMGIS REST API Client

Thin httpx wrapper for interacting with a running MMGIS instance.
Used by rcmes_mcp.tools.mmgis to push climate data layers.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("rcmes.utils.mmgis_client")

def _mmgis_url() -> str:
    return os.environ.get("MMGIS_URL", "http://localhost:2888")

def _mmgis_mission() -> str:
    return os.environ.get("MMGIS_MISSION", "climate")

def _mmgis_token() -> str:
    return os.environ.get("MMGIS_API_TOKEN", "")

def _titiler_url() -> str:
    return os.environ.get("MMGIS_TITILER_EXTERNAL_URL", "http://localhost:8080")

REQUEST_TIMEOUT = 30.0


def _auth_headers(token: str) -> dict[str, str]:
    """Build authorization headers for MMGIS long-term token."""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


# ─── Mission config helpers ────────────────────────────────────────────────


def get_mission_config(
    url: str | None = None,
    mission: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch the current configuration object for a mission.

    Returns the parsed config dict on success, raises on HTTP or API error.
    """
    url = url or _mmgis_url()
    mission = mission or _mmgis_mission()
    token = token if token is not None else _mmgis_token()
    resp = httpx.get(
        f"{url.rstrip('/')}/api/configure/get",
        params={"mission": mission, "full": True},
        headers=_auth_headers(token),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    # With ?full=true: {"status": "success", "config": {...}, "mission": ..., "version": ...}
    # Without full: returns the config dict directly (no wrapper)
    if "status" in data:
        if data.get("status") == "success":
            return data["config"]
        raise RuntimeError(f"MMGIS get_mission_config failed: {data.get('message')}")
    # Direct config dict (no wrapper)
    return data


def upsert_mission_config(
    config: dict[str, Any],
    url: str | None = None,
    mission: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Push an updated mission configuration back to MMGIS."""
    url = url or _mmgis_url()
    mission = mission or _mmgis_mission()
    token = token if token is not None else _mmgis_token()
    resp = httpx.post(
        f"{url.rstrip('/')}/api/configure/upsert",
        json={"mission": mission, "config": config, "forceClientUpdate": True},
        headers={**_auth_headers(token), "Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"MMGIS upsert_mission_config failed: {data.get('message')}")
    return data


def add_mission(
    mission: str,
    url: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Create a new MMGIS mission with a minimal base config."""
    url = url or _mmgis_url()
    token = token if token is not None else _mmgis_token()
    base_config = _base_mission_config(mission)
    resp = httpx.post(
        f"{url.rstrip('/')}/api/configure/add",
        json={"mission": mission, "config": base_config},
        headers={**_auth_headers(token), "Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    # "Mission already exists" is acceptable
    if data.get("status") != "success" and "already exists" not in data.get("message", ""):
        raise RuntimeError(f"MMGIS add_mission failed: {data.get('message')}")
    return data


def mission_exists(
    mission: str,
    url: str | None = None,
    token: str | None = None,
) -> bool:
    """Return True if the named mission already exists in MMGIS."""
    url = url or _mmgis_url()
    token = token if token is not None else _mmgis_token()
    try:
        resp = httpx.get(
            f"{url.rstrip('/')}/api/configure/get",
            params={"mission": mission},
            headers=_auth_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # MMGIS returns the config dict directly — presence of "msv" or "layers" means mission exists.
        return "msv" in data or "layers" in data or data.get("status") == "success"
    except Exception:
        return False


# ─── Layer schema builders ─────────────────────────────────────────────────


def build_tile_layer(
    name: str,
    cog_url: str,
    description: str = "",
    colormap: str = "rdbu_r",
    opacity: float = 0.8,
    titiler_url: str | None = None,
    rescale_min: float | None = None,
    rescale_max: float | None = None,
) -> dict[str, Any]:
    """Build an MMGIS tile layer definition that streams a COG via TiTiler.

    Uses the /titiler/ proxy path so tiles are served through port 8502.
    Auto-fetches data stats for rescale range if not provided.
    """
    rcmes_external = os.environ.get("RCMES_EXTERNAL_URL", "http://localhost:8502")
    titiler_internal = os.environ.get("TITILER_INTERNAL_URL", "http://127.0.0.1:8080")
    titiler_proxy = f"{rcmes_external.rstrip('/')}/titiler"

    # Fetch statistics to determine rescale range
    if rescale_min is None or rescale_max is None:
        try:
            stats_resp = httpx.get(
                f"{titiler_internal.rstrip('/')}/cog/statistics",
                params={"url": cog_url},
                timeout=REQUEST_TIMEOUT,
            )
            if stats_resp.status_code == 200:
                stats = stats_resp.json().get("b1", {})
                rescale_min = stats.get("percentile_2", stats.get("min", 0))
                rescale_max = stats.get("percentile_98", stats.get("max", 1))
        except Exception:
            rescale_min = rescale_min or 0
            rescale_max = rescale_max or 1

    tile_url = (
        f"{titiler_proxy}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?url={cog_url}&colormap_name={colormap}&rescale={rescale_min},{rescale_max}"
    )
    return {
        "name": name,
        "type": "tile",
        "url": tile_url,
        "description": description,
        "opacity": opacity,
        "initialOpacity": opacity,
        "on": True,
        "minZoom": 0,
        "maxNativeZoom": 18,
        "maxZoom": 20,
        "tileformat": "xyz",
        "legend": {"type": "colorbar", "colormap": colormap},
    }


def build_vector_layer(
    name: str,
    geojson_url: str,
    description: str = "",
    color: str = "#0080ff",
    opacity: float = 0.7,
) -> dict[str, Any]:
    """Build an MMGIS vector (GeoJSON) layer definition."""
    return {
        "name": name,
        "type": "vector",
        "url": geojson_url,
        "description": description,
        "opacity": opacity,
        "on": True,
        "style": {
            "fillColor": color,
            "color": color,
            "weight": 1,
            "fillOpacity": opacity,
        },
    }


def inject_layer(
    config: dict[str, Any],
    layer: dict[str, Any],
) -> dict[str, Any]:
    """Insert or update a layer in the mission config's layer tree.

    Finds an existing layer with the same name and replaces it; otherwise
    appends to the top-level 'layers' list. Returns the modified config.
    """
    import copy
    import uuid

    config = copy.deepcopy(config)
    layers: list[dict] = config.setdefault("layers", [])

    # Ensure the layer has a UUID (MMGIS requires one)
    if "uuid" not in layer:
        layer["uuid"] = str(uuid.uuid4())

    # Replace existing layer with same name, or append
    for i, existing in enumerate(layers):
        if existing.get("name") == layer["name"]:
            layer.setdefault("uuid", existing.get("uuid", layer["uuid"]))
            layers[i] = layer
            return config

    layers.append(layer)
    return config


# ─── Base mission config ───────────────────────────────────────────────────


def _base_mission_config(mission_name: str) -> dict[str, Any]:
    """Minimal MMGIS mission configuration for a global climate map."""
    return {
        "mission": mission_name,
        "layers": [],
        "map": {
            "projection": "EPSG:4326",
            "center": [0, 0],
            "zoom": 2,
            "minZoom": 1,
            "maxZoom": 18,
        },
        "globe": {"enabled": False},
        "metadata": {
            "description": "RCMES Climate Analysis — auto-generated by rcmes-mcp",
        },
    }
