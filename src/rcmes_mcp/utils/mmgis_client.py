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

# Defaults pulled from environment (overridable per-call)
_DEFAULT_MMGIS_URL = os.environ.get("MMGIS_URL", "http://localhost:2888")
_DEFAULT_MISSION = os.environ.get("MMGIS_MISSION", "climate")
_DEFAULT_TOKEN = os.environ.get("MMGIS_API_TOKEN", "")
_DEFAULT_TITILER_EXTERNAL = os.environ.get(
    "MMGIS_TITILER_EXTERNAL_URL", "http://localhost:8080"
)

REQUEST_TIMEOUT = 30.0


def _auth_headers(token: str) -> dict[str, str]:
    """Build authorization headers for MMGIS long-term token."""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


# ─── Mission config helpers ────────────────────────────────────────────────


def get_mission_config(
    url: str = _DEFAULT_MMGIS_URL,
    mission: str = _DEFAULT_MISSION,
    token: str = _DEFAULT_TOKEN,
) -> dict[str, Any]:
    """Fetch the current configuration object for a mission.

    Returns the parsed config dict on success, raises on HTTP or API error.
    """
    resp = httpx.get(
        f"{url.rstrip('/')}/api/configure/get",
        params={"mission": mission, "full": True},
        headers=_auth_headers(token),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"MMGIS get_mission_config failed: {data.get('message')}")
    return data.get("config", {})


def upsert_mission_config(
    config: dict[str, Any],
    url: str = _DEFAULT_MMGIS_URL,
    mission: str = _DEFAULT_MISSION,
    token: str = _DEFAULT_TOKEN,
) -> dict[str, Any]:
    """Push an updated mission configuration back to MMGIS.

    Returns the upsert response dict (includes 'version').
    """
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
    url: str = _DEFAULT_MMGIS_URL,
    token: str = _DEFAULT_TOKEN,
) -> dict[str, Any]:
    """Create a new MMGIS mission with a minimal base config."""
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
    url: str = _DEFAULT_MMGIS_URL,
    token: str = _DEFAULT_TOKEN,
) -> bool:
    """Return True if the named mission already exists in MMGIS."""
    try:
        resp = httpx.get(
            f"{url.rstrip('/')}/api/configure/get",
            params={"mission": mission},
            headers=_auth_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # MMGIS returns the config dict directly (no status wrapper) for public GETs
        return "mission" in data or data.get("status") == "success"
    except Exception:
        return False


# ─── Layer schema builders ─────────────────────────────────────────────────


def build_tile_layer(
    name: str,
    cog_url: str,
    description: str = "",
    colormap: str = "RdBu_r",
    opacity: float = 0.8,
    titiler_url: str = _DEFAULT_TITILER_EXTERNAL,
) -> dict[str, Any]:
    """Build an MMGIS tile layer definition that streams a COG via TiTiler.

    The resulting tile URL pattern uses TiTiler's /cog/tiles endpoint so that
    MMGIS can render the GeoTIFF as an XYZ raster tile layer.
    """
    tile_url = (
        f"{titiler_url.rstrip('/')}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?url={cog_url}&colormap_name={colormap}&rescale=auto"
    )
    return {
        "name": name,
        "type": "tile",
        "url": tile_url,
        "description": description,
        "opacity": opacity,
        "on": True,
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
