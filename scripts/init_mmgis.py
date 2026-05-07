#!/usr/bin/env python3
"""
MMGIS Mission Initializer

Waits for MMGIS to become healthy, then ensures the configured mission exists.
Idempotent — safe to run on every container start.

Usage (standalone):
    python scripts/init_mmgis.py

Environment variables (read from .env or docker-compose):
    MMGIS_URL          — internal MMGIS URL (default: http://localhost:2888)
    MMGIS_MISSION      — mission name to create (default: climate)
    MMGIS_API_TOKEN    — long-term API token
"""

import os
import sys
import time

import httpx

MMGIS_URL = os.environ.get("MMGIS_URL", "http://localhost:2888")
MMGIS_MISSION = os.environ.get("MMGIS_MISSION", "climate")
MMGIS_API_TOKEN = os.environ.get("MMGIS_API_TOKEN", "")

MAX_WAIT_SECONDS = 120
POLL_INTERVAL = 5


def wait_for_mmgis() -> bool:
    """Poll MMGIS health endpoint until it responds or we time out."""
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{MMGIS_URL}/", timeout=5)
            if resp.status_code < 500:
                print(f"[init_mmgis] MMGIS is up at {MMGIS_URL}")
                return True
        except Exception:
            pass
        remaining = int(deadline - time.time())
        print(f"[init_mmgis] Waiting for MMGIS... ({remaining}s left)")
        time.sleep(POLL_INTERVAL)
    return False


def ensure_mission() -> None:
    """Create the climate mission if it does not yet exist."""
    # Lazy import so this script works even if utils/ isn't on PYTHONPATH
    # when run from repo root via `python scripts/init_mmgis.py`
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from rcmes_mcp.utils.mmgis_client import add_mission, mission_exists

    if mission_exists(MMGIS_MISSION, url=MMGIS_URL, token=MMGIS_API_TOKEN):
        print(f"[init_mmgis] Mission '{MMGIS_MISSION}' already exists — skipping.")
        return

    print(f"[init_mmgis] Creating mission '{MMGIS_MISSION}'...")
    result = add_mission(MMGIS_MISSION, url=MMGIS_URL, token=MMGIS_API_TOKEN)
    print(f"[init_mmgis] Mission created: {result}")


def main() -> None:
    print(f"[init_mmgis] Initializing MMGIS at {MMGIS_URL}, mission='{MMGIS_MISSION}'")

    if not wait_for_mmgis():
        print(
            "[init_mmgis] ERROR: MMGIS did not become ready within "
            f"{MAX_WAIT_SECONDS}s. Continuing anyway — push_layer_to_mmgis "
            "will retry when called."
        )
        return

    try:
        ensure_mission()
        print("[init_mmgis] Initialization complete.")
    except Exception as exc:
        print(f"[init_mmgis] WARNING: Mission setup failed: {exc}")
        print("[init_mmgis] Continuing — push_layer_to_mmgis will create the mission on demand.")


if __name__ == "__main__":
    main()
