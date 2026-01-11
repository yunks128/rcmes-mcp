"""Pytest configuration and shared fixtures."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "network: marks tests requiring network access")


@pytest.fixture(scope="session")
def check_network():
    """Check if network is available."""
    import socket

    try:
        socket.create_connection(("s3.amazonaws.com", 443), timeout=5)
        return True
    except OSError:
        return False
