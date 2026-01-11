"""
RCMES-MCP: Regional Climate Model Evaluation System as an MCP Server

This package provides an MCP (Model Context Protocol) server that exposes
climate analysis capabilities for AI agents. It enables conversational
interaction with large climate datasets like NEX-GDDP-CMIP6 without
requiring users to download data or write code.

Key Features:
- Access to NASA's NEX-GDDP-CMIP6 dataset (38TB of climate projections)
- Climate data loading and subsetting
- ETCCDI climate extreme indices calculation
- Model evaluation metrics
- Visualization generation
"""

__version__ = "0.1.0"
__author__ = "NASA JPL"

from rcmes_mcp.server import mcp

__all__ = ["mcp", "__version__"]
