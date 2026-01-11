"""
RCMES-MCP Server

Main entry point for the Regional Climate Model Evaluation System MCP Server.
This server exposes climate analysis tools to AI agents via the Model Context Protocol.
"""

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP(
    name="rcmes-mcp",
    instructions="""
    You are connected to the Regional Climate Model Evaluation System (RCMES) MCP Server.

    This server provides access to NASA's NEX-GDDP-CMIP6 dataset - 38TB of global
    downscaled climate projections from CMIP6 models at 0.25° resolution.

    Available capabilities:
    - Load and subset climate data by region and time period
    - Calculate climate statistics and trends
    - Compute ETCCDI climate extreme indices (heatwaves, drought, etc.)
    - Evaluate climate models against observations
    - Generate visualizations (maps, time series, Taylor diagrams)

    When users ask about climate trends:
    1. Clarify the geographic region of interest
    2. Clarify the time period (historical or future projections)
    3. For future projections, clarify the emissions scenario (SSP)
    4. Use appropriate tools to load, analyze, and visualize data

    Available scenarios for future projections:
    - SSP1-2.6: Low emissions, sustainable development
    - SSP2-4.5: Middle of the road
    - SSP3-7.0: Regional rivalry, high emissions
    - SSP5-8.5: Fossil-fueled development, very high emissions
    """
)

# Import and register tools from submodules
from rcmes_mcp.tools import data_access, processing, analysis, indices, visualization

# Import and register resources
from rcmes_mcp.resources import datasets

# Import and register prompts
from rcmes_mcp.prompts import workflows


def main():
    """Run the RCMES MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
