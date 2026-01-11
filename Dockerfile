# RCMES-MCP Docker Image
# Regional Climate Model Evaluation System as MCP Server

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for NetCDF, HDF5, and GEOS (for cartopy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnetcdf-dev \
    libhdf5-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash rcmes && \
    chown -R rcmes:rcmes /app

USER rcmes

# Expose port for HTTP/SSE transport
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import rcmes_mcp; print('healthy')" || exit 1

# Default command - run the MCP server
CMD ["python", "-m", "rcmes_mcp.server"]
