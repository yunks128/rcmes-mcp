# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Local file caching for S3 downloads to avoid re-downloading data
- Country selection with auto-bounds and auto-masking
- Country masking and ETCCDI batch calculation functionality
- React web UI with interactive climate data exploration
- Azure OpenAI chat integration with function calling
- SLIM best-practice compliance files (LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, etc.)

### Fixed

- Normalized 0-360 longitudes for proper global map rendering
- Stripped base64 images from chat tool results to prevent token overflow

## [0.1.0] - 2025-01-01

### Added

- Initial release of RCMES-MCP server
- MCP tools for data access, processing, analysis, indices, and visualization
- Access to NEX-GDDP-CMIP6 dataset (35 models, 9 variables, 5 scenarios)
- SessionManager for chaining tool operations via dataset IDs
- Azure OpenAI interactive chat client
- Docker support
- FastAPI REST API backend
