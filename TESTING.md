# RCMES-MCP Testing

## Introduction

This document provides an overview of the testing architecture for RCMES-MCP.

## Testing Categories

- [x] **Static Code Analysis:** `ruff` for linting, `mypy` for type checking
- [x] **Unit Tests:** `pytest` for tool and utility testing
- [ ] **Integration Tests:** end-to-end tests with S3 data access
- [ ] **Security Tests:** dependency vulnerability scanning

### Static Code Analysis

- Location: `src/`
- Purpose: Enforce code style and catch type errors
- Running Tests:
  - Manually:
    ```bash
    ruff check src/
    mypy src/
    ```
  - Automatically: Runs on every pull request via GitHub Actions

### Unit Tests

- Location: `tests/`
- Purpose: Verify individual tools and utilities work correctly
- Running Tests:
  - Manually:
    ```bash
    pytest                        # Run all tests
    pytest tests/test_tools.py    # Run specific test file
    pytest -m "not slow"          # Skip slow tests
    pytest -m "not network"       # Skip network-dependent tests
    pytest --cov=rcmes_mcp       # With coverage report
    ```
  - Automatically: Runs on every pull request via GitHub Actions
- Contributing:
  - Framework: [pytest](https://docs.pytest.org/)
  - Tips:
    - Use `@pytest.mark.slow` for long-running tests
    - Use `@pytest.mark.network` for tests requiring S3/network access
    - Mock S3 access for unit tests that don't need real data
