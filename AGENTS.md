# AGENTS.md

This file provides guidance to AI coding agents working with code in this repository.

## Overview

pgeocode is a Python library for high-performance offline geocoding using postal codes. It queries GPS coordinates, region names, and municipality names from the GeoNames database. Supports 80+ countries and calculates distances between postal codes using the Haversine formula.

## Build and Test Commands

```bash
# Run all tests
pytest -sv .

# Run a single test
pytest -sv test_pgeocode.py::test_nominatim_query_postal_code

# Run slow tests (tests all countries)
pytest --runslow

# Install in development mode
pip install -e .

# Install with fuzzy search support
pip install -e ".[fuzzy]"

# Run linting
pre-commit run --all-files

# Build documentation
cd doc && make html
```

## Architecture

**Single-module design**: All functionality is in `pgeocode.py` (~526 lines).

### Key Classes

- **Nominatim**: Main class for querying postal code data
  - `query_postal_code(codes)`: Returns `pd.Series` for single code, `pd.DataFrame` for multiple
  - `query_location(name)`: Search by place name (supports fuzzy matching with `thefuzz`)
  - `unique` parameter: When True (default), merges duplicate postcodes; when False, returns all entries

- **GeoDistance**: Extends Nominatim for distance calculations
  - `query_postal_code(x, y)`: Returns distance in kilometers between postal codes

- **haversine_distance(x, y)**: Utility function for great-circle distance

### Data Flow

1. Data downloaded from GeoNames (with fallback to GitHub Pages mirror)
2. Cached locally in `~/.cache/pgeocode/` as `{COUNTRY}.txt`
3. Unique index files stored as `{COUNTRY}-index.txt`
4. Cache location configurable via `PGEOCODE_DATA_DIR` environment variable

### Country-specific Normalization

GB, IE, and CA postal codes are normalized to prefix-only (first part before space) in `_normalize_postal_code()`.

## Testing Notes

- Tests use a `temp_dir` fixture (in `test_pgeocode.py`) that redirects `STORAGE_DIR` to a temporary directory
- Ireland (IE) tests are marked `xfail`
- Slow tests (marked `@pytest.mark.slow`) require `--runslow` flag
- Haversine tests validate against geopy's `great_circle` implementation

## Linting

Uses ruff for linting and formatting, configured in `pyproject.toml`. Pre-commit hooks include:
- ruff (lint + format)
- mypy (type checking)
- Standard hooks (trailing whitespace, end-of-file-fixer, check-yaml)

## Commit Guidelines

Do not add AI agent attribution (e.g., "Co-Authored-By") to commits.
