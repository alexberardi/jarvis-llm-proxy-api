# Script Consolidation Summary

## Overview
Consolidated duplicated logic between `run.sh` and `run-prod.sh` into a shared `scripts/common.sh` file to improve maintainability and reduce code duplication.

## Before Consolidation
- `run.sh`: 223 lines (complex setup, dependency management, diagnostics)
- `run-prod.sh`: 195 lines (similar logic with production-specific tweaks)
- **Total**: 418 lines with significant duplication

## After Consolidation
- `run.sh`: 39 lines (simple, focused on development-specific config)
- `run-prod.sh`: 41 lines (simple, focused on production-specific config)
- `scripts/common.sh`: 328 lines (shared functions and logic)
- **Total**: 408 lines (10 lines saved, but much better organization)

## Key Benefits

### 1. **Maintainability**
- Single source of truth for common logic
- Bug fixes and improvements only need to be made in one place
- Consistent behavior between development and production

### 2. **Clarity**
- `run.sh` and `run-prod.sh` now clearly show only the differences between dev/prod
- Common logic is well-organized in functions with clear names

### 3. **Extensibility**
- Easy to add new run modes (e.g., `run-test.sh`, `run-debug.sh`)
- Functions can be reused by other scripts

## Consolidated Functions

### Environment & Setup
- `init_common_vars()` - Initialize paths and variables
- `check_setup()` - Verify setup configuration exists
- `load_env()` - Load environment variables from .env files

### Virtual Environment Management
- `create_venv_dev()` - Development venv (pyenv + fallback)
- `create_venv_prod()` - Production venv (system python only)

### Dependency Management
- `install_base_requirements()` - Install base Python packages
- `install_conditional_requirements()` - Install based on env vars (vLLM, Transformers)
- `needs_llama_cpp()` - Check if llama-cpp-python is needed for current backend config
- `should_install_llama_cpp()` - Check if llama-cpp-python needs installation/update
- `install_llama_cpp()` - Install with proper acceleration flags
- `install_acceleration_requirements()` - Install platform-specific packages

### Diagnostics & Server
- `run_diagnostics()` - Test installations and configurations
- `cleanup()` - Graceful shutdown and GPU memory clearing
- `setup_signal_handlers()` - Set up cleanup on exit
- `start_server()` - Start uvicorn with appropriate configuration

## Key Differences Between Dev/Prod

| Aspect | Development (`run.sh`) | Production (`run-prod.sh`) |
|--------|----------------------|---------------------------|
| **Python Setup** | pyenv preferred, fallback to system | system python3 only |
| **Environment File** | `.env` (default) | `prod.env` (default) |
| **Uvicorn Mode** | `--reload` (configurable) | `--workers 4` (always) |
| **Error Handling** | More verbose, development-friendly | Production-focused |

## Usage

Both scripts now follow the same pattern:
1. Load common functions
2. Set script-specific configuration
3. Call common setup functions
4. Start server with appropriate mode

```bash
# Development
./run.sh

# Production  
./run-prod.sh
```

The scripts automatically detect and install the right dependencies based on your environment variables, just like before, but now with much cleaner and more maintainable code.
