#!/usr/bin/env bash
# Development run script

# Load common functions
source "$(dirname "$0")/scripts/common.sh"

# Initialize common variables
init_common_vars
ENV_FILE="${ENV_FILE:-.env}"

# Development-specific configuration
ENABLE_RELOAD="${JARVIS_ENABLE_RELOAD:-true}"

echo -e "${GREEN}🚀 Jarvis LLM Proxy API - Development Mode${NC}"
echo -e "${BLUE}📁 Root directory: $ROOT${NC}"

# Setup and configuration
check_setup
load_env "$ENV_FILE"

# Create virtual environment (development strategy)
create_venv_dev
echo -e "${GREEN}🐍 Using Python: $PY${NC}"

# Install requirements
install_base_requirements
install_conditional_requirements

# Install llama-cpp-python only if needed
if [[ "$(needs_llama_cpp)" == "true" ]]; then
    echo -e "${BLUE}🔍 llama-cpp-python needed for current configuration${NC}"
    should_install=$(should_install_llama_cpp "$ACCELERATION")
    install_acceleration_requirements "$ACCELERATION" "$should_install"
else
    echo -e "${YELLOW}⏭️  Skipping llama-cpp-python installation (not needed for current backends)${NC}"
fi

# Run diagnostics
run_diagnostics

# Setup cleanup handlers
setup_signal_handlers

# Start development server
start_server "$ENABLE_RELOAD"