# Cross-Platform Support Changes

This document summarizes the changes made to add cross-platform support with hardware acceleration to the Jarvis LLM Proxy API.

## Summary of Changes

The project has been updated to support multiple operating systems and hardware acceleration options:

- **Linux**: CPU, CUDA (NVIDIA), ROCm (AMD - experimental)
- **macOS**: CPU, Metal (Apple Silicon)
- **Windows**: CPU, CUDA (planned)

## Files Added

### 1. `setup.sh` - Interactive Platform Configuration
- Detects operating system and hardware
- Prompts user for acceleration preference
- Creates `.setup_config` with platform settings
- Provides guidance for next steps

### 2. Requirements Files Split
- `requirements-base.txt` - Common dependencies for all platforms
- `requirements-cuda.txt` - NVIDIA GPU specific (placeholder)
- `requirements-metal.txt` - Apple Silicon specific (includes mlx-lm)
- `requirements-rocm.txt` - AMD GPU specific (placeholder)
- `requirements-cpu.txt` - CPU-only specific (placeholder)

### 3. `env.template` - Comprehensive Environment Template
- Hardware acceleration settings for all platforms
- Performance tuning options
- Multiple configuration presets
- Detailed comments explaining each option

### 4. `SETUP.md` - Detailed Setup Guide
- Platform-specific installation instructions
- Hardware requirements and recommendations
- Performance tuning guidelines
- Troubleshooting section
- Model recommendations by platform

### 5. `CROSS_PLATFORM_CHANGES.md` - This summary document

## Files Modified

### 1. `run.sh` - Enhanced Development Script
- Added OS detection and configuration loading
- Platform-specific llama-cpp-python installation with CMAKE_ARGS
- Colored output for better user experience
- Automatic setup.sh execution if configuration missing

### 2. `run-prod.sh` - Enhanced Production Script
- Same improvements as run.sh but for production deployment
- Consistent cross-platform behavior

### 3. `requirements.txt` - Updated Main Requirements
- Now references requirements-base.txt
- Includes setup instructions
- Documents platform-specific requirements approach

### 4. `README.md` - Updated Documentation
- New quick start process referencing setup.sh
- Updated backend options with hardware acceleration details
- Reference to detailed SETUP.md guide

## Key Features

### Automatic Hardware Detection
- Detects NVIDIA GPUs via nvidia-smi
- Detects AMD GPUs via rocm-smi
- Detects Apple Silicon architecture
- Falls back to CPU-only if no GPU found

### Platform-Specific Installation
- CUDA: `CMAKE_ARGS="-DLLAMA_CUDA=on"`
- Metal: `CMAKE_ARGS="-DLLAMA_METAL=on"` + mlx-lm
- ROCm: `CMAKE_ARGS="-DLLAMA_HIPBLAS=on"`
- CPU: Standard installation without special flags

### Intelligent Defaults
- Auto-selects appropriate acceleration based on detected hardware
- Provides manual override options
- Graceful fallback to CPU if GPU acceleration fails

### User Experience Improvements
- Colored terminal output for better readability
- Clear progress indicators
- Helpful error messages and next steps
- Comprehensive documentation

## Usage Flow

1. **Initial Setup**: User runs `./setup.sh`
   - System detects OS and hardware
   - User selects acceleration preference
   - Configuration saved to `.setup_config`

2. **Development**: User runs `./run.sh`
   - Script loads configuration from `.setup_config`
   - Installs appropriate dependencies
   - Starts development server with hot reload

3. **Production**: User runs `./run-prod.sh`
   - Same configuration loading and dependency installation
   - Starts production server

## Future Enhancements

### Windows Support
The framework is ready for Windows support. To add:
1. Update `setup.sh` with Windows detection (`$OSTYPE` checks)
2. Add Windows-specific CUDA detection
3. Test and validate Windows installation process

### Additional Acceleration Backends
- Intel GPU support (Intel Extension for PyTorch)
- Vulkan support for cross-platform GPU acceleration
- OpenCL support for older GPUs

### Enhanced Auto-Detection
- GPU memory detection for optimal settings
- CPU core count detection for thread optimization
- Available RAM detection for model size recommendations

## Testing

The setup has been tested on:
- ✅ Ubuntu 20.04 with NVIDIA RTX 3080 Ti (CUDA detection working)
- ⏳ macOS with Apple Silicon (framework ready, needs testing)
- ⏳ Linux with AMD GPU (framework ready, needs testing)

## Backward Compatibility

All changes maintain backward compatibility:
- Existing `.env` files continue to work
- Manual installation still possible via `pip install -r requirements.txt`
- Original run scripts behavior preserved when `.setup_config` exists

The new setup process is opt-in and enhances the user experience without breaking existing workflows.
