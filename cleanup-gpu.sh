#!/usr/bin/env bash
# GPU Memory Cleanup Script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}🧹 Cleaning up GPU memory and processes...${NC}"

# Kill vLLM processes
echo -e "${BLUE}Killing vLLM processes...${NC}"
pkill -9 -f "VLLM::EngineCore" 2>/dev/null && echo "✅ vLLM EngineCore killed" || echo "ℹ️  No vLLM EngineCore processes"
pkill -9 -f "vllm" 2>/dev/null && echo "✅ vLLM processes killed" || echo "ℹ️  No vLLM processes"

# Kill uvicorn processes
echo -e "${BLUE}Killing uvicorn processes...${NC}"
pkill -9 -f "uvicorn" 2>/dev/null && echo "✅ Uvicorn processes killed" || echo "ℹ️  No uvicorn processes"

# Kill any Python processes using GPU
echo -e "${BLUE}Killing GPU Python processes...${NC}"
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | while read pid; do
    if [ -n "$pid" ]; then
        echo "Killing GPU process: $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# Clear CUDA cache
echo -e "${BLUE}Clearing CUDA cache...${NC}"
python3 -c "
import torch
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('✅ CUDA cache cleared')
    else:
        print('ℹ️  CUDA not available')
except Exception as e:
    print(f'⚠️  Could not clear CUDA cache: {e}')
" 2>/dev/null || echo "⚠️  Could not run Python CUDA cleanup"

# Wait a moment for cleanup
sleep 2

# Show final GPU status
echo -e "\n${GREEN}🎯 Final GPU Status:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read used total; do
    echo "GPU Memory: ${used}MB / ${total}MB used"
done

echo -e "\n${GREEN}✅ GPU cleanup complete!${NC}"
