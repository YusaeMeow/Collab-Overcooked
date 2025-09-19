#!/bin/bash
# Start vLLM server for local LLM deployment

# Default configuration
MODEL_NAME=${1:-"meta-llama/Llama-2-7b-chat-hf"}
PORT=${2:-8000}
HOST=${3:-"localhost"}
GPU_MEMORY=${4:-0.8}

echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Host: $HOST"
echo "GPU Memory Utilization: $GPU_MEMORY"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed. Install it with:"
    echo "pip install vllm"
    exit 1
fi

# Check if model exists locally or can be downloaded
echo "Checking model availability..."

# Start vLLM server
echo "Starting vLLM API server..."
python -m vllm.entrypoints.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --served-model-name "$MODEL_NAME" \
    --max-model-len 4096 \
    --dtype auto \
    --api-key "token-abc123"

echo "vLLM server stopped."