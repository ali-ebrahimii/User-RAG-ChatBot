set -euo pipefail

ENV_FILE="${ENV_FILE:-./secure_data.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: ENV_FILE not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

export HF_HOME="${HF_HOME:-/opt/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/opt/hf/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/opt/hf/hub}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

MODEL_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is missing (set it in $ENV_FILE or as env var)." >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

python - <<'PY'
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'")
if not token:
    raise RuntimeError("HF_TOKEN is missing")

login(token=token, add_to_git_credential=False)
print("HuggingFace login OK")
PY

echo "Starting vLLM..."
echo "MODEL_ID=$MODEL_ID"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "HOST=$VLLM_HOST PORT=$VLLM_PORT"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"

exec python -m vllm.entrypoints.openai.api_server \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --download-dir "$HF_HOME" \
  --dtype auto \
  --quantization awq \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --enable-prefix-caching \
  --log-level info

