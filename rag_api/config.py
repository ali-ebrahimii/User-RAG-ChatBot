import os

def _as_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))

def _as_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts_dense")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
QWEN_ID = os.environ.get("QWEN_ID", "Qwen/Qwen2.5-7B-Instruct-AWQ")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", QWEN_ID)

TOP_K = max(1, _as_int("TOP_K", "6"))
MIN_SCORE = _as_float("MIN_SCORE", "0.15")
SCORE_GAP = _as_float("SCORE_GAP", "0.01")

MAX_CTX_CHARS = _as_int("MAX_CTX_CHARS", "9000")
REQUEST_TIMEOUT = _as_int("REQUEST_TIMEOUT", "120")

SYN_ENABLED = os.environ.get("SYN_ENABLED", "1").lower() in ("1", "true", "yes", "on")
SYN_EXPAND_MAX_TERMS = _as_int("SYN_EXPAND_MAX_TERMS", "10")

SYNONYMS_PATH = os.environ.get(
    "SYNONYMS_PATH",
    os.path.join(os.path.dirname(__file__), "synonyms_fa.json"),
)

