import time
from typing import List, Dict
import requests
from .config import VLLM_BASE_URL, SERVED_MODEL_NAME, REQUEST_TIMEOUT
from .logging_config import get_logger

logger = get_logger(__name__)

def wait_for_vllm(timeout_sec: int = 180) -> None:
    t0 = time.time()
    url = f"{VLLM_BASE_URL}/v1/models"
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                logger.info("vLLM is up")
                return
        except Exception:
            pass
        if time.time() - t0 > timeout_sec:
            raise RuntimeError("vLLM did not start in time")
        time.sleep(2)



def vllm_chat_completion(messages: List[Dict], max_tokens: int = 800) -> str:
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }

    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        try:
            data = r.json()
        except ValueError:
            logger.error("Non-JSON vLLM response: status=%s body=%s", r.status_code, r.text)
            raise RuntimeError("vLLM returned non-JSON response")

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error("Unexpected vLLM response: %s", data)
            raise RuntimeError("Unexpected vLLM response format")

    except requests.exceptions.HTTPError as e:
        logger.exception(
            "vLLM HTTPError: status=%s body=%s",
            getattr(e.response, "status_code", None),
            getattr(e.response, "text", None),
        )
        raise
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logger.exception("vLLM connection error: %s", str(e))
        raise

