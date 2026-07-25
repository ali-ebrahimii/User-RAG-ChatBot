from typing import Optional, Tuple
import re

from fastapi import FastAPI, HTTPException
from .config import ARTIFACT_DIR, TOP_K, MIN_SCORE, SCORE_GAP
from .logging_config import setup_logging, get_logger
from .middleware import RequestResponseLoggingMiddleware
from .schemas import AskRequest, AskResponse
from .intent import classify_intent_fa_v2, FALLBACK_CONTACT_HTML, MEDICAL_REDIRECT_HTML
from .retriever import DenseRetriever, normalize_fa
from .prompts import build_messages, extract_answer_line
from .llm_client import wait_for_vllm, vllm_chat_completion
from .health import health_payload

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Public User ChatBot RAG API", version="1.0")
app.add_middleware(RequestResponseLoggingMiddleware, header_name="X-Request-ID")

retriever: Optional[DenseRetriever] = None

_REWRITE_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(از\s*کجا)\s*(باید\s*)?(شروع\s*کنم|شروع\s*کنیم)\??"), "مسیر دریافت نوبت ویزیت آنلاین راهنما"),
    (re.compile(r"(چطور|چجوری)\s*(شروع\s*کنم|شروع\s*کنیم)\??"), "مسیر دریافت نوبت ویزیت آنلاین راهنما"),
    (re.compile(r"(راهنمایی|کمک)\s*(میخوام|می‌خوام)?\??"), "راهنمای استفاده از سامانه دریافت نوبت ویزیت آنلاین"),
    (re.compile(r"(چی\s*کار\s*کنم|چیکار\s*کنم)\??"), "راهنمای دریافت نوبت ویزیت آنلاین"),
    (re.compile(r"(از\s*کجا\s*شروع)\??"), "مسیر دریافت نوبت ویزیت آنلاین راهنما"),
)

def rewrite_query_fa(q: str) -> Tuple[str, str]:
    """
    Returns (rewritten_query, applied_rule).
    If no rewrite, returns (original, "").
    """
    q0 = normalize_fa(q)
    if not q0:
        return q, ""

    for i, (pat, replacement) in enumerate(_REWRITE_RULES, start=1):
        if pat.search(q0):
            return replacement, f"rule_{i}"

    tokens = q0.split()
    if len(tokens) <= 3 and any(t in q0 for t in ["راهنما", "کمک", "شروع", "چطور", "چجوری", "چیکار"]):
        return "مسیر دریافت نوبت ویزیت آنلاین راهنما", "generic_short"

    return q, ""


def retrieval_is_trustworthy(query: str, hits, *, strict: bool = False) -> bool:
    if not hits:
        return False

    q0 = normalize_fa(query)
    n_tokens = len(q0.split()) if q0 else 0

    min_score = MIN_SCORE
    score_gap = SCORE_GAP

    # Short queries naturally have lower similarity and smaller gaps → relax
    if n_tokens <= 4:
        min_score = min(MIN_SCORE, 0.10)
        score_gap = 0.0
    
    if strict:
        min_score = max(min_score, 0.18)

    if hits[0]["score"] < min_score:
        return False

    if len(hits) >= 2 and (hits[0]["score"] - hits[1]["score"] < score_gap):
        return False

    return True


def _generate_answer(user_question: str, hits):
    messages = build_messages(user_question, hits)
    raw = vllm_chat_completion(messages, max_tokens=800)
    ans_line = extract_answer_line(raw)
    html = ans_line.replace("ANSWER:", "", 1).strip()
    return html


@app.on_event("startup")
def startup():
    global retriever
    wait_for_vllm(timeout_sec=180)
    retriever = DenseRetriever(ARTIFACT_DIR, device="cpu")
    logger.info("Retriever loaded. artifacts=%s", ARTIFACT_DIR)


@app.post("/answer", response_model=AskResponse)
def answer(req: AskRequest):
    global retriever
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not loaded")

    q = (req.question or "").strip()
    ir = classify_intent_fa_v2(q)
    intent = ir.intent
    anchor = (ir.anchor_query or "").strip()
    logger.info("Intent=%s conf=%.2f anchor=%r q=%r", intent, ir.confidence, anchor, q)

    # Policy: fixed fallbacks
    if intent in ["empty", "out_of_scope"]:
        return {"Answer": FALLBACK_CONTACT_HTML, "status": intent}

    if intent == "medical_advice":
        return {"Answer": MEDICAL_REDIRECT_HTML, "status": "medical_redirect"}

    # SPECIAL ROUTING: onboarding is vague -> prefer anchor FIRST
    if intent == "onboarding_start" and anchor:
        hits_a = retriever.search(anchor, k=TOP_K, intent=intent)
        logger.info("Anchor applied FIRST. intent=%s q=%r anchor=%r", intent, q, anchor)
        logger.info("Top hits (anchor-first): %s", hits_a[:3])

        if retrieval_is_trustworthy(anchor, hits_a, strict=True):
            html = _generate_answer(q, hits_a)
            return {"Answer": html, "status": "ok_anchor_first"}

    # Pass 1: normal retrieval
    hits = retriever.search(q, k=TOP_K, intent=intent)
    logger.info("Top hits (pass1): %s", hits[:3])

    if retrieval_is_trustworthy(q, hits):
        html = _generate_answer(q, hits)
        return {"Answer": html, "status": "ok"}

    # Pass 2 (fallback): intent anchor retrieval
    if anchor:
        hits_a = retriever.search(anchor, k=TOP_K, intent=intent)
        logger.info("Anchor applied. intent=%s q=%r anchor=%r", intent, q, anchor)
        logger.info("Top hits (anchor): %s", hits_a[:3])

        if retrieval_is_trustworthy(anchor, hits_a, strict=True):
            html = _generate_answer(q, hits_a)
            return {"Answer": html, "status": "ok_anchor"}

    # Pass 3: regex rewrite fallback
    q2, rule = rewrite_query_fa(q)
    if q2 != q:
        hits2 = retriever.search(q2, k=TOP_K, intent=intent)
        logger.info("Rewrite applied (%s). q=%r => q2=%r", rule, q, q2)
        logger.info("Top hits (rewrite): %s", hits2[:3])

        if retrieval_is_trustworthy(q2, hits2):
            html = _generate_answer(q, hits2)
            return {"Answer": html, "status": "ok_rewrite"}

    return {"Answer": FALLBACK_CONTACT_HTML, "status": "fallback"}


@app.get("/healthz")
def healthz():
    return health_payload()

