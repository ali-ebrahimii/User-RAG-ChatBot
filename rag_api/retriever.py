import os
import json
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import SYN_ENABLED, SYN_EXPAND_MAX_TERMS, SYNONYMS_PATH
from .logging_config import get_logger

logger = get_logger(__name__)

_FA_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

_SEMANTIC_FALLBACK_DEFAULT = True
_SEMANTIC_TOP_KEYS_DEFAULT = 4
_SEMANTIC_MIN_SIM_DEFAULT = 0.40

# Intents that are often vague -> disable synonym/semantic to avoid drift
_DISABLE_SYNONYMS_FOR_INTENTS = {
    "onboarding_start",
    "green_policy_faq",   # very generic bucket; expansion often hurts
}

# For these intents, allow synonyms but do NOT do semantic fallback (to reduce hallucinated anchors)
_DISABLE_SEMANTIC_FOR_INTENTS = {
    "onboarding_start",
}

# For these intents, allow a bit more semantic help
_ALLOW_SEMANTIC_FOR_INTENTS = {
    "appointment_booking",
    "appointment_cancel_edit",
    "auth_login",
    "payment_billing",
    "e_prescription",
    "support_contact",
    "privacy_terms",
    "insurance_info",
}


def normalize_fa(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace("ي", "ی").replace("ك", "ک")
    t = t.replace("\u200c", " ")
    t = _FA_DIACRITICS.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _fa_tokens(text: str) -> List[str]:
    t = normalize_fa(text)
    return t.split() if t else []


def load_synonyms(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        out: Dict[str, List[str]] = {}
        for k, vals in raw.items():
            nk = normalize_fa(str(k))
            if not nk:
                continue
            uniq = []
            seen = set([nk])
            for v in (vals or []):
                nv = normalize_fa(str(v))
                if nv and nv not in seen:
                    seen.add(nv)
                    uniq.append(nv)
            if uniq:
                out[nk] = uniq
        return out
    except FileNotFoundError:
        logger.warning("Synonyms file not found: %s", path)
        return {}
    except Exception as e:
        logger.exception("Failed to load synonyms from %s: %s", path, str(e))
        return {}


def _semantic_settings_for_intent(intent: Optional[str]) -> Tuple[bool, int, float]:
    """
    Decide whether semantic synonym fallback should run for this intent.
    """
    if not intent:
        return (_SEMANTIC_FALLBACK_DEFAULT, _SEMANTIC_TOP_KEYS_DEFAULT, _SEMANTIC_MIN_SIM_DEFAULT)

    if intent in _DISABLE_SEMANTIC_FOR_INTENTS:
        return (False, _SEMANTIC_TOP_KEYS_DEFAULT, _SEMANTIC_MIN_SIM_DEFAULT)

    if intent in _ALLOW_SEMANTIC_FOR_INTENTS:
        # allow semantic help with default thresholds
        return (True, _SEMANTIC_TOP_KEYS_DEFAULT, _SEMANTIC_MIN_SIM_DEFAULT)

    # unknown intents: keep semantic off by default to be safe
    return (False, _SEMANTIC_TOP_KEYS_DEFAULT, _SEMANTIC_MIN_SIM_DEFAULT)


def expand_query_with_synonyms(
    query: str,
    syn_map: Dict[str, List[str]],
    max_terms: int,
    *,
    intent: Optional[str] = None,
    query_emb: Optional[np.ndarray] = None,
    syn_keys: Optional[List[str]] = None,
    syn_key_embs: Optional[np.ndarray] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (variants, injected_terms).

    Improvements vs old:
      - intent-aware disabling/limiting
      - token-based matching (not only substring)
      - semantic fallback is controlled by intent + stricter threshold
    """
    q0 = normalize_fa(query)
    if not q0 or not syn_map:
        return [query], []

    # Disable synonyms for vague intents (prevents drift)
    if intent in _DISABLE_SYNONYMS_FOR_INTENTS:
        return [query], []

    injected: List[str] = []

    # Token/substring matching
    q_tokens = set(_fa_tokens(q0))
    for key, vals in syn_map.items():
        key_tokens = set(_fa_tokens(key))
        token_hit = bool(q_tokens & key_tokens)
        substr_hit = (key and key in q0)

        if token_hit or substr_hit:
            for v in vals:
                if v in q0:
                    continue
                injected.append(v)
                if len(injected) >= max_terms:
                    break

        if len(injected) >= max_terms:
            break

    # Semantic fallback
    sem_enabled, sem_top, sem_min_sim = _semantic_settings_for_intent(intent)
    if (not injected) and sem_enabled and query_emb is not None and syn_keys and syn_key_embs is not None:
        try:
            qv = query_emb[0]
            sims = syn_key_embs @ qv  # cosine similarities (normalized vectors)
            top_idx = np.argsort(-sims)[: max(1, sem_top)]

            for idx in top_idx:
                sim = float(sims[int(idx)])
                if sim < sem_min_sim:
                    continue
                key = syn_keys[int(idx)]
                vals = syn_map.get(key, [])
                for v in vals:
                    if v in q0:
                        continue
                    injected.append(v)
                    if len(injected) >= max_terms:
                        break
                if len(injected) >= max_terms:
                    break
        except Exception as e:
            logger.exception("Semantic synonym fallback failed: %s", str(e))

    if not injected:
        return [query], []

    expanded = (q0 + " " + " ".join(injected)).strip()
    return [query, expanded], injected


class DenseRetriever:
    def __init__(self, artifact_dir: str, device: Optional[str] = None):
        self.chunks_path = os.path.join(artifact_dir, "chunks.jsonl")
        self.faiss_path = os.path.join(artifact_dir, "faiss.index")
        self.meta_path = os.path.join(artifact_dir, "meta.json")

        for p in [self.chunks_path, self.faiss_path, self.meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing artifact: {p} (run build_artifacts first)")

        self.meta = json.load(open(self.meta_path, "r", encoding="utf-8"))

        self.all_chunks: List[Dict] = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.all_chunks.append(json.loads(line))

        self.index = faiss.read_index(self.faiss_path)

        # Embedder
        self.embedder = SentenceTransformer(self.meta["embed_model"], device=device)

        # Dim check
        test = self.embedder.encode(
            ["query: test"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        if test.shape[1] != self.index.d:
            raise RuntimeError(f"Embedding dim {test.shape[1]} != index dim {self.index.d}. Rebuild artifacts.")

        # Synonyms
        self.syn_enabled = SYN_ENABLED
        self.syn_map = load_synonyms(SYNONYMS_PATH) if self.syn_enabled else {}

        self._syn_keys: List[str] = []
        self._syn_key_embs: Optional[np.ndarray] = None

        if self.syn_enabled and self.syn_map:
            self._syn_keys = list(self.syn_map.keys())
            try:
                key_texts = [f"query: {k}" for k in self._syn_keys]
                embs = self.embedder.encode(
                    key_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")
                if embs.ndim == 2 and embs.shape[1] == self.index.d:
                    self._syn_key_embs = embs
            except Exception as e:
                logger.exception("Failed to precompute synonym key embeddings: %s", str(e))
                self._syn_key_embs = None

        logger.info(
            "DenseRetriever ready. synonyms=%s synonyms_terms=%s semantic_fallback=%s",
            self.syn_enabled,
            len(self.syn_map),
            bool(self._syn_key_embs is not None),
        )

    def _embed_query(self, query: str) -> np.ndarray:
        q = (query or "").strip()
        if not q:
            return np.zeros((1, self.index.d), dtype="float32")
        q_text = f"query: {q}"
        return self.embedder.encode(
            [q_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def search(self, query: str, k: int = 6, intent: Optional[str] = None) -> List[Dict]:
        """
        intent-aware retrieval:
          - disables synonyms for vague intents
          - controls semantic synonym fallback by intent
        """
        variants = [query]
        injected: List[str] = []

        q_emb_for_syn: Optional[np.ndarray] = None
        if self.syn_enabled and self.syn_map:
            q_emb_for_syn = self._embed_query(query)
            # In vague intents, max_terms effectively irrelevant because expansion is disabled
            variants, injected = expand_query_with_synonyms(
                query,
                self.syn_map,
                SYN_EXPAND_MAX_TERMS,
                intent=intent,
                query_emb=q_emb_for_syn,
                syn_keys=self._syn_keys,
                syn_key_embs=self._syn_key_embs,
            )

        if injected:
            logger.info(
                "Synonym expansion applied. intent=%s original=%r injected=%s variants=%s",
                intent, query, injected, variants
            )
        else:
            logger.debug("No synonym injection. intent=%s query=%r", intent, query)

        best: Dict[int, float] = {}
        for qv in variants:
            q_emb = q_emb_for_syn if (q_emb_for_syn is not None and qv == query) else self._embed_query(qv)
            scores, idxs = self.index.search(q_emb, k)

            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                idx = int(idx)
                s = float(score)
                if (idx not in best) or (s > best[idx]):
                    best[idx] = s

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for idx, score in ranked:
            ch = self.all_chunks[idx]
            out.append({
                "score": float(score),
                "source": ch["source"],
                "chunk_id": ch["chunk_id"],
                "text": ch["text"],
            })
        return out

