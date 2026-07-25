import os, json, hashlib
from typing import List, Dict
import numpy as np
import faiss
import docx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from huggingface_hub import login

token = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'")
if not token:
    raise RuntimeError("HF_TOKEN is missing")
login(token=token, add_to_git_credential=False)

DOCS_DIR = os.environ.get("DOCS_DIR", "./Documents")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts_dense")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CHUNKS_JSONL = os.path.join(ARTIFACT_DIR, "chunks.jsonl")
FAISS_INDEX  = os.path.join(ARTIFACT_DIR, "faiss.index")
META_JSON    = os.path.join(ARTIFACT_DIR, "meta.json")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large")
CHUNK_TOKENIZER_ID = os.environ.get("CHUNK_TOKENIZER_ID", "Qwen/Qwen2.5-7B-Instruct-AWQ")

CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "350"))
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", "80"))
if OVERLAP_TOKENS >= CHUNK_TOKENS:
    raise ValueError("OVERLAP_TOKENS must be < CHUNK_TOKENS")
EMBED_BATCH_SIZE = 8 

def docx_to_text(path: str) -> str:
    doc = docx.Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)

def load_all_docs(doc_dir: str) -> List[Dict]:
    out = []
    for fname in sorted(os.listdir(doc_dir)):
        if fname.lower().endswith(".docx"):
            full = os.path.join(doc_dir, fname)
            txt = docx_to_text(full).strip()
            if txt:
                out.append({"source": fname, "text": txt})
    return out

def fingerprint_docs(docs: List[Dict]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d["source"].encode("utf-8"))
        h.update(b"\n")
        h.update(d["text"].encode("utf-8", errors="ignore"))
        h.update(b"\n---\n")
    return h.hexdigest()

def token_chunk_text(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    chunks = []
    start = 0
    n = len(ids)
    while start < n:
        end = min(start + chunk_tokens, n)
        window = ids[start:end]
        ch = tokenizer.decode(window, skip_special_tokens=True).strip()
        if ch:
            chunks.append(ch)
        if end >= n:
            break
        start = max(0, end - overlap_tokens)
    
    if not chunks:
        raise RuntimeError("No chunks produced. Check DOCS_DIR or chunking params.")

    return chunks

def embed_passages(embedder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    emb = embedder.encode(
        prefixed,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    return emb

def build(force_rebuild: bool = False):
    docs = load_all_docs(DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No .docx in {DOCS_DIR}")

    fp = fingerprint_docs(docs)

    if (not force_rebuild) and os.path.exists(META_JSON) and os.path.exists(CHUNKS_JSONL) and os.path.exists(FAISS_INDEX):
        meta = json.load(open(META_JSON, "r", encoding="utf-8"))
        if (
            meta.get("doc_fingerprint") == fp
            and meta.get("embed_model") == EMBED_MODEL_NAME
            and meta.get("chunking", {}).get("chunk_tokens") == CHUNK_TOKENS
            and meta.get("chunking", {}).get("overlap_tokens") == OVERLAP_TOKENS
            and meta.get("chunk_tokenizer") == CHUNK_TOKENIZER_ID
        ):
            print("Artifacts already up-to-date.")
            return

    print("Building artifacts...")

    chunk_tok = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER_ID, use_fast=True)

    chunks = []
    for d in docs:
        parts = token_chunk_text(d["text"], chunk_tok, CHUNK_TOKENS, OVERLAP_TOKENS)
        for i, ch in enumerate(parts):
            chunks.append({"source": d["source"], "chunk_id": i, "text": ch})

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    embs = embed_passages(embedder, [c["text"] for c in chunks], EMBED_BATCH_SIZE)

    index = faiss.IndexFlatIP(embs.shape[1]) # If data grows big (100k+ chunks), we might move to IVF/HNSW.
    index.add(embs)
    faiss.write_index(index, FAISS_INDEX)

    meta = {
        "doc_fingerprint": fp,
        "embed_model": EMBED_MODEL_NAME,
        "faiss_dim": int(embs.shape[1]),
        "num_chunks": len(chunks),
        "chunking": {"chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS},
        "chunk_tokenizer": CHUNK_TOKENIZER_ID,
    }
    
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Done:", ARTIFACT_DIR)



if __name__ == "__main__":
    build(force_rebuild=False)

