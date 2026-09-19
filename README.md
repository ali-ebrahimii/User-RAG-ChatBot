# Persian User RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)
![Qwen](https://img.shields.io/badge/LLM-Qwen_2.5-6C5CE7)
![FAISS](https://img.shields.io/badge/Retrieval-FAISS-0467DF)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Pre--production-orange)

A production-oriented Retrieval-Augmented Generation (RAG) assistant designed to answer Persian user questions from approved, domain-specific sources.

The project was developed in the context of **Saman Salamat (Saman Insurance Group)** for a health-insurance platform. It combines semantic retrieval with a Persian-capable large language model to provide useful answers while reducing hallucinations and keeping responses grounded in company-approved documents.

> This public repository is a sanitized portfolio snapshot. It includes approved reference documents and generated demo artifacts needed to inspect retrieval behavior. Credentials, private datasets, infrastructure configuration, and proprietary deployment files are excluded.

## The Problem

Important user information was distributed across multiple Persian resources, including:

- Frequently asked questions
- Privacy policies
- Terms and conditions
- User guidance documents
- Health-related PDF documents

A general-purpose chatbot could respond fluently, but it could also generate unsupported or outdated information. In a health-insurance setting, a confident but incorrect answer is not acceptable.

The goal was therefore to build an assistant that:

1. retrieves the most relevant information from approved sources;
2. answers in clear Persian using only the retrieved context;
3. avoids unsupported claims; and
4. returns a safe fallback when the available evidence is insufficient.

## Key Capabilities

- Persian text extraction, cleaning, and normalization
- Configurable document chunking and metadata preservation
- Dense semantic retrieval using a FAISS vector index
- Context-grounded response generation with Qwen 2.5
- Confidence-based rejection for low-quality retrieval
- Safe fallback when an answer is not present in the sources
- REST API integration for web and mobile clients
- Separate API and model-serving layers
- Rebuildable retrieval artifacts when source documents change
- Separate API and model-serving layers designed for deployment

## System Architecture

```mermaid
flowchart TD
    A["Approved Persian documents"] --> B["Extraction and normalization"]
    B --> C["Chunking and dense embeddings"]
    C --> D["FAISS vector index"]
    E["User question"] --> F["FastAPI RAG service"]
    D --> F
    F --> G["Qwen 2.5 served with vLLM"]
    G --> H["Grounded answer or safe fallback"]
```

The retrieval layer stores its generated artifacts in a dedicated directory, including the processed chunks, FAISS index, and index metadata. These artifacts can be rebuilt whenever documents are added or updated.

## How It Works

1. **Ingestion:** Approved documents are collected and their Persian text is extracted.
2. **Normalization:** Arabic/Persian character variants, spacing, punctuation, and common text noise are normalized.
3. **Indexing:** Documents are divided into meaningful chunks and converted into dense embeddings.
4. **Retrieval:** The user question is normalized and matched against the FAISS index.
5. **Validation:** Retrieval scores are checked before context is passed to the language model.
6. **Generation:** Qwen 2.5 produces a Persian answer constrained by the retrieved evidence.
7. **Fallback:** If the system cannot find adequate evidence, it reports that the answer is not available in the current sources instead of guessing.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| API layer | FastAPI, Pydantic, REST |
| Language model | Qwen 2.5 Instruct |
| Model serving | vLLM |
| Retrieval | Dense embeddings, FAISS |
| Text processing | Persian NLP normalization, OCR preprocessing |
| Deployment | Docker and Docker Compose in the original deployment; private deployment files are omitted here |
| Data artifacts | JSONL chunks, FAISS index, metadata |
| Testing | API, retrieval, grounding, and fallback checks |

## Run the Public Snapshot

This demo is intended for a Linux machine with an NVIDIA GPU capable of serving the default AWQ model. The model server, API, and UI run as separate processes.

1. Create two Python environments if you want to isolate the GPU model server from the API dependencies.
2. Install API dependencies:

```bash
python -m pip install -r requirements.api.txt
```

3. Copy the example model-server configuration and add your own Hugging Face token:

```bash
cp secure_data.env.example secure_data.env
```

4. Install the vLLM dependencies from `requirements.vllm.txt`, then start the OpenAI-compatible model server:

```bash
bash run_vllm.sh
```

5. In another terminal, start the RAG API on port 8080:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000 \
uvicorn rag_api.main:app --host 0.0.0.0 --port 8080
```

6. Optionally start the Streamlit interface:

```bash
API_BASE_URL=http://127.0.0.1:8080 \
streamlit run ui/streamlit_app.py
```

Health check: `GET http://127.0.0.1:8080/healthz`

> Model downloads are large and GPU requirements depend on the selected model and quantization. Never commit `secure_data.env` or access tokens.

## Main Engineering Challenges

| Challenge | Approach |
|---|---|
| Persian spelling and character variation | Added Persian-aware normalization before indexing and querying |
| Semantically similar questions with different wording | Used dense vector retrieval rather than exact keyword matching |
| LLM hallucination | Restricted generation to retrieved evidence and added an explicit no-answer fallback |
| Weak or irrelevant retrieval results | Introduced a configurable confidence threshold before generation |
| Updating the knowledge base | Separated source ingestion from generated retrieval artifacts |
| Reproducible deployment | Separated the API and model-serving components; the original deployment was containerized |

## My Contribution

I designed and implemented the core workflow end to end, including:

- analyzing the product and user-support requirements;
- preparing and normalizing Persian source documents;
- designing the chunking, embedding, and retrieval pipeline;
- building and validating the FAISS-based dense index;
- integrating Qwen 2.5 with the RAG service;
- developing the API layer and response guardrails;
- containerizing the application and preparing it for technical handoff; and
- evaluating retrieval quality, grounded responses, and failure cases.

This work was connected to my broader responsibilities in health-data and business analytics, including large-scale data collection from Darmanet and other health-related sources, Python pipelines for data cleaning and normalization, medical PDF processing with OCR/NLP, and evaluation of Persian language models such as Qwen, Llama, and mT5.

## Project Status

The system reached a **working pre-production prototype** with a testable API and deployment-oriented structure. It was evaluated internally and prepared for technical review and handoff.

It is not presented here as a publicly deployed production service. This snapshot includes approved reference documents and generated artifacts for technical inspection; credentials, private datasets, infrastructure details, and proprietary deployment files remain excluded.

## Design Principles

- **Grounded over fluent:** a supported answer is more valuable than an impressive guess.
- **Safe by default:** missing evidence should produce a transparent fallback.
- **Persian-aware:** normalization and retrieval are designed for real Persian user input.
- **Modular:** ingestion, retrieval, generation, and API serving can evolve independently.
- **Maintainable:** knowledge-base artifacts can be rebuilt without redesigning the service.

## Author

**Ali Ebrahimi**  
Data Analyst & AI Specialist  
PhD Candidate in Electrical Engineering - Electronics, University of Tehran

- [GitHub](https://github.com/ali-ebrahimii)
- [LinkedIn](https://www.linkedin.com/in/ali-ebrahimii/)

---

For interview discussions, this repository demonstrates my experience in **RAG architecture, Persian NLP, LLM integration, API development, retrieval evaluation, guardrails, and containerized AI services**.
