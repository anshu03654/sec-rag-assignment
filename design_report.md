# SEC RAG Assignment – Concept Note

## 1. Chunking Strategy
SEC filings are long (200–300+ pages). To make them retrievable:
- **RecursiveCharacterTextSplitter** with `chunk_size = 700` and `overlap = 150`.
- Overlap ensures continuity of context between chunks.
- Metadata (page number, SEC item number, document name) is preserved and each chunk is assigned a unique ID for traceability.
- This allows retrieval at the right granularity (paragraph/section level) without loading full filings into memory.

## 2. LLM Choice
- **Generative model:** `openai/gpt-oss-20b` via **Groq API**.
  - Chosen for speed and structured JSON generation.
  - Runs deterministically with `temperature = 0.0` for factual answers.
- **Embedding model:** `Qwen/Qwen3-Embedding-0.6B`
  - Lightweight yet high-quality embeddings for financial/legal text.
  - Runs efficiently on GPU (Kaggle P100/T4).
- **Reranker:** `ms-marco-MiniLM-L-12-v2` with **FlashRank**
  - Improves relevance by combining dense vector search with keyword + pattern matching.

## 3. Out-of-Scope Handling
- Prompt template explicitly enforces:
  - “If the answer is missing or ambiguous, return a fixed JSON:  
    `{"answer":"This question cannot be answered based on the provided documents.","ref_ids":[],"reason":"<reason>"}`
  - Prevents hallucination by forcing strict JSON responses.
- Questions outside Apple/Tesla filings (e.g., Tesla HQ paint color, stock price forecast) are gracefully rejected with the **not-found message**.
- This ensures academic integrity and aligns with SEC analysis use case.

---

**Summary:**  
The system is designed as a reproducible **retrieval-augmented pipeline**. Chunking allows precise retrieval, the chosen LLM stack balances quality and performance, and strict handling of out-of-scope queries prevents hallucinations.
