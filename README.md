# SEC RAG Assignment

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/anshuraj1/sec-rag-assignment)

Runnable RAG pipeline for answering structured questions on SEC 10-K filings (Apple & Tesla).  
Uses **LangChain, Chroma, FAISS, FlashRank, and Groq LLM**.

## How to Run
1. Attach Kaggle datasets:
   - `sec-dataset` → 10-K PDFs  
   - `sec-assignment` → contains `sec-assignment.json` with Groq API key
2. Enable **GPU** in Notebook settings.
3. Clone this repo and install dependencies:
   ```bash
   !git clone https://github.com/<your-username>/sec-rag-assignment.git
   %cd sec-rag-assignment
   !pip install -r requirements.txt
